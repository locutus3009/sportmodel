mod analysis;
mod domain;
mod error;
mod excel;
mod formulas;
mod gp;
mod server;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{Duration, Local, NaiveDate};
use clap::Parser;

use crate::analysis::{
    MovementAnalysis, analyze_training_data, find_most_reliable_date_olympic,
    find_most_reliable_date_powerlifting,
};
use crate::domain::Movement;
use crate::excel::load_training_data;
use crate::formulas::{calculate_ipf_gl, calculate_sinclair};
use crate::server::{AppState, CompositeAnalysis, CompositePrediction};

/// Strength training analytics tool for Olympic weightlifting and powerlifting.
#[derive(Parser, Debug)]
#[command(name = "sportmodel")]
#[command(about = "Personal strength training analytics with Gaussian Process regression")]
#[command(version)]
struct Args {
    /// Path to the Excel file containing training data
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// Port number for the web server
    #[arg(value_name = "PORT")]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Parse command line arguments
    let args = Args::parse();

    // Load training data
    println!("Loading training data from: {}", args.file.display());
    let data = load_training_data(&args.file)
        .with_context(|| format!("Failed to load training data from {}", args.file.display()))?;

    // Print summary
    println!();
    println!("=== Training Data Summary ===");
    println!();
    println!("Total observations: {}", data.total_count());

    if let Some((min_date, max_date)) = data.overall_date_range() {
        println!("Date range: {} to {}", min_date, max_date);
    }

    println!();

    // Per-movement stats
    for movement in Movement::all() {
        let count = data.count(*movement);
        if count > 0 {
            let range = data.date_range(*movement).unwrap();
            println!(
                "{:15} {:4} entries  ({} to {})",
                movement.display_name(),
                count,
                range.0,
                range.1
            );
        }
    }

    // Run GP analysis
    println!();
    println!("=== Running GP Analysis ===");

    let today = Local::now().date_naive();

    // Set prediction range: 12 months ago to 6 months future
    let prediction_start = today - Duration::days(365);
    let prediction_end = today + Duration::days(180);

    let analysis_results = analyze_training_data(&data, prediction_start, prediction_end);

    // Count movements with predictions
    let movements_with_predictions = analysis_results
        .values()
        .filter(|a| a.has_predictions())
        .count();
    println!(
        "Movements with GP predictions: {}",
        movements_with_predictions
    );

    // Calculate composite indices
    println!();
    println!("=== Calculating Composite Indices ===");

    let ipf_gl = calculate_ipf_gl_series(&analysis_results, prediction_start, prediction_end);
    let sinclair = calculate_sinclair_series(&analysis_results, prediction_start, prediction_end);

    if ipf_gl.is_some() {
        println!(
            "IPF GL: computed ({} prediction days)",
            ipf_gl.as_ref().unwrap().predictions.len()
        );
    } else {
        println!("IPF GL: insufficient data (need squat, bench, deadlift, bodyweight)");
    }

    if sinclair.is_some() {
        println!(
            "Sinclair: computed ({} prediction days)",
            sinclair.as_ref().unwrap().predictions.len()
        );
    } else {
        println!("Sinclair: insufficient data (need snatch, C&J, bodyweight)");
    }

    // Build application state
    let state = Arc::new(AppState {
        training_data: data,
        analyses: analysis_results,
        ipf_gl,
        sinclair,
    });

    // Determine static directory (relative to executable or cwd)
    let static_dir = find_static_dir()?;
    println!();
    println!("Static files: {}", static_dir.display());

    // Start server
    println!();
    server::run_server(state, args.port, static_dir).await?;

    Ok(())
}

/// Finds the static directory for serving frontend files.
fn find_static_dir() -> Result<PathBuf> {
    // Try relative to current working directory
    let cwd_static = PathBuf::from("static");
    if cwd_static.is_dir() {
        return Ok(cwd_static);
    }

    // Try relative to executable
    if let Ok(exe_path) = std::env::current_exe()
        && let Some(exe_dir) = exe_path.parent()
    {
        let exe_static = exe_dir.join("static");
        if exe_static.is_dir() {
            return Ok(exe_static);
        }
    }

    // Default to cwd/static (will be created)
    Ok(cwd_static)
}

/// Calculates IPF GL time series from individual lift analyses.
fn calculate_ipf_gl_series(
    analyses: &HashMap<Movement, MovementAnalysis>,
    prediction_start: NaiveDate,
    prediction_end: NaiveDate,
) -> Option<CompositeAnalysis> {
    let squat = analyses.get(&Movement::Squat)?;
    let bench = analyses.get(&Movement::Bench)?;
    let deadlift = analyses.get(&Movement::Deadlift)?;
    let bodyweight = analyses.get(&Movement::Bodyweight)?;

    // Need predictions for all components
    if !squat.has_predictions()
        || !bench.has_predictions()
        || !deadlift.has_predictions()
        || !bodyweight.has_predictions()
    {
        return None;
    }

    // Find most reliable date
    let squat_dates: Vec<NaiveDate> = squat.data_points.iter().map(|p| p.date).collect();
    let bench_dates: Vec<NaiveDate> = bench.data_points.iter().map(|p| p.date).collect();
    let deadlift_dates: Vec<NaiveDate> = deadlift.data_points.iter().map(|p| p.date).collect();
    let most_reliable_date =
        find_most_reliable_date_powerlifting(&squat_dates, &bench_dates, &deadlift_dates);

    // Generate predictions for each day
    let mut predictions = Vec::new();
    let mut current = prediction_start;

    while current <= prediction_end {
        let squat_pred = squat.prediction_for(current);
        let bench_pred = bench.prediction_for(current);
        let deadlift_pred = deadlift.prediction_for(current);
        let bw_pred = bodyweight.prediction_for(current);

        if let (Some(s), Some(b), Some(d), Some(bw)) =
            (squat_pred, bench_pred, deadlift_pred, bw_pred)
            && let Some(gl) = calculate_ipf_gl(s.mean, b.mean, d.mean, bw.mean)
        {
            // Propagate uncertainty: use max relative std dev among components
            let rel_uncertainties = [
                s.std_dev / s.mean.abs().max(1.0),
                b.std_dev / b.mean.abs().max(1.0),
                d.std_dev / d.mean.abs().max(1.0),
            ];
            let max_rel_uncertainty = rel_uncertainties.iter().cloned().fold(0.0, f64::max);
            let gl_std = gl * max_rel_uncertainty;

            predictions.push(CompositePrediction {
                date: current,
                value: gl,
                ci_lower: gl - 1.96 * gl_std,
                ci_upper: gl + 1.96 * gl_std,
            });
        }

        current = current.succ_opt().unwrap_or(current);
        if current == prediction_start {
            break; // Overflow protection
        }
    }

    if predictions.is_empty() {
        return None;
    }

    // Get current value (today or most recent)
    let today = Local::now().date_naive();
    let current_value = predictions
        .iter()
        .rev()
        .find(|p| p.date <= today)
        .map(|p| p.value)
        .unwrap_or_else(|| predictions.first().map(|p| p.value).unwrap_or(0.0));

    Some(CompositeAnalysis {
        predictions,
        most_reliable_date,
        current_value,
    })
}

/// Calculates Sinclair time series from individual lift analyses.
fn calculate_sinclair_series(
    analyses: &HashMap<Movement, MovementAnalysis>,
    prediction_start: NaiveDate,
    prediction_end: NaiveDate,
) -> Option<CompositeAnalysis> {
    let snatch = analyses.get(&Movement::Snatch)?;
    let cj = analyses.get(&Movement::CleanAndJerk)?;
    let bodyweight = analyses.get(&Movement::Bodyweight)?;

    // Need predictions for all components
    if !snatch.has_predictions() || !cj.has_predictions() || !bodyweight.has_predictions() {
        return None;
    }

    // Find most reliable date
    let snatch_dates: Vec<NaiveDate> = snatch.data_points.iter().map(|p| p.date).collect();
    let cj_dates: Vec<NaiveDate> = cj.data_points.iter().map(|p| p.date).collect();
    let most_reliable_date = find_most_reliable_date_olympic(&snatch_dates, &cj_dates);

    // Generate predictions for each day
    let mut predictions = Vec::new();
    let mut current = prediction_start;

    while current <= prediction_end {
        let snatch_pred = snatch.prediction_for(current);
        let cj_pred = cj.prediction_for(current);
        let bw_pred = bodyweight.prediction_for(current);

        if let (Some(sn), Some(c), Some(bw)) = (snatch_pred, cj_pred, bw_pred)
            && let Some(sinclair) = calculate_sinclair(sn.mean, c.mean, bw.mean)
        {
            // Propagate uncertainty: use max relative std dev among components
            let rel_uncertainties = [
                sn.std_dev / sn.mean.abs().max(1.0),
                c.std_dev / c.mean.abs().max(1.0),
            ];
            let max_rel_uncertainty = rel_uncertainties.iter().cloned().fold(0.0, f64::max);
            let sinclair_std = sinclair * max_rel_uncertainty;

            predictions.push(CompositePrediction {
                date: current,
                value: sinclair,
                ci_lower: sinclair - 1.96 * sinclair_std,
                ci_upper: sinclair + 1.96 * sinclair_std,
            });
        }

        current = current.succ_opt().unwrap_or(current);
        if current == prediction_start {
            break; // Overflow protection
        }
    }

    if predictions.is_empty() {
        return None;
    }

    // Get current value (today or most recent)
    let today = Local::now().date_naive();
    let current_value = predictions
        .iter()
        .rev()
        .find(|p| p.date <= today)
        .map(|p| p.value)
        .unwrap_or_else(|| predictions.first().map(|p| p.value).unwrap_or(0.0));

    Some(CompositeAnalysis {
        predictions,
        most_reliable_date,
        current_value,
    })
}
