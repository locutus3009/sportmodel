mod analysis;
mod domain;
mod error;
mod excel;
mod formulas;
mod gp;

use anyhow::{Context, Result};
use chrono::{Duration, Local};
use clap::Parser;
use std::path::PathBuf;

use crate::analysis::analyze_training_data;
use crate::domain::Movement;
use crate::excel::load_training_data;

/// Strength training analytics tool for Olympic weightlifting and powerlifting.
#[derive(Parser, Debug)]
#[command(name = "sportmodel")]
#[command(about = "Personal strength training analytics with Gaussian Process regression")]
#[command(version)]
struct Args {
    /// Path to the Excel file containing training data
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// Port number for the web server (reserved for Phase 3)
    #[arg(value_name = "PORT")]
    port: u16,
}

fn main() -> Result<()> {
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

    // Overall stats
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
    println!("=== GP Regression Analysis ===");
    println!();

    if let Some((min_date, max_date)) = data.overall_date_range() {
        // Predict from first observation to 6 months after the last
        let prediction_end = max_date + Duration::days(180);
        let today = Local::now().date_naive();

        let analysis_results = analyze_training_data(&data, min_date, prediction_end);

        for movement in Movement::all() {
            if let Some(analysis) = analysis_results.get(movement) {
                if analysis.has_predictions() {
                    println!("{}:", movement.display_name());

                    // Show prediction at last observation date
                    if let Some(last_date) = analysis.last_observation_date
                        && let Some(pred) = analysis.prediction_for(last_date)
                    {
                        println!(
                            "  At last obs ({}): {:.1} kg  [95% CI: {:.1} - {:.1}]",
                            last_date,
                            pred.mean,
                            pred.ci_lower(),
                            pred.ci_upper()
                        );
                    }

                    // Show prediction for today (if in range)
                    if today >= min_date
                        && today <= prediction_end
                        && let Some(pred) = analysis.prediction_for(today)
                    {
                        let days_since = analysis
                            .last_observation_date
                            .map(|d| (today - d).num_days())
                            .unwrap_or(0);
                        println!(
                            "  Today ({}): {:.1} kg  [95% CI: {:.1} - {:.1}]  ({} days since last obs)",
                            today,
                            pred.mean,
                            pred.ci_lower(),
                            pred.ci_upper(),
                            days_since
                        );
                    }

                    // Show prediction 3 months out from last observation
                    if let Some(last_date) = analysis.last_observation_date {
                        let future_date = last_date + Duration::days(90);
                        if future_date <= prediction_end
                            && let Some(pred) = analysis.prediction_for(future_date)
                        {
                            println!(
                                "  +3 months ({}): {:.1} kg  [95% CI: {:.1} - {:.1}]",
                                future_date,
                                pred.mean,
                                pred.ci_lower(),
                                pred.ci_upper()
                            );
                        }
                    }

                    println!();
                } else if !analysis.data_points.is_empty() {
                    println!(
                        "{}: insufficient data ({} point(s))",
                        movement.display_name(),
                        analysis.data_points.len()
                    );
                    println!();
                }
            }
        }
    }

    println!("Port {} reserved for web server (Phase 3)", args.port);
    println!();

    Ok(())
}
