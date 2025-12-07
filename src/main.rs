mod domain;
mod error;
mod excel;
mod formulas;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

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

    println!();
    println!("Port {} reserved for web server (Phase 3)", args.port);
    println!();

    Ok(())
}
