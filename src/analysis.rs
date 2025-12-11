//! Analysis orchestration for training data.
//!
//! This module integrates GP regression with the domain model and provides
//! higher-level analysis functions.

use std::collections::HashMap;

use chrono::NaiveDate;
use rayon::prelude::*;

use crate::domain::{DataPoint, Movement, TrainingData};
use crate::gp::{GpHyperparameters, GpModel, GpPrediction};

/// Analysis results for a single movement.
#[derive(Debug)]
pub struct MovementAnalysis {
    #[allow(dead_code)] // Used in Phase 3
    pub movement: Movement,
    pub predictions: Vec<GpPrediction>,
    pub last_observation_date: Option<NaiveDate>,
    pub data_points: Vec<DataPoint>,
}

impl MovementAnalysis {
    /// Returns true if analysis was successful (has predictions).
    pub fn has_predictions(&self) -> bool {
        !self.predictions.is_empty()
    }

    /// Returns the most recent prediction.
    #[allow(dead_code)] // Used in Phase 3
    pub fn latest_prediction(&self) -> Option<&GpPrediction> {
        self.predictions.last()
    }

    /// Returns prediction for a specific date if available.
    pub fn prediction_for(&self, date: NaiveDate) -> Option<&GpPrediction> {
        self.predictions.iter().find(|p| p.date == date)
    }
}

/// Analyzes training data for a single movement.
///
/// Returns None if there's insufficient data for GP regression.
pub fn analyze_movement(
    movement: Movement,
    data: &[DataPoint],
    prediction_start: NaiveDate,
    prediction_end: NaiveDate,
) -> Option<MovementAnalysis> {
    if data.len() < 2 {
        return Some(MovementAnalysis {
            movement,
            predictions: Vec::new(),
            last_observation_date: data.first().map(|p| p.date),
            data_points: data.to_vec(),
        });
    }

    // Estimate hyperparameters from data
    let hyperparams = match GpHyperparameters::estimate_from_data(data) {
        Ok(hp) => hp,
        Err(_) => return None,
    };

    // Fit GP model
    let model = match GpModel::fit(data, hyperparams) {
        Ok(m) => m,
        Err(_) => return None,
    };

    // Generate predictions
    let predictions = model
        .predict_range(prediction_start, prediction_end)
        .unwrap_or_default();

    Some(MovementAnalysis {
        movement,
        predictions,
        last_observation_date: data.last().map(|p| p.date),
        data_points: data.to_vec(),
    })
}

/// Analyzes all movements in training data.
///
/// Returns a HashMap of movement to analysis results. Movements with
/// insufficient data will have empty predictions.
///
/// Uses parallel processing via rayon to analyze movements concurrently,
/// significantly reducing total analysis time on multi-core systems.
pub fn analyze_training_data(
    data: &TrainingData,
    prediction_start: NaiveDate,
    prediction_end: NaiveDate,
) -> HashMap<Movement, MovementAnalysis> {
    // Process movements in parallel using rayon
    Movement::all()
        .par_iter()
        .filter_map(|&movement| {
            let points = data.get(movement).unwrap_or(&[]);
            analyze_movement(movement, points, prediction_start, prediction_end)
                .map(|analysis| (movement, analysis))
        })
        .collect()
}

/// Finds the date when all component lifts were most recently measured.
///
/// This is useful for determining when composite indices (like IPF GL)
/// are most reliable - when all component lifts have recent measurements.
///
/// Returns None if any of the required lifts has no data.
#[allow(dead_code)] // Used in Phase 3
pub fn find_most_reliable_date_powerlifting(
    squat_dates: &[NaiveDate],
    bench_dates: &[NaiveDate],
    deadlift_dates: &[NaiveDate],
) -> Option<NaiveDate> {
    // Get the most recent date for each lift
    let squat_latest = squat_dates.iter().max()?;
    let bench_latest = bench_dates.iter().max()?;
    let deadlift_latest = deadlift_dates.iter().max()?;

    // Return the minimum of the three maximums
    // This is when all three lifts were "freshest"
    Some(
        *[squat_latest, bench_latest, deadlift_latest]
            .into_iter()
            .min()?,
    )
}

/// Finds the date when all Olympic lifts were most recently measured.
///
/// Returns None if either lift has no data.
#[allow(dead_code)] // Used in Phase 3
pub fn find_most_reliable_date_olympic(
    snatch_dates: &[NaiveDate],
    cj_dates: &[NaiveDate],
) -> Option<NaiveDate> {
    let snatch_latest = snatch_dates.iter().max()?;
    let cj_latest = cj_dates.iter().max()?;

    Some(*[snatch_latest, cj_latest].into_iter().min()?)
}

/// Computes staleness (days since measurement) for each lift at a given date.
#[allow(dead_code)] // Used in Phase 3
pub fn compute_staleness(dates: &[NaiveDate], at_date: NaiveDate) -> Option<i64> {
    let latest = dates.iter().max()?;
    if at_date >= *latest {
        Some((at_date - *latest).num_days())
    } else {
        // Date is before the latest measurement - find closest preceding date
        dates
            .iter()
            .filter(|&&d| d <= at_date)
            .max()
            .map(|d| (at_date - *d).num_days())
    }
}

/// Summary statistics for a movement's predictions.
#[derive(Debug)]
#[allow(dead_code)] // Used in Phase 3
pub struct PredictionSummary {
    pub movement: Movement,
    pub current_estimate: f64,
    pub current_ci_lower: f64,
    pub current_ci_upper: f64,
    pub days_since_last_obs: i64,
    pub trend_30d: Option<f64>, // Change over last 30 days
}

impl PredictionSummary {
    /// Creates a summary from a movement analysis at a specific date.
    #[allow(dead_code)] // Used in Phase 3
    pub fn from_analysis(analysis: &MovementAnalysis, at_date: NaiveDate) -> Option<Self> {
        let current = analysis.prediction_for(at_date)?;

        let days_since = analysis
            .last_observation_date
            .map(|d| (at_date - d).num_days())
            .unwrap_or(0);

        // Calculate 30-day trend if we have enough history
        let trend_30d = {
            let past_date = at_date - chrono::Duration::days(30);
            analysis
                .prediction_for(past_date)
                .map(|past_pred| current.mean - past_pred.mean)
        };

        Some(Self {
            movement: analysis.movement,
            current_estimate: current.mean,
            current_ci_lower: current.ci_lower(),
            current_ci_upper: current.ci_upper(),
            days_since_last_obs: days_since,
            trend_30d,
        })
    }
}

/// Attempts to fit a GP model with error handling.
///
/// Returns a descriptive error message if fitting fails.
#[allow(dead_code)] // Used in Phase 3
pub fn fit_with_diagnostics(movement: Movement, data: &[DataPoint]) -> Result<GpModel, String> {
    if data.is_empty() {
        return Err(format!("No data for {}", movement.display_name()));
    }

    if data.len() == 1 {
        return Err(format!(
            "Only 1 observation for {} - need at least 2 for GP regression",
            movement.display_name()
        ));
    }

    let hyperparams = GpHyperparameters::estimate_from_data(data).map_err(|e| {
        format!(
            "Failed to estimate hyperparameters for {}: {}",
            movement.display_name(),
            e
        )
    })?;

    GpModel::fit(data, hyperparams)
        .map_err(|e| format!("Failed to fit GP for {}: {}", movement.display_name(), e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    #[test]
    fn test_analyze_movement_insufficient_data() {
        let data = vec![DataPoint {
            date: make_date(2024, 1, 1),
            value: 100.0,
        }];

        let analysis = analyze_movement(
            Movement::Squat,
            &data,
            make_date(2024, 1, 1),
            make_date(2024, 3, 1),
        )
        .unwrap();

        assert!(!analysis.has_predictions());
        assert_eq!(analysis.data_points.len(), 1);
    }

    #[test]
    fn test_analyze_movement_sufficient_data() {
        let data = vec![
            DataPoint {
                date: make_date(2024, 1, 1),
                value: 100.0,
            },
            DataPoint {
                date: make_date(2024, 2, 1),
                value: 110.0,
            },
            DataPoint {
                date: make_date(2024, 3, 1),
                value: 115.0,
            },
        ];

        let analysis = analyze_movement(
            Movement::Squat,
            &data,
            make_date(2024, 1, 1),
            make_date(2024, 4, 1),
        )
        .unwrap();

        assert!(analysis.has_predictions());
        assert!(!analysis.predictions.is_empty());
    }

    #[test]
    fn test_find_most_reliable_date_powerlifting() {
        let squat = vec![make_date(2024, 1, 10), make_date(2024, 2, 5)];
        let bench = vec![make_date(2024, 1, 15), make_date(2024, 2, 10)];
        let dead = vec![make_date(2024, 1, 20), make_date(2024, 1, 25)];

        // Deadlift is oldest at 1/25, so that's the most reliable date
        let result = find_most_reliable_date_powerlifting(&squat, &bench, &dead);
        assert_eq!(result, Some(make_date(2024, 1, 25)));
    }

    #[test]
    fn test_find_most_reliable_date_missing_data() {
        let squat = vec![make_date(2024, 1, 10)];
        let bench = vec![];
        let dead = vec![make_date(2024, 1, 20)];

        let result = find_most_reliable_date_powerlifting(&squat, &bench, &dead);
        assert!(result.is_none());
    }

    #[test]
    fn test_compute_staleness() {
        let dates = vec![make_date(2024, 1, 10), make_date(2024, 2, 5)];

        // 10 days after latest observation
        let staleness = compute_staleness(&dates, make_date(2024, 2, 15));
        assert_eq!(staleness, Some(10));

        // Same day as latest observation
        let staleness = compute_staleness(&dates, make_date(2024, 2, 5));
        assert_eq!(staleness, Some(0));
    }

    #[test]
    fn test_fit_with_diagnostics_no_data() {
        let result = fit_with_diagnostics(Movement::Squat, &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No data"));
    }

    #[test]
    fn test_fit_with_diagnostics_one_point() {
        let data = vec![DataPoint {
            date: make_date(2024, 1, 1),
            value: 100.0,
        }];

        let result = fit_with_diagnostics(Movement::Squat, &data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Only 1 observation"));
    }

    #[test]
    fn test_fit_with_diagnostics_success() {
        let data = vec![
            DataPoint {
                date: make_date(2024, 1, 1),
                value: 100.0,
            },
            DataPoint {
                date: make_date(2024, 2, 1),
                value: 110.0,
            },
        ];

        let result = fit_with_diagnostics(Movement::Squat, &data);
        assert!(result.is_ok());
    }
}
