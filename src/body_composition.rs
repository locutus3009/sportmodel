//! Body composition calculations from anthropometric measurements.
//!
//! Calculates Body Fat Percentage (BF%) and Lean Body Mass (LBM) by:
//! 1. Computing exact values from matched measurements (neck+waist on same day)
//! 2. Fitting GP regression directly to these computed values
//! 3. Interpolating BF% and LBM directly (not through input measurements)
//!
//! This approach produces correct dynamics because the GP learns the actual
//! BF% and LBM trends rather than propagating uncertainty through a nonlinear formula.

use std::collections::HashMap;

use chrono::NaiveDate;

use crate::analysis::MovementAnalysis;
use crate::domain::{DataPoint, Movement};
use crate::gp::{optimize_noise_with_metadata, GpConfig, GpHyperparameters, GpModel};

/// Height constant for BF% calculation (in cm).
/// This is a user-specific constant that could be made configurable in the future.
const HEIGHT_CM: f64 = 180.0;

/// Calculates body fat percentage using the US Navy formula (men).
///
/// Formula (measurements in cm):
/// ```text
/// BF% = 495 / (1.0324 - 0.19077 × log10(waist - neck) + 0.15456 × log10(height)) - 450
/// ```
///
/// Returns None if waist <= neck (invalid measurement).
pub fn calculate_body_fat_pct(waist_cm: f64, neck_cm: f64) -> Option<f64> {
    let d = waist_cm - neck_cm;
    if d <= 0.0 {
        return None;
    }

    let a = 1.0324 - 0.19077 * d.log10() + 0.15456 * HEIGHT_CM.log10();
    if a <= 0.0 {
        return None;
    }

    Some(495.0 / a - 450.0)
}

/// Calculates Lean Body Mass from bodyweight and body fat percentage.
///
/// Formula:
/// ```text
/// LBM = bodyweight × (1 - BF% / 100)
/// ```
pub fn calculate_lbm(bodyweight_kg: f64, body_fat_pct: f64) -> f64 {
    bodyweight_kg * (1.0 - body_fat_pct / 100.0)
}

/// Calculates raw BF% data points from matched neck/waist measurements.
///
/// For each date where BOTH neck AND waist measurements exist,
/// computes the exact BF% value using the US Navy formula.
///
/// Returns data points sorted by date.
pub fn calculate_raw_body_fat_points(
    neck_points: &[DataPoint],
    waist_points: &[DataPoint],
) -> Vec<DataPoint> {
    // Build lookup of neck measurements by date
    let neck_by_date: HashMap<NaiveDate, f64> =
        neck_points.iter().map(|p| (p.date, p.value)).collect();

    let mut bf_points = Vec::new();

    for waist in waist_points {
        if let Some(&neck_value) = neck_by_date.get(&waist.date)
            && let Some(bf) = calculate_body_fat_pct(waist.value, neck_value)
        {
            bf_points.push(DataPoint {
                date: waist.date,
                value: bf,
            });
        }
    }

    // Sort by date
    bf_points.sort_by_key(|p| p.date);
    bf_points
}

/// Calculates raw LBM data points from matched bodyweight/neck/waist measurements.
///
/// For each date where ALL THREE measurements exist,
/// computes the exact LBM value.
///
/// Returns data points sorted by date.
pub fn calculate_raw_lbm_points(
    bodyweight_points: &[DataPoint],
    neck_points: &[DataPoint],
    waist_points: &[DataPoint],
) -> Vec<DataPoint> {
    // Build lookups by date
    let neck_by_date: HashMap<NaiveDate, f64> =
        neck_points.iter().map(|p| (p.date, p.value)).collect();
    let waist_by_date: HashMap<NaiveDate, f64> =
        waist_points.iter().map(|p| (p.date, p.value)).collect();

    let mut lbm_points = Vec::new();

    for bw in bodyweight_points {
        if let (Some(&neck_value), Some(&waist_value)) =
            (neck_by_date.get(&bw.date), waist_by_date.get(&bw.date))
            && let Some(bf) = calculate_body_fat_pct(waist_value, neck_value)
        {
            let lbm = calculate_lbm(bw.value, bf);
            lbm_points.push(DataPoint {
                date: bw.date,
                value: lbm,
            });
        }
    }

    // Sort by date
    lbm_points.sort_by_key(|p| p.date);
    lbm_points
}

/// Analyzes body fat percentage by fitting GP directly to computed BF% values.
///
/// This is the correct approach: we compute exact BF% from matched measurements,
/// then let the GP learn the BF% dynamics directly.
///
/// Uses body composition length scale and optimizes noise variance via log marginal likelihood.
pub fn analyze_body_fat(
    analyses: &HashMap<Movement, MovementAnalysis>,
    prediction_start: NaiveDate,
    prediction_end: NaiveDate,
    gp_config: &GpConfig,
) -> Option<MovementAnalysis> {
    let neck_analysis = analyses.get(&Movement::Neck)?;
    let waist_analysis = analyses.get(&Movement::Waist)?;

    // Calculate raw BF% data points from matched measurements
    let bf_points = calculate_raw_body_fat_points(
        &neck_analysis.data_points,
        &waist_analysis.data_points,
    );

    // Need at least 2 points for GP regression
    if bf_points.len() < 2 {
        return Some(MovementAnalysis {
            movement: Movement::Neck, // Placeholder, not used
            predictions: Vec::new(),
            last_observation_date: bf_points.first().map(|p| p.date),
            data_points: bf_points,
        });
    }

    let length_scale = gp_config.length_scale_body_composition();

    // Optimize noise variance via log marginal likelihood, fall back to heuristic
    let hyperparams = match optimize_noise_with_metadata(&bf_points, length_scale) {
        Ok(opt) => {
            log::info!(
                "Body Fat %: l={:.0}, noise_ratio={:.2}, log_lik={:.1} (n={})",
                length_scale as i32,
                opt.noise_ratio,
                opt.log_marginal_likelihood,
                bf_points.len()
            );
            opt.hyperparams
        }
        Err(_) => {
            log::warn!("Body Fat %: optimization failed, using heuristic");
            GpHyperparameters::estimate_from_data(&bf_points).ok()?
        }
    };

    // Fit GP model to BF% values directly
    let model = GpModel::fit(&bf_points, hyperparams).ok()?;

    // Generate predictions
    let predictions = model
        .predict_range(prediction_start, prediction_end)
        .unwrap_or_default();

    Some(MovementAnalysis {
        movement: Movement::Neck, // Placeholder, not used
        predictions,
        last_observation_date: bf_points.last().map(|p| p.date),
        data_points: bf_points,
    })
}

/// Analyzes lean body mass by fitting GP directly to computed LBM values.
///
/// This is the correct approach: we compute exact LBM from matched measurements,
/// then let the GP learn the LBM dynamics directly.
///
/// Uses body composition length scale and optimizes noise variance via log marginal likelihood.
pub fn analyze_lbm(
    analyses: &HashMap<Movement, MovementAnalysis>,
    prediction_start: NaiveDate,
    prediction_end: NaiveDate,
    gp_config: &GpConfig,
) -> Option<MovementAnalysis> {
    let bodyweight_analysis = analyses.get(&Movement::Bodyweight)?;
    let neck_analysis = analyses.get(&Movement::Neck)?;
    let waist_analysis = analyses.get(&Movement::Waist)?;

    // Calculate raw LBM data points from matched measurements
    let lbm_points = calculate_raw_lbm_points(
        &bodyweight_analysis.data_points,
        &neck_analysis.data_points,
        &waist_analysis.data_points,
    );

    // Need at least 2 points for GP regression
    if lbm_points.len() < 2 {
        return Some(MovementAnalysis {
            movement: Movement::Bodyweight, // Placeholder, not used
            predictions: Vec::new(),
            last_observation_date: lbm_points.first().map(|p| p.date),
            data_points: lbm_points,
        });
    }

    let length_scale = gp_config.length_scale_body_composition();

    // Optimize noise variance via log marginal likelihood, fall back to heuristic
    let hyperparams = match optimize_noise_with_metadata(&lbm_points, length_scale) {
        Ok(opt) => {
            log::info!(
                "LBM: l={:.0}, noise_ratio={:.2}, log_lik={:.1} (n={})",
                length_scale as i32,
                opt.noise_ratio,
                opt.log_marginal_likelihood,
                lbm_points.len()
            );
            opt.hyperparams
        }
        Err(_) => {
            log::warn!("LBM: optimization failed, using heuristic");
            GpHyperparameters::estimate_from_data(&lbm_points).ok()?
        }
    };

    // Fit GP model to LBM values directly
    let model = GpModel::fit(&lbm_points, hyperparams).ok()?;

    // Generate predictions
    let predictions = model
        .predict_range(prediction_start, prediction_end)
        .unwrap_or_default();

    Some(MovementAnalysis {
        movement: Movement::Bodyweight, // Placeholder, not used
        predictions,
        last_observation_date: lbm_points.last().map(|p| p.date),
        data_points: lbm_points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    #[test]
    fn test_body_fat_calculation() {
        // Typical measurement: waist 85cm, neck 40cm
        let bf = calculate_body_fat_pct(85.0, 40.0);
        assert!(bf.is_some());
        let bf = bf.unwrap();
        // Should be somewhere in the 15-25% range for typical male
        assert!(bf > 10.0 && bf < 30.0, "BF% = {}", bf);
    }

    #[test]
    fn test_body_fat_invalid_measurements() {
        // Waist less than neck - invalid
        assert!(calculate_body_fat_pct(35.0, 40.0).is_none());

        // Waist equal to neck - invalid
        assert!(calculate_body_fat_pct(40.0, 40.0).is_none());
    }

    #[test]
    fn test_lbm_calculation() {
        // 80kg bodyweight, 20% body fat -> 64kg lean mass
        let lbm = calculate_lbm(80.0, 20.0);
        assert!((lbm - 64.0).abs() < 0.01, "LBM = {}", lbm);
    }

    #[test]
    fn test_bf_sensitivity_to_waist() {
        // Increasing waist should increase BF%
        let bf1 = calculate_body_fat_pct(80.0, 40.0).unwrap();
        let bf2 = calculate_body_fat_pct(85.0, 40.0).unwrap();
        let bf3 = calculate_body_fat_pct(90.0, 40.0).unwrap();

        assert!(bf1 < bf2, "BF% should increase with waist: {} vs {}", bf1, bf2);
        assert!(bf2 < bf3, "BF% should increase with waist: {} vs {}", bf2, bf3);
    }

    #[test]
    fn test_bf_sensitivity_to_neck() {
        // Increasing neck should decrease BF%
        let bf1 = calculate_body_fat_pct(85.0, 38.0).unwrap();
        let bf2 = calculate_body_fat_pct(85.0, 40.0).unwrap();
        let bf3 = calculate_body_fat_pct(85.0, 42.0).unwrap();

        assert!(bf1 > bf2, "BF% should decrease with neck: {} vs {}", bf1, bf2);
        assert!(bf2 > bf3, "BF% should decrease with neck: {} vs {}", bf2, bf3);
    }

    #[test]
    fn test_calculate_raw_body_fat_points_matched_dates() {
        let neck_points = vec![
            DataPoint { date: make_date(2024, 1, 1), value: 38.0 },
            DataPoint { date: make_date(2024, 1, 15), value: 38.5 },
            DataPoint { date: make_date(2024, 2, 1), value: 39.0 },
        ];
        let waist_points = vec![
            DataPoint { date: make_date(2024, 1, 1), value: 90.0 },
            DataPoint { date: make_date(2024, 1, 10), value: 89.0 }, // No neck match
            DataPoint { date: make_date(2024, 2, 1), value: 88.0 },
        ];

        let bf_points = calculate_raw_body_fat_points(&neck_points, &waist_points);

        // Should only have 2 points (1/1 and 2/1)
        assert_eq!(bf_points.len(), 2);
        assert_eq!(bf_points[0].date, make_date(2024, 1, 1));
        assert_eq!(bf_points[1].date, make_date(2024, 2, 1));

        // BF% should be calculated correctly
        let expected_bf_jan1 = calculate_body_fat_pct(90.0, 38.0).unwrap();
        assert!((bf_points[0].value - expected_bf_jan1).abs() < 0.01);
    }

    #[test]
    fn test_calculate_raw_body_fat_points_no_matches() {
        let neck_points = vec![
            DataPoint { date: make_date(2024, 1, 1), value: 38.0 },
        ];
        let waist_points = vec![
            DataPoint { date: make_date(2024, 1, 2), value: 90.0 },
        ];

        let bf_points = calculate_raw_body_fat_points(&neck_points, &waist_points);

        assert!(bf_points.is_empty());
    }

    #[test]
    fn test_calculate_raw_lbm_points() {
        let bodyweight_points = vec![
            DataPoint { date: make_date(2024, 1, 1), value: 80.0 },
            DataPoint { date: make_date(2024, 1, 15), value: 79.0 }, // No neck/waist match
            DataPoint { date: make_date(2024, 2, 1), value: 78.0 },
        ];
        let neck_points = vec![
            DataPoint { date: make_date(2024, 1, 1), value: 38.0 },
            DataPoint { date: make_date(2024, 2, 1), value: 38.5 },
        ];
        let waist_points = vec![
            DataPoint { date: make_date(2024, 1, 1), value: 90.0 },
            DataPoint { date: make_date(2024, 2, 1), value: 88.0 },
        ];

        let lbm_points = calculate_raw_lbm_points(&bodyweight_points, &neck_points, &waist_points);

        // Should only have 2 points (1/1 and 2/1)
        assert_eq!(lbm_points.len(), 2);

        // LBM should be calculated correctly for 1/1
        let bf_jan1 = calculate_body_fat_pct(90.0, 38.0).unwrap();
        let expected_lbm_jan1 = calculate_lbm(80.0, bf_jan1);
        assert!((lbm_points[0].value - expected_lbm_jan1).abs() < 0.01);
    }

    #[test]
    fn test_specific_measurement_values() {
        // Test with approximate values from the task specification
        // waist=100, neck=38, height=180 -> BF% should be ~25-27%
        let bf = calculate_body_fat_pct(100.0, 38.0).unwrap();
        assert!(bf > 24.0 && bf < 28.0, "BF% = {} (expected ~25-27%)", bf);

        // LBM with bodyweight=86, BF%=25 -> LBM should be ~64.5kg
        let lbm = calculate_lbm(86.0, 25.0);
        assert!((lbm - 64.5).abs() < 0.1, "LBM = {} (expected ~64.5)", lbm);
    }
}
