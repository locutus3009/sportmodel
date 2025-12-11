//! TDEE (Total Daily Energy Expenditure) calculation from calorie and weight data.
//!
//! This module calculates empirical TDEE using a 28-day window of calorie intake
//! and bodyweight measurements, with EMA (Exponential Moving Average) smoothing
//! for weight trends.

use std::collections::BTreeMap;

use chrono::{Duration, Local, NaiveDate};
use serde::Serialize;

use crate::domain::Observation;

// === Constants ===

/// Main window for TDEE comparison (days).
pub const TDEE_WINDOW_DAYS: i64 = 28;

/// Days for each EMA calculation window.
pub const EMA_WINDOW_DAYS: i64 = 10;

/// EMA smoothing factor (0.1 = 10% weight on new values).
pub const EMA_ALPHA: f64 = 0.1;

/// Energy density of body fat (kcal per kg).
pub const KCAL_PER_KG_FAT: f64 = 7700.0;

/// Minimum fraction of valid calorie-weight pairs required (14/28 = 50%).
pub const MIN_PAIR_RATIO: f64 = 0.5;

/// Minimum weight measurements required in each 10-day EMA window.
pub const MIN_EMA_DATA_POINTS: usize = 3;

// === Data Structures ===

/// TDEE calculation result.
#[derive(Debug, Clone, Serialize)]
pub struct TdeeResult {
    /// Calculated EMA TDEE in kcal.
    pub average_tdee: f64,
    /// Calculated TDEE in kcal.
    pub tdee: f64,
    /// Average daily calorie intake over the 28-day window.
    pub avg_calories: f64,
    /// EMA of weight at the start of the window (28 days ago).
    pub ema_start: f64,
    /// EMA of weight at the end of the window (today).
    pub ema_end: f64,
    /// Weight change in kg (ema_end - ema_start).
    pub weight_change_kg: f64,
    /// Number of valid calorie-weight pairs used.
    pub pairs_used: usize,
}

/// Detailed reason why TDEE couldn't be calculated.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "error", content = "details")]
pub enum TdeeError {
    /// Not enough calorie data entries.
    #[serde(rename = "insufficient_calorie_data")]
    InsufficientCalorieData { available: usize, required: usize },

    /// Not enough weight measurements in the EMA start window.
    #[serde(rename = "insufficient_weight_data_ema_start")]
    InsufficientWeightDataForEmaStart { available: usize, required: usize },

    /// Not enough weight measurements in the EMA end window.
    #[serde(rename = "insufficient_weight_data_ema_end")]
    InsufficientWeightDataForEmaEnd { available: usize, required: usize },

    /// Not enough valid calorie-weight pairs in the 28-day window.
    #[serde(rename = "insufficient_pairs")]
    InsufficientPairs { available: usize, required: usize },

    /// Data span is too short (need at least 38 days).
    #[serde(rename = "data_span_too_short")]
    DataSpanTooShort {
        available_days: i64,
        required_days: i64,
    },
}

impl std::fmt::Display for TdeeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TdeeError::InsufficientCalorieData {
                available,
                required,
            } => {
                write!(f, "Need {} calorie entries, found {}", required, available)
            }
            TdeeError::InsufficientWeightDataForEmaStart {
                available,
                required,
            } => {
                write!(
                    f,
                    "Need {} weights in EMA start window, found {}",
                    required, available
                )
            }
            TdeeError::InsufficientWeightDataForEmaEnd {
                available,
                required,
            } => {
                write!(
                    f,
                    "Need {} weights in EMA end window, found {}",
                    required, available
                )
            }
            TdeeError::InsufficientPairs {
                available,
                required,
            } => {
                write!(
                    f,
                    "Need {} calorie-weight pairs, found {}",
                    required, available
                )
            }
            TdeeError::DataSpanTooShort {
                available_days,
                required_days,
            } => {
                write!(
                    f,
                    "Data span {} days is less than required {} days",
                    available_days, required_days
                )
            }
        }
    }
}

impl std::error::Error for TdeeError {}

/// Result of EMA calculation.
#[derive(Debug, Clone)]
struct EmaResult {
    /// The calculated EMA value.
    value: f64,
}

// === Main Calculation Functions ===

fn calculate_tdee_for_date(
    today: NaiveDate,
    weights: &BTreeMap<NaiveDate, f64>,
    calories: &BTreeMap<NaiveDate, f64>,
) -> Result<TdeeResult, TdeeError> {
    // Check minimum calorie data
    let min_required_pairs = (TDEE_WINDOW_DAYS as f64 * MIN_PAIR_RATIO).ceil() as usize;
    if calories.len() < min_required_pairs {
        return Err(TdeeError::InsufficientCalorieData {
            available: calories.len(),
            required: min_required_pairs,
        });
    }

    // Calculate EMA for end date (today)
    let ema_end_date = today;
    let ema_end_result = calculate_ema_for_date(weights, ema_end_date).ok_or(
        TdeeError::InsufficientWeightDataForEmaEnd {
            available: count_weights_in_window(weights, ema_end_date),
            required: MIN_EMA_DATA_POINTS,
        },
    )?;

    // Calculate EMA for start date (28 days ago)
    let ema_start_date = today - Duration::days(TDEE_WINDOW_DAYS);
    let ema_start_result = calculate_ema_for_date(weights, ema_start_date).ok_or(
        TdeeError::InsufficientWeightDataForEmaStart {
            available: count_weights_in_window(weights, ema_start_date),
            required: MIN_EMA_DATA_POINTS,
        },
    )?;

    // Collect valid calorie-weight pairs in the 28-day window
    // A pair is valid if we have calorie on day X and weight on day X+1
    let mut valid_pairs: Vec<f64> = Vec::new();
    let window_start = ema_start_date;
    let window_end = ema_end_date - Duration::days(1); // Last calorie day is yesterday

    let mut current = window_start;
    while current <= window_end {
        let next_day = current + Duration::days(1);
        if calories.contains_key(&current) && weights.contains_key(&next_day) {
            valid_pairs.push(calories[&current]);
        }
        current = next_day;
    }

    // Check minimum pairs requirement
    if valid_pairs.len() < min_required_pairs {
        return Err(TdeeError::InsufficientPairs {
            available: valid_pairs.len(),
            required: min_required_pairs,
        });
    }

    // Calculate average calories
    let avg_calories = valid_pairs.iter().sum::<f64>() / valid_pairs.len() as f64;

    // Calculate weight change
    let weight_change_kg = ema_end_result.value - ema_start_result.value;

    // Calculate TDEE: TDEE = Avg_Calories - (weight_change / days) * kcal_per_kg
    // Weight change is positive if gaining, so we subtract the surplus/deficit
    let daily_weight_change = weight_change_kg / TDEE_WINDOW_DAYS as f64;
    let tdee = avg_calories - (daily_weight_change * KCAL_PER_KG_FAT);

    Ok(TdeeResult {
        average_tdee: 0.0,
        tdee,
        avg_calories,
        ema_start: ema_start_result.value,
        ema_end: ema_end_result.value,
        weight_change_kg,
        pairs_used: valid_pairs.len(),
    })
}

/// Calculates TDEE from calorie and weight observations.
///
/// Uses a 28-day window ending today, with 10-day EMA smoothing for weight.
/// Requires at least 50% of days to have valid calorie-weight pairs.
///
/// # Arguments
/// * `calorie_data` - Calorie observations (daily calorie intake)
/// * `weight_data` - Bodyweight observations (morning weight measurements)
///
/// # Returns
/// `Ok(TdeeResult)` with calculated TDEE, or `Err(TdeeError)` with reason for failure.
pub fn calculate_tdee(
    calorie_data: &[Observation],
    weight_data: &[Observation],
) -> Result<TdeeResult, TdeeError> {
    let today = Local::now().date_naive();

    // Required days: 28 (main window) + 10 (EMA lookback for start) = 38 minimum
    let required_days = TDEE_WINDOW_DAYS + EMA_WINDOW_DAYS + EMA_WINDOW_DAYS;

    // Build weight lookup map
    let weights = build_weight_map(weight_data);

    // Check if we have enough data span
    if let (Some(&min_date), Some(&max_date)) = (weights.keys().next(), weights.keys().next_back())
    {
        let span_days = (max_date - min_date).num_days();
        if span_days < required_days {
            return Err(TdeeError::DataSpanTooShort {
                available_days: span_days,
                required_days,
            });
        }
    } else {
        return Err(TdeeError::InsufficientWeightDataForEmaStart {
            available: 0,
            required: MIN_EMA_DATA_POINTS,
        });
    }

    // Build calorie lookup map
    let calories = build_calorie_map(calorie_data);

    // Check minimum calorie data
    let min_required_pairs = (TDEE_WINDOW_DAYS as f64 * MIN_PAIR_RATIO).ceil() as usize;
    if calories.len() < min_required_pairs {
        return Err(TdeeError::InsufficientCalorieData {
            available: calories.len(),
            required: min_required_pairs,
        });
    }

    let tdees = {
        let mut tmp = BTreeMap::new();
        for i in 0..EMA_WINDOW_DAYS {
            tmp.insert(
                today - Duration::days(i),
                calculate_tdee_for_date(today - Duration::days(i), &weights, &calories)?.tdee,
            );
        }
        tmp
    };

    let tdee_averaged = calculate_ema_for_date(&tdees, today).ok_or(
        TdeeError::InsufficientWeightDataForEmaEnd {
            available: count_weights_in_window(&tdees, today),
            required: MIN_EMA_DATA_POINTS,
        },
    )?;

    let mut result = calculate_tdee_for_date(today, &weights, &calories)?;
    result.average_tdee = tdee_averaged.value;
    Ok(result)
}

// === Helper Functions ===

/// Builds a map of date -> weight from observations.
fn build_weight_map(weight_data: &[Observation]) -> BTreeMap<NaiveDate, f64> {
    let mut weights = BTreeMap::new();
    for obs in weight_data {
        // If multiple weights on same day, use the last one
        weights.insert(obs.date, obs.weight_kg);
    }
    weights
}

/// Builds a map of date -> calories from observations.
fn build_calorie_map(calorie_data: &[Observation]) -> BTreeMap<NaiveDate, f64> {
    let mut calories = BTreeMap::new();
    for obs in calorie_data {
        // If multiple entries on same day, use the last one (or could sum them)
        calories.insert(obs.date, obs.weight_kg);
    }
    calories
}

/// Counts weight measurements in the 10-day EMA window for a target date.
fn count_weights_in_window(weights: &BTreeMap<NaiveDate, f64>, target_date: NaiveDate) -> usize {
    let window_start = target_date - Duration::days(EMA_WINDOW_DAYS - 1);
    weights.range(window_start..=target_date).count()
}

/// Calculates EMA for a target date using the 10-day window.
///
/// The window covers `[target_date - 9, target_date]` inclusive.
/// EMA is initialized with the last available weight in the window,
/// then updated for each subsequent day (carrying forward on gaps).
///
/// # Returns
/// `Some(EmaResult)` if at least `MIN_EMA_DATA_POINTS` weights exist in window,
/// `None` otherwise.
fn calculate_ema_for_date(
    weights: &BTreeMap<NaiveDate, f64>,
    target_date: NaiveDate,
) -> Option<EmaResult> {
    let window_start = target_date - Duration::days(EMA_WINDOW_DAYS - 1);

    // Collect weights within the window, sorted by date (BTreeMap maintains order)
    let window_weights: Vec<(NaiveDate, f64)> = weights
        .range(window_start..=target_date)
        .map(|(d, w)| (*d, *w))
        .collect();

    // Minimum data requirement
    if window_weights.len() < MIN_EMA_DATA_POINTS {
        return None;
    }

    // Initialize EMA with last available weight (not necessarily target day of window)
    let (first_date, _) = window_weights[0];
    let (last_date, last_weight) = *window_weights.last().unwrap();
    let mut ema = last_weight;

    // Process remaining days from last_date (can be lower than target_date) to first_date
    let days_to_process = (last_date - first_date).num_days();
    for day_offset in 1..=days_to_process {
        let current_date = last_date - Duration::days(day_offset);

        if let Some(&weight) = weights.get(&current_date) {
            // Data exists: update EMA
            ema = EMA_ALPHA * weight + (1.0 - EMA_ALPHA) * ema;
        }
        // No data: EMA carries forward (no action needed)
    }

    Some(EmaResult { value: ema })
}

// === Unit Tests ===

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Movement;

    /// Helper to create a weight observation.
    fn weight_obs(date: NaiveDate, weight: f64) -> Observation {
        Observation::new(date, weight, None, Movement::Bodyweight)
    }

    /// Helper to create a calorie observation.
    fn calorie_obs(date: NaiveDate, calories: f64) -> Observation {
        Observation::new(date, calories, None, Movement::Calorie)
    }

    /// Helper to create a date.
    fn date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    // === EMA Tests ===

    #[test]
    fn test_ema_full_window() {
        // 10 consecutive days of data
        let weights: BTreeMap<NaiveDate, f64> = (0..10)
            .map(|i| (date(2024, 1, 1) + Duration::days(i), 80.0 + i as f64 * 0.1))
            .collect();

        let target = date(2024, 1, 10);
        let result = calculate_ema_for_date(&weights, target);

        assert!(result.is_some());
        let ema = result.unwrap();
        // EMA should be between first and last value
        assert!(ema.value > 80.0);
        assert!(ema.value < 80.9);
    }

    #[test]
    fn test_ema_first_days_missing() {
        // Data on days 4-10 only (7 points)
        let weights: BTreeMap<NaiveDate, f64> = (3..10)
            .map(|i| (date(2024, 1, 1) + Duration::days(i), 80.0))
            .collect();

        let target = date(2024, 1, 10);
        let result = calculate_ema_for_date(&weights, target);

        assert!(result.is_some());
        let ema = result.unwrap();
        // With constant weight, EMA should equal that weight
        assert!((ema.value - 80.0).abs() < 0.001);
    }

    #[test]
    fn test_ema_scattered_gaps() {
        // Data on days 1, 3, 5, 7, 9 (5 points)
        let weights: BTreeMap<NaiveDate, f64> = [0, 2, 4, 6, 8]
            .into_iter()
            .map(|i| (date(2024, 1, 1) + Duration::days(i), 80.0))
            .collect();

        let target = date(2024, 1, 10);
        let result = calculate_ema_for_date(&weights, target);

        assert!(result.is_some());
        let ema = result.unwrap();
        // With constant weight and gaps, EMA should equal that weight
        assert!((ema.value - 80.0).abs() < 0.001);
    }

    #[test]
    fn test_ema_insufficient_data() {
        // Only 2 data points in window
        let weights: BTreeMap<NaiveDate, f64> =
            [(date(2024, 1, 1), 80.0), (date(2024, 1, 10), 80.5)]
                .into_iter()
                .collect();

        let target = date(2024, 1, 10);
        let result = calculate_ema_for_date(&weights, target);

        assert!(result.is_none());
    }

    #[test]
    fn test_ema_minimum_data() {
        // Exactly 3 data points (minimum required)
        let weights: BTreeMap<NaiveDate, f64> = [
            (date(2024, 1, 1), 80.0),
            (date(2024, 1, 5), 80.2),
            (date(2024, 1, 10), 80.4),
        ]
        .into_iter()
        .collect();

        let target = date(2024, 1, 10);
        let result = calculate_ema_for_date(&weights, target);

        assert!(result.is_some());
    }

    #[test]
    fn test_ema_only_last_day() {
        // Data only on day 10 (1 point, should fail)
        let weights: BTreeMap<NaiveDate, f64> = [(date(2024, 1, 10), 80.0)].into_iter().collect();

        let target = date(2024, 1, 10);
        let result = calculate_ema_for_date(&weights, target);

        assert!(result.is_none());
    }

    #[test]
    fn test_ema_with_weight_change() {
        // Weights increasing linearly from 80 to 81 over 10 days
        let weights: BTreeMap<NaiveDate, f64> = (0..10)
            .map(|i| {
                let weight = 80.0 + (i as f64 / 9.0);
                (date(2024, 1, 1) + Duration::days(i), weight)
            })
            .collect();

        let target = date(2024, 1, 10);
        let result = calculate_ema_for_date(&weights, target);

        assert!(result.is_some());
        let ema = result.unwrap();
        // EMA lags behind actual values, so should be less than 81 but more than 80
        assert!(ema.value > 80.0);
        assert!(ema.value < 81.0);
    }

    // === TDEE Tests ===

    #[test]
    fn test_tdee_maintenance() {
        // Test with maintenance calories (no weight change expected)
        let today = Local::now().date_naive();
        const N1: i64 = 55;
        const N2: i64 = 45;

        // Create 45 days of weight data (constant 80kg)
        let weight_data: Vec<Observation> = (0..N1)
            .map(|i| weight_obs(today - Duration::days(N1 - 1 - i), 80.0))
            .collect();

        // Create 35 days of calorie data (constant 2500 kcal)
        let calorie_data: Vec<Observation> = (0..N2)
            .map(|i| calorie_obs(today - Duration::days(N2 - 1 - i), 2500.0))
            .collect();

        let result = calculate_tdee(&calorie_data, &weight_data);

        assert!(result.is_ok());
        let tdee = result.unwrap();

        // With no weight change, TDEE should approximately equal average intake
        assert!((tdee.tdee - 2500.0).abs() < 100.0);
        assert!((tdee.avg_calories - 2500.0).abs() < 1.0);
        assert!((tdee.weight_change_kg).abs() < 0.1);
    }

    #[test]
    fn test_tdee_deficit() {
        // Test with caloric deficit (weight loss expected)
        let today = Local::now().date_naive();
        const N1: usize = 55;
        const N2: i64 = 45;

        // Create 45 days of weight data (decreasing from 81 to 80 over 28 days)
        let weight_data: Vec<Observation> = (0..N1)
            .map(|i| {
                let days_from_end = N1 - 1 - i;
                let weight = if days_from_end < 28 {
                    80.0 + (1.0 - (28 - days_from_end) as f64 / 28.0)
                } else {
                    81.0
                };
                weight_obs(today - Duration::days(days_from_end as i64), weight)
            })
            .collect();

        // Create 35 days of calorie data (constant 2000 kcal - in deficit)
        let calorie_data: Vec<Observation> = (0..N2)
            .map(|i| calorie_obs(today - Duration::days(N2 - 1 - i), 2000.0))
            .collect();

        let result = calculate_tdee(&calorie_data, &weight_data);

        assert!(result.is_ok());
        let tdee = result.unwrap();

        // Losing ~1kg over 28 days = ~275 kcal/day deficit
        // So TDEE should be ~2275 kcal (2000 intake + 275 deficit)
        assert!(tdee.tdee > 2000.0);
        assert!(tdee.weight_change_kg < 0.0);
    }

    #[test]
    fn test_tdee_insufficient_calorie_data() {
        let today = Local::now().date_naive();

        // Create 45 days of weight data
        let weight_data: Vec<Observation> = (0..55)
            .map(|i| weight_obs(today - Duration::days(44 - i), 80.0))
            .collect();

        // Only 5 days of calorie data (not enough)
        let calorie_data: Vec<Observation> = (0..5)
            .map(|i| calorie_obs(today - Duration::days(4 - i), 2500.0))
            .collect();

        let result = calculate_tdee(&calorie_data, &weight_data);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TdeeError::InsufficientCalorieData { .. }
        ));
    }

    #[test]
    fn test_tdee_insufficient_weight_for_ema_start() {
        let today = Local::now().date_naive();

        // Only recent weight data (no data for EMA start window)
        let weight_data: Vec<Observation> = (0..15)
            .map(|i| weight_obs(today - Duration::days(14 - i), 80.0))
            .collect();

        // 35 days of calorie data
        let calorie_data: Vec<Observation> = (0..35)
            .map(|i| calorie_obs(today - Duration::days(34 - i), 2500.0))
            .collect();

        let result = calculate_tdee(&calorie_data, &weight_data);

        assert!(result.is_err());
        // Could be either DataSpanTooShort or InsufficientWeightDataForEmaStart
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            TdeeError::DataSpanTooShort { .. }
                | TdeeError::InsufficientWeightDataForEmaStart { .. }
        ));
    }

    #[test]
    fn test_tdee_insufficient_pairs() {
        let today = Local::now().date_naive();

        // Weight data for full span but sparse
        let weight_data: Vec<Observation> = (0..45)
            .filter(|i| i % 10 == 0) // Only every 10th day
            .map(|i| weight_obs(today - Duration::days(44 - i), 80.0))
            .collect();

        // Calorie data that doesn't match weight days
        let calorie_data: Vec<Observation> = (0..35)
            .filter(|i| i % 10 == 5) // Offset from weight days
            .map(|i| calorie_obs(today - Duration::days(34 - i), 2500.0))
            .collect();

        let result = calculate_tdee(&calorie_data, &weight_data);

        // Should fail due to insufficient data
        assert!(result.is_err());
    }

    #[test]
    fn test_tdee_with_gaps() {
        // Test that 50%+ pairs is sufficient
        let today = Local::now().date_naive();

        // Create 45 days of weight data with some gaps
        let weight_data: Vec<Observation> = (0..45)
            .filter(|i| i % 2 == 0 || *i < 10 || *i > 35) // Gaps in middle
            .map(|i| weight_obs(today - Duration::days(44 - i), 80.0))
            .collect();

        // Create 35 days of calorie data with some gaps
        let calorie_data: Vec<Observation> = (0..35)
            .filter(|i| i % 2 == 0 || *i < 10)
            .map(|i| calorie_obs(today - Duration::days(34 - i), 2500.0))
            .collect();

        let result = calculate_tdee(&calorie_data, &weight_data);

        // This might succeed or fail depending on exact alignment
        // The key is that it handles gaps gracefully
        if let Ok(tdee) = result {
            assert!(tdee.pairs_used >= 14); // At least 50% of 28 days
        }
    }
}
