//! Gaussian Process regression for strength training time series.
//!
//! This module implements GP regression for modeling trends in training data.
//! It uses a squared exponential (RBF) kernel which is well-suited for smooth
//! functions like strength progression over time.

use chrono::NaiveDate;
use nalgebra::{Cholesky, DMatrix, DVector, Dyn};
use thiserror::Error;

use crate::domain::DataPoint;

/// Reference date for converting dates to numeric values.
/// All dates are converted to days since this reference.
const REFERENCE_DATE: NaiveDate = match NaiveDate::from_ymd_opt(2020, 1, 1) {
    Some(d) => d,
    None => panic!("Invalid reference date"),
};

/// Errors that can occur during GP operations.
#[derive(Debug, Error)]
pub enum GpError {
    #[error("insufficient data: need at least 2 observations, got {0}")]
    InsufficientData(usize),

    #[error("singular matrix: cannot solve GP system (data may be degenerate)")]
    SingularMatrix,

    #[error("invalid hyperparameters: {0}")]
    InvalidHyperparameters(String),

    #[error("invalid date range: start {start} is after end {end}")]
    DateRangeError { start: NaiveDate, end: NaiveDate },
}

/// Hyperparameters for GP regression.
#[derive(Debug, Clone)]
pub struct GpHyperparameters {
    /// Length scale in days - controls smoothness of predictions.
    /// Larger values = smoother predictions.
    pub length_scale_days: f64,

    /// Signal variance - controls amplitude of variation.
    pub signal_variance: f64,

    /// Noise variance - models observation noise and daily variation.
    pub noise_variance: f64,
}

impl GpHyperparameters {
    /// Creates new hyperparameters with validation.
    pub fn new(
        length_scale_days: f64,
        signal_variance: f64,
        noise_variance: f64,
    ) -> Result<Self, GpError> {
        if length_scale_days <= 0.0 {
            return Err(GpError::InvalidHyperparameters(
                "length_scale_days must be positive".to_string(),
            ));
        }
        if signal_variance <= 0.0 {
            return Err(GpError::InvalidHyperparameters(
                "signal_variance must be positive".to_string(),
            ));
        }
        if noise_variance < 0.0 {
            return Err(GpError::InvalidHyperparameters(
                "noise_variance must be non-negative".to_string(),
            ));
        }

        Ok(Self {
            length_scale_days,
            signal_variance,
            noise_variance,
        })
    }

    /// Estimates hyperparameters from training data.
    ///
    /// Uses heuristics appropriate for strength training:
    /// - Length scale ~90 days (strength changes over months)
    /// - Signal variance from data variance
    /// - Noise variance ~5% of signal variance
    pub fn estimate_from_data(data: &[DataPoint]) -> Result<Self, GpError> {
        if data.len() < 2 {
            return Err(GpError::InsufficientData(data.len()));
        }

        // Calculate variance of observed values
        let values: Vec<f64> = data.iter().map(|p| p.value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        // Use data variance as signal variance, with a minimum floor
        let signal_variance = variance.max(1.0);

        // Noise variance ~5% of signal variance (measurement noise + daily variation)
        let noise_variance = signal_variance * 0.05;

        // Length scale of ~90 days - strength adapts over months
        let length_scale_days = 90.0;

        Self::new(length_scale_days, signal_variance, noise_variance)
    }
}

impl Default for GpHyperparameters {
    fn default() -> Self {
        Self {
            length_scale_days: 90.0,
            signal_variance: 100.0, // ~10kg standard deviation for e1RM
            noise_variance: 5.0,    // ~2.2kg observation noise
        }
    }
}

/// Result of GP prediction at a single point.
#[derive(Debug, Clone)]
pub struct GpPrediction {
    pub date: NaiveDate,
    pub mean: f64,
    pub std_dev: f64,
}

impl GpPrediction {
    /// Lower bound of 95% confidence interval.
    pub fn ci_lower(&self) -> f64 {
        self.mean - 1.96 * self.std_dev
    }

    /// Upper bound of 95% confidence interval.
    pub fn ci_upper(&self) -> f64 {
        self.mean + 1.96 * self.std_dev
    }
}

/// Fitted Gaussian Process model for a single movement.
///
/// Note: Manual Debug implementation since Cholesky<f64, Dyn> doesn't implement Debug
pub struct GpModel {
    /// Training data times (days since reference)
    train_times: Vec<f64>,
    /// Training data values (centered)
    #[allow(dead_code)] // Retained for potential future use (e.g., model diagnostics)
    train_values: DVector<f64>,
    /// Mean of training values (for de-centering predictions)
    train_mean: f64,
    /// Precomputed alpha = K^-1 * y for efficient prediction
    alpha: DVector<f64>,
    /// Cached Cholesky decomposition of kernel matrix (L where K = LLᵀ)
    /// Used for efficient variance computation without rebuilding the matrix
    cholesky: Cholesky<f64, Dyn>,
    /// Hyperparameters
    hyperparams: GpHyperparameters,
}

impl std::fmt::Debug for GpModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpModel")
            .field("train_times", &self.train_times)
            .field("train_values", &self.train_values)
            .field("train_mean", &self.train_mean)
            .field("alpha", &self.alpha)
            .field("cholesky", &"<Cholesky decomposition>")
            .field("hyperparams", &self.hyperparams)
            .finish()
    }
}

impl GpModel {
    /// Fits a GP model to the training data.
    ///
    /// Returns an error if there's insufficient data or the kernel matrix
    /// is singular.
    pub fn fit(data: &[DataPoint], hyperparams: GpHyperparameters) -> Result<Self, GpError> {
        if data.len() < 2 {
            return Err(GpError::InsufficientData(data.len()));
        }

        // Convert dates to numeric values
        let train_times: Vec<f64> = data.iter().map(|p| date_to_days(p.date)).collect();

        // Center the values (subtract mean)
        let values: Vec<f64> = data.iter().map(|p| p.value).collect();
        let train_mean = values.iter().sum::<f64>() / values.len() as f64;
        let centered_values: Vec<f64> = values.iter().map(|v| v - train_mean).collect();
        let train_values = DVector::from_vec(centered_values);

        // Build kernel matrix
        let k = build_kernel_matrix(
            &train_times,
            hyperparams.length_scale_days,
            hyperparams.signal_variance,
            hyperparams.noise_variance,
        );

        // Solve K * alpha = y using Cholesky decomposition
        // Store the Cholesky factor for efficient variance computation later
        let (alpha, cholesky) = solve_via_cholesky(&k, &train_values)?;

        Ok(Self {
            train_times,
            train_values,
            train_mean,
            alpha,
            cholesky,
            hyperparams,
        })
    }

    /// Predicts at specific dates.
    pub fn predict(&self, dates: &[NaiveDate]) -> Vec<GpPrediction> {
        if dates.is_empty() {
            return Vec::new();
        }

        let test_times: Vec<f64> = dates.iter().map(|d| date_to_days(*d)).collect();

        // Build cross-kernel matrix K* (test vs train)
        let k_star = build_cross_kernel_matrix(
            &self.train_times,
            &test_times,
            self.hyperparams.length_scale_days,
            self.hyperparams.signal_variance,
        );

        // Mean prediction: μ* = K* × α + train_mean
        let means = &k_star * &self.alpha;

        // For variance, we need K** and K* × K^-1 × K*^T
        // Compute variance at each test point using batched matrix operations
        let variances = self.compute_predictive_variance(&k_star);

        dates
            .iter()
            .enumerate()
            .map(|(i, &date)| GpPrediction {
                date,
                mean: means[i] + self.train_mean,
                std_dev: variances[i].sqrt().max(0.0),
            })
            .collect()
    }

    /// Predicts over a date range with daily resolution.
    pub fn predict_range(
        &self,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<GpPrediction>, GpError> {
        if start > end {
            return Err(GpError::DateRangeError { start, end });
        }

        let mut dates = Vec::new();
        let mut current = start;
        while current <= end {
            dates.push(current);
            current = current.succ_opt().unwrap_or(current);
            if current == start {
                // Overflow protection
                break;
            }
        }

        Ok(self.predict(&dates))
    }

    /// Computes predictive variance at test points.
    ///
    /// Uses the cached Cholesky decomposition from fit() to avoid
    /// redundant O(n³) matrix factorization.
    ///
    /// Batches all triangular solves into a single matrix operation
    /// for better cache utilization and BLAS optimization.
    fn compute_predictive_variance(&self, k_star: &DMatrix<f64>) -> Vec<f64> {
        let m = k_star.nrows(); // Number of test points

        // k_star is m×n (test points × train points)
        // We need to solve L * V = K*ᵀ where K*ᵀ is n×m
        // This gives us V (n×m) in a single batch operation
        let k_star_t = k_star.transpose();
        let v = self.cholesky.l().solve_lower_triangular(&k_star_t);

        match v {
            Some(v) => {
                // Compute diagonal of K** - VᵀV efficiently
                // For each test point i: var_i = k**(t_i, t_i) - ||v_col_i||²
                // K** diagonal element (self-covariance at test point)
                // For RBF kernel, k(t,t) = signal_variance (constant for all test points)
                let k_star_star = self.hyperparams.signal_variance;

                let mut variances = Vec::with_capacity(m);
                for i in 0..m {
                    // Get column i of V and compute its squared norm
                    let v_col = v.column(i);
                    let v_norm_sq = v_col.dot(&v_col);

                    // Predictive variance = posterior variance + noise variance
                    let posterior_var = k_star_star - v_norm_sq;
                    let predictive_var = posterior_var + self.hyperparams.noise_variance;
                    variances.push(predictive_var.max(0.0));
                }
                variances
            }
            None => {
                // Fallback: return signal + noise variance as upper bound
                vec![
                    self.hyperparams.signal_variance + self.hyperparams.noise_variance;
                    m
                ]
            }
        }
    }

    /// Returns the training data times for inspection.
    #[allow(dead_code)] // Used in Phase 3
    pub fn train_times(&self) -> &[f64] {
        &self.train_times
    }

    /// Returns the number of training points.
    #[allow(dead_code)] // Used in Phase 3
    pub fn n_train(&self) -> usize {
        self.train_times.len()
    }
}

/// Converts a date to days since the reference date.
fn date_to_days(date: NaiveDate) -> f64 {
    (date - REFERENCE_DATE).num_days() as f64
}

/// Converts days since reference back to a date.
#[allow(dead_code)]
fn days_to_date(days: f64) -> NaiveDate {
    REFERENCE_DATE + chrono::Duration::days(days.round() as i64)
}

/// Squared exponential (RBF) kernel function.
///
/// k(x, x') = σ² × exp(-0.5 × (x - x')² / l²)
fn squared_exp_kernel(x1: f64, x2: f64, length_scale: f64, signal_variance: f64) -> f64 {
    let diff = x1 - x2;
    let scaled_sq_dist = (diff * diff) / (length_scale * length_scale);
    signal_variance * (-0.5 * scaled_sq_dist).exp()
}

/// Builds the kernel matrix for training points.
///
/// Adds noise variance to the diagonal for numerical stability
/// and to model observation noise.
fn build_kernel_matrix(
    times: &[f64],
    length_scale: f64,
    signal_variance: f64,
    noise_variance: f64,
) -> DMatrix<f64> {
    let n = times.len();
    let mut k = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            k[(i, j)] = squared_exp_kernel(times[i], times[j], length_scale, signal_variance);
        }
        // Add noise variance to diagonal
        k[(i, i)] += noise_variance;
    }

    k
}

/// Builds the cross-kernel matrix between training and test points.
fn build_cross_kernel_matrix(
    train_times: &[f64],
    test_times: &[f64],
    length_scale: f64,
    signal_variance: f64,
) -> DMatrix<f64> {
    let n_test = test_times.len();
    let n_train = train_times.len();
    let mut k = DMatrix::zeros(n_test, n_train);

    for i in 0..n_test {
        for j in 0..n_train {
            k[(i, j)] =
                squared_exp_kernel(test_times[i], train_times[j], length_scale, signal_variance);
        }
    }

    k
}

/// Solves K × α = y using Cholesky decomposition.
///
/// Returns both the solution α and the Cholesky factor for reuse.
/// Adds jitter to diagonal if initial decomposition fails.
fn solve_via_cholesky(
    k: &DMatrix<f64>,
    y: &DVector<f64>,
) -> Result<(DVector<f64>, Cholesky<f64, Dyn>), GpError> {
    // Try direct Cholesky first
    if let Some(chol) = k.clone().cholesky() {
        let alpha = chol.solve(y);
        return Ok((alpha, chol));
    }

    // Add jitter and retry
    let n = k.nrows();
    let mut k_jitter = k.clone();
    let jitter = 1e-6;
    for i in 0..n {
        k_jitter[(i, i)] += jitter;
    }

    if let Some(chol) = k_jitter.clone().cholesky() {
        let alpha = chol.solve(y);
        return Ok((alpha, chol));
    }

    // Try larger jitter
    let jitter = 1e-4;
    for i in 0..n {
        k_jitter[(i, i)] += jitter;
    }

    if let Some(chol) = k_jitter.cholesky() {
        let alpha = chol.solve(y);
        return Ok((alpha, chol));
    }

    Err(GpError::SingularMatrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    #[test]
    fn test_kernel_same_point() {
        // k(x, x) should equal signal_variance + noise (but we don't add noise in the kernel fn)
        let k = squared_exp_kernel(0.0, 0.0, 90.0, 100.0);
        assert!((k - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_symmetry() {
        let k1 = squared_exp_kernel(10.0, 20.0, 90.0, 100.0);
        let k2 = squared_exp_kernel(20.0, 10.0, 90.0, 100.0);
        assert!((k1 - k2).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_decreases_with_distance() {
        let k_close = squared_exp_kernel(0.0, 10.0, 90.0, 100.0);
        let k_far = squared_exp_kernel(0.0, 100.0, 90.0, 100.0);
        assert!(k_close > k_far);
    }

    #[test]
    fn test_kernel_matrix_is_symmetric() {
        let times = vec![0.0, 30.0, 60.0, 90.0];
        let k = build_kernel_matrix(&times, 90.0, 100.0, 5.0);

        for i in 0..4 {
            for j in 0..4 {
                assert!((k[(i, j)] - k[(j, i)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_date_conversion_roundtrip() {
        let date = make_date(2024, 6, 15);
        let days = date_to_days(date);
        let back = days_to_date(days);
        assert_eq!(date, back);
    }

    #[test]
    fn test_insufficient_data_error() {
        let data = vec![DataPoint {
            date: make_date(2024, 1, 1),
            value: 100.0,
        }];
        let result = GpModel::fit(&data, GpHyperparameters::default());
        assert!(matches!(result, Err(GpError::InsufficientData(1))));
    }

    #[test]
    fn test_gp_fit_and_predict() {
        // Create synthetic data with clear trend
        let data = vec![
            DataPoint {
                date: make_date(2024, 1, 1),
                value: 100.0,
            },
            DataPoint {
                date: make_date(2024, 2, 1),
                value: 105.0,
            },
            DataPoint {
                date: make_date(2024, 3, 1),
                value: 110.0,
            },
            DataPoint {
                date: make_date(2024, 4, 1),
                value: 112.0,
            },
        ];

        let model = GpModel::fit(&data, GpHyperparameters::default()).unwrap();

        // Predict at training points - should be close to observations
        let train_dates: Vec<NaiveDate> = data.iter().map(|p| p.date).collect();
        let predictions = model.predict(&train_dates);

        for (pred, actual) in predictions.iter().zip(data.iter()) {
            // Should be within a few kg of actual
            assert!(
                (pred.mean - actual.value).abs() < 10.0,
                "Prediction {} too far from actual {}",
                pred.mean,
                actual.value
            );
        }
    }

    #[test]
    fn test_uncertainty_increases_with_distance() {
        let data = vec![
            DataPoint {
                date: make_date(2024, 1, 1),
                value: 100.0,
            },
            DataPoint {
                date: make_date(2024, 2, 1),
                value: 105.0,
            },
            DataPoint {
                date: make_date(2024, 3, 1),
                value: 110.0,
            },
        ];

        let model = GpModel::fit(&data, GpHyperparameters::default()).unwrap();

        // Predict at training point and far future
        let predictions = model.predict(&[
            make_date(2024, 2, 1),  // Training point
            make_date(2024, 12, 1), // 9 months out
        ]);

        // Uncertainty should be higher for future prediction
        assert!(
            predictions[1].std_dev > predictions[0].std_dev,
            "Future uncertainty {} should be > training point uncertainty {}",
            predictions[1].std_dev,
            predictions[0].std_dev
        );
    }

    #[test]
    fn test_predict_range() {
        let data = vec![
            DataPoint {
                date: make_date(2024, 1, 1),
                value: 100.0,
            },
            DataPoint {
                date: make_date(2024, 2, 1),
                value: 105.0,
            },
        ];

        let model = GpModel::fit(&data, GpHyperparameters::default()).unwrap();
        let predictions = model
            .predict_range(make_date(2024, 1, 1), make_date(2024, 1, 10))
            .unwrap();

        assert_eq!(predictions.len(), 10);
        assert_eq!(predictions[0].date, make_date(2024, 1, 1));
        assert_eq!(predictions[9].date, make_date(2024, 1, 10));
    }

    #[test]
    fn test_confidence_intervals() {
        let pred = GpPrediction {
            date: make_date(2024, 1, 1),
            mean: 100.0,
            std_dev: 5.0,
        };

        assert!((pred.ci_lower() - 90.2).abs() < 0.1);
        assert!((pred.ci_upper() - 109.8).abs() < 0.1);
    }

    #[test]
    fn test_hyperparameters_validation() {
        assert!(GpHyperparameters::new(0.0, 100.0, 5.0).is_err());
        assert!(GpHyperparameters::new(90.0, -1.0, 5.0).is_err());
        assert!(GpHyperparameters::new(90.0, 100.0, -1.0).is_err());
        assert!(GpHyperparameters::new(90.0, 100.0, 5.0).is_ok());
    }

    #[test]
    fn test_estimate_hyperparameters() {
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
                value: 105.0,
            },
        ];

        let hp = GpHyperparameters::estimate_from_data(&data).unwrap();

        assert!(hp.signal_variance > 0.0);
        assert!(hp.noise_variance > 0.0);
        assert_eq!(hp.length_scale_days, 90.0);
    }
}
