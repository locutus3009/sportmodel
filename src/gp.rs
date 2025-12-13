//! Gaussian Process regression for strength training time series.
//!
//! This module implements GP regression for modeling trends in training data.
//! It uses a squared exponential (RBF) kernel which is well-suited for smooth
//! functions like strength progression over time.

use chrono::NaiveDate;
use nalgebra::{Cholesky, DMatrix, DVector, Dyn};
use thiserror::Error;

use crate::domain::{DataPoint, Movement};

/// Reference date for converting dates to numeric values.
/// All dates are converted to days since this reference.
const REFERENCE_DATE: NaiveDate = match NaiveDate::from_ymd_opt(2020, 1, 1) {
    Some(d) => d,
    None => panic!("Invalid reference date"),
};

/// Default GP hyperparameter configuration.
///
/// Length scale is domain knowledge, not optimized from data.
/// This struct allows easy customization per movement category,
/// with the option to add per-movement overrides in the future.
#[derive(Debug, Clone)]
pub struct GpConfig {
    /// Length scale for strength movements (squat, bench, deadlift, snatch, cj).
    /// Strength adaptations are slow, taking months to manifest.
    pub length_scale_strength: f64,

    /// Length scale for body composition (bodyweight, neck, waist, BF%, LBM).
    /// Body composition changes faster during cuts/bulks.
    pub length_scale_body_comp: f64,

    /// Length scale for energy/calorie tracking.
    pub length_scale_energy: f64,
}

impl Default for GpConfig {
    fn default() -> Self {
        Self {
            length_scale_strength: 90.0,
            length_scale_body_comp: 60.0,
            length_scale_energy: 60.0,
        }
    }
}

impl GpConfig {
    /// Returns the length scale for a given movement type.
    pub fn length_scale_for(&self, movement: Movement) -> f64 {
        match movement {
            Movement::Squat
            | Movement::Bench
            | Movement::Deadlift
            | Movement::Snatch
            | Movement::CleanAndJerk => self.length_scale_strength,

            Movement::Bodyweight | Movement::Neck | Movement::Waist => self.length_scale_body_comp,

            Movement::Calorie => self.length_scale_energy,
        }
    }

    /// Returns the length scale for derived body composition metrics (BF%, LBM).
    pub fn length_scale_body_composition(&self) -> f64 {
        self.length_scale_body_comp
    }
}

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
                vec![self.hyperparams.signal_variance + self.hyperparams.noise_variance; m]
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

    /// Computes the log marginal likelihood of the training data given the hyperparameters.
    ///
    /// log p(y|X,θ) = -½ yᵀK⁻¹y - ½ log|K| - n/2 log(2π)
    ///
    /// Uses cached Cholesky decomposition for efficiency:
    /// - yᵀK⁻¹y = yᵀα (α already computed during fit)
    /// - log|K| = 2 Σᵢ log(Lᵢᵢ) where K = LLᵀ
    pub fn log_marginal_likelihood(&self) -> f64 {
        let n = self.train_times.len() as f64;

        // Data fit term: -½ yᵀK⁻¹y = -½ yᵀα
        let data_fit = -0.5 * self.train_values.dot(&self.alpha);

        // Complexity penalty: -½ log|K| = -Σᵢ log(Lᵢᵢ)
        let l = self.cholesky.l();
        let log_det = (0..l.nrows()).map(|i| l[(i, i)].ln()).sum::<f64>();
        let complexity = -log_det;

        // Constant term: -n/2 log(2π)
        let constant = -0.5 * n * (2.0 * std::f64::consts::PI).ln();

        data_fit + complexity + constant
    }

    /// Returns the hyperparameters used by this model.
    #[allow(dead_code)] // Useful for diagnostics
    pub fn hyperparameters(&self) -> &GpHyperparameters {
        &self.hyperparams
    }
}

/// Optimizes noise variance via log marginal likelihood maximization.
///
/// Length scale is fixed (domain knowledge), only noise is optimized.
/// This prevents overfitting that occurs when length_scale is in the search grid.
///
/// Searches over noise ratios: [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
/// Signal variance is estimated from data variance (with minimum floor).
#[allow(dead_code)] // Use optimize_noise_with_metadata for logging
pub fn optimize_noise(data: &[DataPoint], length_scale: f64) -> Result<GpHyperparameters, GpError> {
    if data.len() < 2 {
        return Err(GpError::InsufficientData(data.len()));
    }

    // Estimate signal variance from data
    let values: Vec<f64> = data.iter().map(|p| p.value).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let signal_variance = variance.max(1.0);

    // Fine grid for noise ratios (cheap since 1D search)
    let noise_ratios = [
        0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
    ];

    let mut best_log_lik = f64::NEG_INFINITY;
    let mut best_hyperparams: Option<GpHyperparameters> = None;

    for &noise_ratio in &noise_ratios {
        let noise_variance = signal_variance * noise_ratio;

        // Try to create valid hyperparameters
        let hyperparams =
            match GpHyperparameters::new(length_scale, signal_variance, noise_variance) {
                Ok(hp) => hp,
                Err(_) => continue,
            };

        // Try to fit the model
        let model = match GpModel::fit(data, hyperparams.clone()) {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Compute log marginal likelihood
        let log_lik = model.log_marginal_likelihood();

        // Check for valid likelihood (not NaN or -Inf)
        if log_lik.is_finite() && log_lik > best_log_lik {
            best_log_lik = log_lik;
            best_hyperparams = Some(hyperparams);
        }
    }

    // Return best or fall back to heuristic estimate
    best_hyperparams.ok_or_else(|| {
        GpError::InvalidHyperparameters(
            "noise optimization found no valid hyperparameters".to_string(),
        )
    })
}

/// Optimized hyperparameters with metadata about the optimization.
#[derive(Debug, Clone)]
pub struct OptimizedHyperparameters {
    pub hyperparams: GpHyperparameters,
    pub log_marginal_likelihood: f64,
    pub noise_ratio: f64,
}

/// Optimizes noise variance and returns additional metadata for logging.
///
/// Length scale is fixed (domain knowledge), only noise is optimized.
/// This prevents overfitting that occurs when length_scale is in the search grid.
pub fn optimize_noise_with_metadata(
    data: &[DataPoint],
    length_scale: f64,
) -> Result<OptimizedHyperparameters, GpError> {
    if data.len() < 2 {
        return Err(GpError::InsufficientData(data.len()));
    }

    // Estimate signal variance from data
    let values: Vec<f64> = data.iter().map(|p| p.value).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let signal_variance = variance.max(1.0);

    // Fine grid for noise ratios (cheap since 1D search)
    let noise_ratios = [
        0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
    ];

    let mut best_log_lik = f64::NEG_INFINITY;
    let mut best_hyperparams: Option<GpHyperparameters> = None;
    let mut best_noise_ratio = 0.1;

    for &noise_ratio in &noise_ratios {
        let noise_variance = signal_variance * noise_ratio;

        let hyperparams =
            match GpHyperparameters::new(length_scale, signal_variance, noise_variance) {
                Ok(hp) => hp,
                Err(_) => continue,
            };

        let model = match GpModel::fit(data, hyperparams.clone()) {
            Ok(m) => m,
            Err(_) => continue,
        };

        let log_lik = model.log_marginal_likelihood();

        if log_lik.is_finite() && log_lik > best_log_lik {
            best_log_lik = log_lik;
            best_hyperparams = Some(hyperparams);
            best_noise_ratio = noise_ratio;
        }
    }

    best_hyperparams
        .map(|hp| OptimizedHyperparameters {
            hyperparams: hp,
            log_marginal_likelihood: best_log_lik,
            noise_ratio: best_noise_ratio,
        })
        .ok_or_else(|| {
            GpError::InvalidHyperparameters(
                "noise optimization found no valid hyperparameters".to_string(),
            )
        })
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

    #[test]
    fn test_log_marginal_likelihood_is_finite() {
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
                value: 108.0,
            },
        ];

        let model = GpModel::fit(&data, GpHyperparameters::default()).unwrap();
        let log_lik = model.log_marginal_likelihood();

        assert!(
            log_lik.is_finite(),
            "Log marginal likelihood should be finite, got {}",
            log_lik
        );
        // Log likelihood is typically negative (it's a log probability)
        assert!(
            log_lik < 0.0,
            "Log marginal likelihood should be negative, got {}",
            log_lik
        );
    }

    #[test]
    fn test_log_marginal_likelihood_prefers_correct_noise() {
        // Generate data with known noise level
        let base_data = vec![
            DataPoint {
                date: make_date(2024, 1, 1),
                value: 100.0,
            },
            DataPoint {
                date: make_date(2024, 2, 1),
                value: 100.5,
            },
            DataPoint {
                date: make_date(2024, 3, 1),
                value: 99.5,
            },
            DataPoint {
                date: make_date(2024, 4, 1),
                value: 100.2,
            },
            DataPoint {
                date: make_date(2024, 5, 1),
                value: 99.8,
            },
        ];

        // With very noisy-looking data (values bounce around), higher noise_variance should fit better
        let signal_var = 1.0;

        // Low noise model
        let hp_low = GpHyperparameters::new(90.0, signal_var, 0.01).unwrap();
        let model_low = GpModel::fit(&base_data, hp_low).unwrap();

        // High noise model
        let hp_high = GpHyperparameters::new(90.0, signal_var, 0.3).unwrap();
        let model_high = GpModel::fit(&base_data, hp_high).unwrap();

        // For data that bounces around a constant, higher noise should be preferred
        assert!(
            model_high.log_marginal_likelihood() > model_low.log_marginal_likelihood(),
            "Higher noise model should have higher likelihood for noisy data"
        );
    }

    #[test]
    fn test_optimize_noise_returns_valid_params() {
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

        let length_scale = 90.0; // Fixed length scale
        let hp = optimize_noise(&data, length_scale).unwrap();

        // Length scale should be what we passed
        assert_eq!(hp.length_scale_days, length_scale);
        assert!(hp.signal_variance > 0.0);
        assert!(hp.noise_variance > 0.0);

        // Noise ratio should be reasonable (between 1% and 50% of signal)
        let noise_ratio = hp.noise_variance / hp.signal_variance;
        assert!(
            noise_ratio >= 0.01 && noise_ratio <= 0.5,
            "Noise ratio {} out of expected range",
            noise_ratio
        );
    }

    #[test]
    fn test_optimize_noise_with_noisy_data() {
        // Data with high daily variation - should prefer higher noise ratio
        let data = vec![
            DataPoint {
                date: make_date(2024, 1, 1),
                value: 80.0,
            },
            DataPoint {
                date: make_date(2024, 1, 2),
                value: 81.5,
            },
            DataPoint {
                date: make_date(2024, 1, 3),
                value: 79.0,
            },
            DataPoint {
                date: make_date(2024, 1, 4),
                value: 82.0,
            },
            DataPoint {
                date: make_date(2024, 1, 5),
                value: 78.5,
            },
            DataPoint {
                date: make_date(2024, 1, 6),
                value: 81.0,
            },
            DataPoint {
                date: make_date(2024, 1, 7),
                value: 79.5,
            },
            DataPoint {
                date: make_date(2024, 1, 8),
                value: 80.5,
            },
        ];

        let length_scale = 60.0; // Body composition length scale
        let result = optimize_noise_with_metadata(&data, length_scale).unwrap();

        // For high daily variation data, optimizer should find higher noise ratio
        assert!(
            result.noise_ratio >= 0.1,
            "Expected higher noise ratio for noisy data, got {}",
            result.noise_ratio
        );
    }

    #[test]
    fn test_optimize_noise_with_smooth_trend() {
        // Data with clear smooth trend - should have low noise ratio
        let data = vec![
            DataPoint {
                date: make_date(2024, 1, 1),
                value: 100.0,
            },
            DataPoint {
                date: make_date(2024, 3, 1),
                value: 105.0,
            },
            DataPoint {
                date: make_date(2024, 5, 1),
                value: 110.0,
            },
            DataPoint {
                date: make_date(2024, 7, 1),
                value: 115.0,
            },
            DataPoint {
                date: make_date(2024, 9, 1),
                value: 120.0,
            },
            DataPoint {
                date: make_date(2024, 11, 1),
                value: 125.0,
            },
        ];

        let length_scale = 90.0;
        let result = optimize_noise_with_metadata(&data, length_scale).unwrap();

        // For smooth trend data with fixed length scale, noise should be relatively low
        assert!(
            result.noise_ratio <= 0.3,
            "Expected lower noise ratio for smooth trend, got {}",
            result.noise_ratio
        );
    }

    #[test]
    fn test_optimize_noise_insufficient_data() {
        let data = vec![DataPoint {
            date: make_date(2024, 1, 1),
            value: 100.0,
        }];

        let result = optimize_noise(&data, 90.0);
        assert!(matches!(result, Err(GpError::InsufficientData(1))));
    }

    #[test]
    fn test_gp_config_default_values() {
        let config = GpConfig::default();

        assert_eq!(config.length_scale_strength, 90.0);
        assert_eq!(config.length_scale_body_comp, 60.0);
        assert_eq!(config.length_scale_energy, 60.0);
    }

    #[test]
    fn test_gp_config_length_scale_for_movements() {
        let config = GpConfig::default();

        // Strength movements
        assert_eq!(config.length_scale_for(Movement::Squat), 90.0);
        assert_eq!(config.length_scale_for(Movement::Bench), 90.0);
        assert_eq!(config.length_scale_for(Movement::Deadlift), 90.0);
        assert_eq!(config.length_scale_for(Movement::Snatch), 90.0);
        assert_eq!(config.length_scale_for(Movement::CleanAndJerk), 90.0);

        // Body composition
        assert_eq!(config.length_scale_for(Movement::Bodyweight), 60.0);
        assert_eq!(config.length_scale_for(Movement::Neck), 60.0);
        assert_eq!(config.length_scale_for(Movement::Waist), 60.0);

        // Energy
        assert_eq!(config.length_scale_for(Movement::Calorie), 60.0);
    }
}
