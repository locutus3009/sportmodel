//! Strength calculation formulas for e1RM and composite indices.

/// Maximum reps for which e1RM formulas are considered reliable.
const MAX_RELIABLE_REPS: u32 = 15;

/// IPF GL coefficients for male raw lifters.
#[allow(dead_code)] // Used in Phase 3
mod ipf_gl {
    pub const A: f64 = 1199.72839;
    pub const B: f64 = 1025.18162;
    pub const C: f64 = 0.00921;
}

/// Sinclair coefficients for male lifters (2021-2024 cycle).
#[allow(dead_code)] // Used in Phase 3
mod sinclair {
    pub const A: f64 = 0.722762521;
    pub const B: f64 = 193.609; // World record holder bodyweight
}

/// Calculates estimated 1RM from weight and reps using the average of
/// Epley, Brzycki, and Lander formulas.
///
/// For reps == 1, returns the weight directly (actual 1RM).
/// For reps > MAX_RELIABLE_REPS (15), caps reps at 15 as formulas become unreliable.
///
/// # Arguments
/// * `weight_kg` - Weight lifted in kilograms
/// * `reps` - Number of repetitions performed
///
/// # Returns
/// Estimated 1RM in kilograms
pub fn calculate_e1rm(weight_kg: f64, reps: u32) -> f64 {
    if weight_kg <= 0.0 {
        return 0.0;
    }

    if reps == 0 {
        return 0.0;
    }

    if reps == 1 {
        return weight_kg;
    }

    // Cap reps at MAX_RELIABLE_REPS
    let r = reps.min(MAX_RELIABLE_REPS) as f64;
    let w = weight_kg;

    // Epley: w × (1 + r/30)
    let epley = w * (1.0 + r / 30.0);

    // Brzycki: w × 36 / (37 - r)
    // Guard against division by zero (r = 37, but capped at 15)
    let brzycki = w * 36.0 / (37.0 - r);

    // Lander: w × 100 / (101.3 - 2.67 × r)
    // Guard against division by zero (r ≈ 37.9, but capped at 15)
    let lander = w * 100.0 / (101.3 - 2.67 * r);

    // Average of all three formulas
    (epley + brzycki + lander) / 3.0
}

/// Calculates IPF GoodLift score for powerlifting total.
///
/// Formula: GL = Total × 100 / (A - B × e^(-C × BW))
///
/// # Arguments
/// * `squat_e1rm` - Estimated 1RM for squat in kg
/// * `bench_e1rm` - Estimated 1RM for bench press in kg
/// * `deadlift_e1rm` - Estimated 1RM for deadlift in kg
/// * `bodyweight_kg` - Athlete's bodyweight in kg
///
/// # Returns
/// IPF GL score, or None if inputs are invalid
#[allow(dead_code)] // Used in Phase 3
pub fn calculate_ipf_gl(
    squat_e1rm: f64,
    bench_e1rm: f64,
    deadlift_e1rm: f64,
    bodyweight_kg: f64,
) -> Option<f64> {
    // Validate inputs
    if squat_e1rm <= 0.0 || bench_e1rm <= 0.0 || deadlift_e1rm <= 0.0 {
        return None;
    }
    if bodyweight_kg <= 0.0 {
        return None;
    }

    let total = squat_e1rm + bench_e1rm + deadlift_e1rm;
    let denominator = ipf_gl::A - ipf_gl::B * (-ipf_gl::C * bodyweight_kg).exp();

    // Guard against division by zero or negative denominator
    if denominator <= 0.0 {
        return None;
    }

    Some(total * 100.0 / denominator)
}

/// Calculates Sinclair score for Olympic lifting total.
///
/// Formula:
/// - If BW < B: Sinclair = Total × 10^(A × (log10(BW/B))²)
/// - If BW >= B: Sinclair = Total
///
/// # Arguments
/// * `snatch_e1rm` - Estimated 1RM for snatch in kg
/// * `cj_e1rm` - Estimated 1RM for clean & jerk in kg
/// * `bodyweight_kg` - Athlete's bodyweight in kg
///
/// # Returns
/// Sinclair score, or None if inputs are invalid
#[allow(dead_code)] // Used in Phase 3
pub fn calculate_sinclair(snatch_e1rm: f64, cj_e1rm: f64, bodyweight_kg: f64) -> Option<f64> {
    // Validate inputs
    if snatch_e1rm <= 0.0 || cj_e1rm <= 0.0 {
        return None;
    }
    if bodyweight_kg <= 0.0 {
        return None;
    }

    let total = snatch_e1rm + cj_e1rm;

    // If bodyweight >= world record holder bodyweight, no adjustment needed
    if bodyweight_kg >= sinclair::B {
        return Some(total);
    }

    // Sinclair = Total × 10^(A × (log10(BW/B))²)
    let log_ratio = (bodyweight_kg / sinclair::B).log10();
    let exponent = sinclair::A * log_ratio * log_ratio;
    let multiplier = 10.0_f64.powf(exponent);

    Some(total * multiplier)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to check floating point equality with tolerance
    fn approx_eq(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    #[test]
    fn test_e1rm_single_rep() {
        // For 1 rep, should return the weight directly
        assert_eq!(calculate_e1rm(100.0, 1), 100.0);
        assert_eq!(calculate_e1rm(150.0, 1), 150.0);
    }

    #[test]
    fn test_e1rm_zero_reps() {
        assert_eq!(calculate_e1rm(100.0, 0), 0.0);
    }

    #[test]
    fn test_e1rm_zero_weight() {
        assert_eq!(calculate_e1rm(0.0, 5), 0.0);
    }

    #[test]
    fn test_e1rm_negative_weight() {
        assert_eq!(calculate_e1rm(-100.0, 5), 0.0);
    }

    #[test]
    fn test_e1rm_five_reps() {
        // 100kg × 5 reps
        // Epley: 100 × (1 + 5/30) = 100 × 1.1667 = 116.67
        // Brzycki: 100 × 36 / (37 - 5) = 100 × 36 / 32 = 112.5
        // Lander: 100 × 100 / (101.3 - 2.67 × 5) = 100 × 100 / 87.95 = 113.70
        // Average: (116.67 + 112.5 + 113.70) / 3 ≈ 114.29
        let e1rm = calculate_e1rm(100.0, 5);
        assert!(approx_eq(e1rm, 114.29, 0.5));
    }

    #[test]
    fn test_e1rm_ten_reps() {
        // 100kg × 10 reps
        // Epley: 100 × (1 + 10/30) = 100 × 1.333 = 133.33
        // Brzycki: 100 × 36 / (37 - 10) = 100 × 36 / 27 = 133.33
        // Lander: 100 × 100 / (101.3 - 26.7) = 100 × 100 / 74.6 = 134.05
        // Average: (133.33 + 133.33 + 134.05) / 3 ≈ 133.57
        let e1rm = calculate_e1rm(100.0, 10);
        assert!(approx_eq(e1rm, 133.57, 0.5));
    }

    #[test]
    fn test_e1rm_capped_at_15_reps() {
        // 20 reps should be capped at 15
        let e1rm_15 = calculate_e1rm(100.0, 15);
        let e1rm_20 = calculate_e1rm(100.0, 20);
        assert_eq!(e1rm_15, e1rm_20);
    }

    #[test]
    fn test_ipf_gl_valid_inputs() {
        // Test with reasonable powerlifting numbers
        // 180kg squat, 120kg bench, 220kg deadlift = 520kg total at 85kg BW
        let gl = calculate_ipf_gl(180.0, 120.0, 220.0, 85.0);
        assert!(gl.is_some());
        let score = gl.unwrap();
        // GL scores typically range from 50-150 for competitive lifters
        assert!(score > 50.0 && score < 150.0);
    }

    #[test]
    fn test_ipf_gl_invalid_inputs() {
        assert!(calculate_ipf_gl(0.0, 120.0, 220.0, 85.0).is_none());
        assert!(calculate_ipf_gl(180.0, -10.0, 220.0, 85.0).is_none());
        assert!(calculate_ipf_gl(180.0, 120.0, 220.0, 0.0).is_none());
        assert!(calculate_ipf_gl(180.0, 120.0, 220.0, -85.0).is_none());
    }

    #[test]
    fn test_sinclair_valid_inputs() {
        // Test with reasonable Olympic lifting numbers
        // 100kg snatch, 130kg C&J = 230kg total at 81kg BW
        let sinclair = calculate_sinclair(100.0, 130.0, 81.0);
        assert!(sinclair.is_some());
        let score = sinclair.unwrap();
        // Sinclair should be higher than raw total for lighter lifters
        assert!(score > 230.0);
    }

    #[test]
    fn test_sinclair_heavy_lifter() {
        // For lifters at or above the reference bodyweight (193.609kg),
        // Sinclair equals the raw total
        let sinclair = calculate_sinclair(150.0, 200.0, 200.0);
        assert!(sinclair.is_some());
        let score = sinclair.unwrap();
        assert_eq!(score, 350.0); // 150 + 200
    }

    #[test]
    fn test_sinclair_invalid_inputs() {
        assert!(calculate_sinclair(0.0, 130.0, 81.0).is_none());
        assert!(calculate_sinclair(100.0, -10.0, 81.0).is_none());
        assert!(calculate_sinclair(100.0, 130.0, 0.0).is_none());
    }
}
