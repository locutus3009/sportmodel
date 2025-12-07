//! Error types for the sportmodel application.

use thiserror::Error;

/// Errors that can occur when parsing training data.
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("file not found: {0}")]
    FileNotFound(String),

    #[error("cannot read file: {0}")]
    CannotRead(String),

    #[error("invalid Excel format: {0}")]
    InvalidFormat(String),

    #[error("missing required column: {0}")]
    MissingColumn(String),

    #[error("invalid date format in row {row}: {value}")]
    InvalidDate { row: usize, value: String },

    #[error("invalid weight value in row {row}: {value}")]
    InvalidWeight { row: usize, value: String },

    #[error("invalid repetitions value in row {row}: {value}")]
    InvalidReps { row: usize, value: String },

    #[error("unknown movement in row {row}: {value}")]
    UnknownMovement { row: usize, value: String },
}

/// Errors that can occur in formula calculations.
#[derive(Debug, Error)]
#[allow(dead_code)] // Used in Phase 3
#[allow(clippy::enum_variant_names)]
pub enum FormulaError {
    #[error("weight must be positive: {0}")]
    BadWeight(f64),

    #[error("repetitions must be positive: {0}")]
    BadReps(u32),

    #[error("bodyweight must be positive: {0}")]
    BadBodyweight(f64),
}
