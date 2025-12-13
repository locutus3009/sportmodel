//! Excel file parsing for training data.

use calamine::{Data, DataType, Reader, Xlsx, open_workbook};
use chrono::NaiveDate;
use log::warn;
use std::path::Path;
use std::str::FromStr;

use crate::domain::{Movement, Observation, TrainingData};
use crate::error::ParseError;

/// Expected column names (case-insensitive).
const COL_DATE: &str = "date";
const COL_WEIGHT: &str = "weight";
const COL_REPS: &str = "repetitions";
const COL_MOVEMENT: &str = "movement";

/// Finds column indices from the header row.
struct ColumnIndices {
    date: usize,
    weight: usize,
    reps: usize,
    movement: usize,
}

impl ColumnIndices {
    fn from_header(header: &[Data]) -> Result<Self, ParseError> {
        let find_column = |name: &str| -> Result<usize, ParseError> {
            header
                .iter()
                .position(|cell| {
                    cell.get_string()
                        .is_some_and(|s| s.trim().eq_ignore_ascii_case(name))
                })
                .ok_or_else(|| ParseError::MissingColumn(name.to_string()))
        };

        Ok(Self {
            date: find_column(COL_DATE)?,
            weight: find_column(COL_WEIGHT)?,
            reps: find_column(COL_REPS)?,
            movement: find_column(COL_MOVEMENT)?,
        })
    }
}

/// Loads training data from an Excel file.
///
/// # Arguments
/// * `path` - Path to the Excel file (.xlsx)
///
/// # Returns
/// TrainingData containing all valid observations, sorted by date per movement.
///
/// # Errors
/// Returns ParseError if the file cannot be read or has invalid format.
pub fn load_training_data<P: AsRef<Path>>(path: P) -> Result<TrainingData, ParseError> {
    let path = path.as_ref();

    // Check if file exists
    if !path.exists() {
        return Err(ParseError::FileNotFound(path.display().to_string()));
    }

    // Open workbook
    let mut workbook: Xlsx<_> = open_workbook(path)
        .map_err(|e| ParseError::CannotRead(format!("{}: {}", path.display(), e)))?;

    // Get the first worksheet
    let sheet_names = workbook.sheet_names().to_vec();
    let sheet_name = sheet_names
        .first()
        .ok_or_else(|| ParseError::InvalidFormat("workbook has no sheets".to_string()))?;

    let range = workbook.worksheet_range(sheet_name).map_err(|e| {
        ParseError::CannotRead(format!("cannot read sheet '{}': {}", sheet_name, e))
    })?;

    let mut rows = range.rows();

    // Parse header row
    let header = rows
        .next()
        .ok_or_else(|| ParseError::InvalidFormat("empty worksheet".to_string()))?;

    let indices = ColumnIndices::from_header(header)?;

    // Parse data rows
    let mut observations = Vec::new();

    for (row_idx, row) in rows.enumerate() {
        let row_num = row_idx + 2; // +1 for 0-index, +1 for header row

        // Skip empty rows silently (common at end of spreadsheets)
        if row[indices.date] == Data::Empty {
            continue;
        }

        // Parse date
        let date = match parse_date(&row[indices.date], row_num) {
            Ok(d) => d,
            Err(e) => {
                warn!("{}", e);
                continue;
            }
        };

        // Parse movement (needed early to determine if reps is required)
        let movement = match parse_movement(&row[indices.movement], row_num) {
            Ok(m) => m,
            Err(e) => {
                warn!("{}", e);
                continue;
            }
        };

        // Parse weight
        let weight = match parse_weight(&row[indices.weight], row_num) {
            Ok(w) => w,
            Err(e) => {
                warn!("{}", e);
                continue;
            }
        };

        // Parse repetitions (optional for bodyweight)
        let reps = match parse_reps(&row[indices.reps], row_num, movement) {
            Ok(r) => r,
            Err(e) => {
                warn!("{}", e);
                continue;
            }
        };

        observations.push(Observation::new(date, weight, reps, movement));
    }

    Ok(TrainingData::from_observations(observations))
}

/// Loads raw observations from an Excel file.
///
/// Unlike `load_training_data`, this returns raw observations without
/// converting them to data points. Useful for calculations that need
/// the original values (like TDEE calculation).
///
/// # Arguments
/// * `path` - Path to the Excel file (.xlsx)
///
/// # Returns
/// Vec of raw observations, or error if file cannot be read.
pub fn load_observations<P: AsRef<Path>>(path: P) -> Result<Vec<Observation>, ParseError> {
    let path = path.as_ref();

    // Check if file exists
    if !path.exists() {
        return Err(ParseError::FileNotFound(path.display().to_string()));
    }

    // Open workbook
    let mut workbook: Xlsx<_> = open_workbook(path)
        .map_err(|e| ParseError::CannotRead(format!("{}: {}", path.display(), e)))?;

    // Get the first worksheet
    let sheet_names = workbook.sheet_names().to_vec();
    let sheet_name = sheet_names
        .first()
        .ok_or_else(|| ParseError::InvalidFormat("workbook has no sheets".to_string()))?;

    let range = workbook.worksheet_range(sheet_name).map_err(|e| {
        ParseError::CannotRead(format!("cannot read sheet '{}': {}", sheet_name, e))
    })?;

    let mut rows = range.rows();

    // Parse header row
    let header = rows
        .next()
        .ok_or_else(|| ParseError::InvalidFormat("empty worksheet".to_string()))?;

    let indices = ColumnIndices::from_header(header)?;

    // Parse data rows
    let mut observations = Vec::new();

    for (row_idx, row) in rows.enumerate() {
        let row_num = row_idx + 2; // +1 for 0-index, +1 for header row

        // Skip empty rows silently (common at end of spreadsheets)
        if row[indices.date] == Data::Empty {
            continue;
        }

        // Parse date
        let date = match parse_date(&row[indices.date], row_num) {
            Ok(d) => d,
            Err(e) => {
                warn!("{}", e);
                continue;
            }
        };

        // Parse movement (needed early to determine if reps is required)
        let movement = match parse_movement(&row[indices.movement], row_num) {
            Ok(m) => m,
            Err(e) => {
                warn!("{}", e);
                continue;
            }
        };

        // Parse weight
        let weight = match parse_weight(&row[indices.weight], row_num) {
            Ok(w) => w,
            Err(e) => {
                warn!("{}", e);
                continue;
            }
        };

        // Parse repetitions (optional for bodyweight/calorie)
        let reps = match parse_reps(&row[indices.reps], row_num, movement) {
            Ok(r) => r,
            Err(e) => {
                warn!("{}", e);
                continue;
            }
        };

        observations.push(Observation::new(date, weight, reps, movement));
    }

    Ok(observations)
}

/// Parses a date from a cell.
fn parse_date(cell: &Data, row: usize) -> Result<NaiveDate, ParseError> {
    match cell {
        Data::DateTime(dt) => {
            // calamine DateTime to NaiveDate via chrono
            dt.as_datetime()
                .map(|ndt| ndt.date())
                .ok_or_else(|| ParseError::InvalidDate {
                    row,
                    value: format!("{:?}", dt),
                })
        }
        Data::DateTimeIso(s) => {
            NaiveDate::parse_from_str(s, "%Y-%m-%d").map_err(|_| ParseError::InvalidDate {
                row,
                value: s.clone(),
            })
        }
        Data::String(s) => {
            // Try common date formats
            NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .or_else(|_| NaiveDate::parse_from_str(s, "%d/%m/%Y"))
                .or_else(|_| NaiveDate::parse_from_str(s, "%m/%d/%Y"))
                .map_err(|_| ParseError::InvalidDate {
                    row,
                    value: s.clone(),
                })
        }
        Data::Empty => Err(ParseError::InvalidDate {
            row,
            value: "(empty)".to_string(),
        }),
        other => Err(ParseError::InvalidDate {
            row,
            value: format!("{:?}", other),
        }),
    }
}

/// Parses a weight value from a cell.
fn parse_weight(cell: &Data, row: usize) -> Result<f64, ParseError> {
    match cell {
        Data::Float(f) => {
            if *f > 0.0 {
                Ok(*f)
            } else {
                Err(ParseError::InvalidWeight {
                    row,
                    value: f.to_string(),
                })
            }
        }
        Data::Int(i) => {
            if *i > 0 {
                Ok(*i as f64)
            } else {
                Err(ParseError::InvalidWeight {
                    row,
                    value: i.to_string(),
                })
            }
        }
        Data::String(s) => s.parse::<f64>().map_err(|_| ParseError::InvalidWeight {
            row,
            value: s.clone(),
        }),
        Data::Empty => Err(ParseError::InvalidWeight {
            row,
            value: "(empty)".to_string(),
        }),
        other => Err(ParseError::InvalidWeight {
            row,
            value: format!("{:?}", other),
        }),
    }
}

/// Parses repetitions from a cell.
/// Returns None for bodyweight, calorie, neck, and waist movements.
fn parse_reps(cell: &Data, row: usize, movement: Movement) -> Result<Option<u32>, ParseError> {
    // Bodyweight, Calorie, Neck, and Waist don't need reps
    if matches!(
        movement,
        Movement::Bodyweight | Movement::Calorie | Movement::Neck | Movement::Waist
    ) {
        return Ok(None);
    }

    match cell {
        Data::Float(f) => {
            let reps = *f as u32;
            if reps > 0 {
                Ok(Some(reps))
            } else {
                Err(ParseError::InvalidReps {
                    row,
                    value: f.to_string(),
                })
            }
        }
        Data::Int(i) => {
            if *i > 0 {
                Ok(Some(*i as u32))
            } else {
                Err(ParseError::InvalidReps {
                    row,
                    value: i.to_string(),
                })
            }
        }
        Data::String(s) => {
            let reps: u32 = s.parse().map_err(|_| ParseError::InvalidReps {
                row,
                value: s.clone(),
            })?;
            if reps > 0 {
                Ok(Some(reps))
            } else {
                Err(ParseError::InvalidReps {
                    row,
                    value: s.clone(),
                })
            }
        }
        Data::Empty => {
            // Empty reps for a lift defaults to 1 (actual 1RM)
            Ok(Some(1))
        }
        other => Err(ParseError::InvalidReps {
            row,
            value: format!("{:?}", other),
        }),
    }
}

/// Parses a movement from a cell.
fn parse_movement(cell: &Data, row: usize) -> Result<Movement, ParseError> {
    match cell {
        Data::String(s) => Movement::from_str(s).map_err(|_| ParseError::UnknownMovement {
            row,
            value: s.clone(),
        }),
        Data::Empty => Err(ParseError::UnknownMovement {
            row,
            value: "(empty)".to_string(),
        }),
        other => Err(ParseError::UnknownMovement {
            row,
            value: format!("{:?}", other),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_indices_from_header() {
        let header = vec![
            Data::String("Date".to_string()),
            Data::String("Weight".to_string()),
            Data::String("Repetitions".to_string()),
            Data::String("Movement".to_string()),
        ];

        let indices = ColumnIndices::from_header(&header).unwrap();
        assert_eq!(indices.date, 0);
        assert_eq!(indices.weight, 1);
        assert_eq!(indices.reps, 2);
        assert_eq!(indices.movement, 3);
    }

    #[test]
    fn test_column_indices_case_insensitive() {
        let header = vec![
            Data::String("DATE".to_string()),
            Data::String("WEIGHT".to_string()),
            Data::String("REPETITIONS".to_string()),
            Data::String("MOVEMENT".to_string()),
        ];

        let indices = ColumnIndices::from_header(&header);
        assert!(indices.is_ok());
    }

    #[test]
    fn test_column_indices_missing_column() {
        let header = vec![
            Data::String("Date".to_string()),
            Data::String("Weight".to_string()),
            // Missing Repetitions and Movement
        ];

        let indices = ColumnIndices::from_header(&header);
        assert!(indices.is_err());
    }

    #[test]
    fn test_parse_weight_float() {
        let cell = Data::Float(100.5);
        let weight = parse_weight(&cell, 1).unwrap();
        assert_eq!(weight, 100.5);
    }

    #[test]
    fn test_parse_weight_int() {
        let cell = Data::Int(100);
        let weight = parse_weight(&cell, 1).unwrap();
        assert_eq!(weight, 100.0);
    }

    #[test]
    fn test_parse_weight_invalid() {
        let cell = Data::Float(-10.0);
        assert!(parse_weight(&cell, 1).is_err());

        let cell = Data::Empty;
        assert!(parse_weight(&cell, 1).is_err());
    }

    #[test]
    fn test_parse_reps_bodyweight() {
        // Bodyweight should return None regardless of cell content
        let cell = Data::Empty;
        let reps = parse_reps(&cell, 1, Movement::Bodyweight).unwrap();
        assert!(reps.is_none());
    }

    #[test]
    fn test_parse_reps_lift_empty() {
        // Empty reps for a lift defaults to 1
        let cell = Data::Empty;
        let reps = parse_reps(&cell, 1, Movement::Squat).unwrap();
        assert_eq!(reps, Some(1));
    }

    #[test]
    fn test_parse_movement_valid() {
        let cell = Data::String("squat".to_string());
        let movement = parse_movement(&cell, 1).unwrap();
        assert_eq!(movement, Movement::Squat);
    }

    #[test]
    fn test_parse_movement_invalid() {
        let cell = Data::String("unknown".to_string());
        assert!(parse_movement(&cell, 1).is_err());
    }
}
