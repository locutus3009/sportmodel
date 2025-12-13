//! Domain types for training data representation.

use chrono::NaiveDate;
use std::collections::HashMap;
use std::str::FromStr;

use crate::error::ParseError;
use crate::formulas::calculate_e1rm;

/// Training movements tracked by the application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Movement {
    Bodyweight,
    Squat,
    Bench,
    Deadlift,
    Snatch,
    CleanAndJerk,
    Calorie,
    Neck,
    Waist,
}

impl Movement {
    /// Returns all movement variants.
    pub fn all() -> &'static [Movement] {
        &[
            Movement::Bodyweight,
            Movement::Squat,
            Movement::Bench,
            Movement::Deadlift,
            Movement::Snatch,
            Movement::CleanAndJerk,
            Movement::Calorie,
            Movement::Neck,
            Movement::Waist,
        ]
    }

    /// Returns the display name for the movement.
    pub fn display_name(&self) -> &'static str {
        match self {
            Movement::Bodyweight => "Bodyweight",
            Movement::Squat => "Squat",
            Movement::Bench => "Bench",
            Movement::Deadlift => "Deadlift",
            Movement::Snatch => "Snatch",
            Movement::CleanAndJerk => "Clean & Jerk",
            Movement::Calorie => "Calorie",
            Movement::Neck => "Neck",
            Movement::Waist => "Waist",
        }
    }
}

impl FromStr for Movement {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "bodyweight" => Ok(Movement::Bodyweight),
            "squat" => Ok(Movement::Squat),
            "bench" => Ok(Movement::Bench),
            "deadlift" => Ok(Movement::Deadlift),
            "snatch" => Ok(Movement::Snatch),
            "cj" | "clean and jerk" | "cleanandjerk" => Ok(Movement::CleanAndJerk),
            "calorie" | "calories" => Ok(Movement::Calorie),
            "neck" => Ok(Movement::Neck),
            "waist" => Ok(Movement::Waist),
            _ => Err(ParseError::UnknownMovement {
                row: 0,
                value: s.to_string(),
            }),
        }
    }
}

impl std::fmt::Display for Movement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// A raw observation from the Excel file.
#[derive(Debug, Clone)]
pub struct Observation {
    pub date: NaiveDate,
    pub weight_kg: f64,
    pub repetitions: Option<u32>,
    pub movement: Movement,
}

impl Observation {
    /// Creates a new observation.
    pub fn new(
        date: NaiveDate,
        weight_kg: f64,
        repetitions: Option<u32>,
        movement: Movement,
    ) -> Self {
        Self {
            date,
            weight_kg,
            repetitions,
            movement,
        }
    }

    /// Converts this observation to a data point.
    /// For bodyweight, returns the raw weight.
    /// For calories, returns the calorie value (stored in weight_kg field).
    /// For neck/waist, returns the raw measurement value (in cm).
    /// For lifts, calculates e1RM.
    pub fn to_data_point(&self) -> DataPoint {
        let value = match self.movement {
            Movement::Bodyweight | Movement::Calorie | Movement::Neck | Movement::Waist => {
                self.weight_kg
            }
            _ => {
                let reps = self.repetitions.unwrap_or(1);
                calculate_e1rm(self.weight_kg, reps)
            }
        };

        DataPoint {
            date: self.date,
            value,
        }
    }
}

/// A processed data point ready for GP regression.
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub date: NaiveDate,
    #[allow(dead_code)] // Used in Phase 2 for GP regression
    pub value: f64,
}

/// Container for all parsed training data, organized by movement.
#[derive(Debug, Clone, Default)]
pub struct TrainingData {
    data: HashMap<Movement, Vec<DataPoint>>,
}

impl TrainingData {
    /// Creates a new empty TrainingData container.
    #[allow(dead_code)] // Useful utility for Phase 2/3
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Creates TrainingData from a list of observations.
    /// Converts observations to data points and sorts by date.
    pub fn from_observations(observations: Vec<Observation>) -> Self {
        let mut data: HashMap<Movement, Vec<DataPoint>> = HashMap::new();

        for obs in observations {
            let point = obs.to_data_point();
            data.entry(obs.movement).or_default().push(point);
        }

        // Sort each movement's data by date
        for points in data.values_mut() {
            points.sort_by_key(|p| p.date);
        }

        Self { data }
    }

    /// Returns data points for a specific movement.
    #[allow(dead_code)] // Used in Phase 2 for GP regression
    pub fn get(&self, movement: Movement) -> Option<&[DataPoint]> {
        self.data.get(&movement).map(|v| v.as_slice())
    }

    /// Returns the number of data points for a specific movement.
    pub fn count(&self, movement: Movement) -> usize {
        self.data.get(&movement).map(|v| v.len()).unwrap_or(0)
    }

    /// Returns the total number of data points across all movements.
    pub fn total_count(&self) -> usize {
        self.data.values().map(|v| v.len()).sum()
    }

    /// Returns the date range for a specific movement.
    pub fn date_range(&self, movement: Movement) -> Option<(NaiveDate, NaiveDate)> {
        self.data.get(&movement).and_then(|points| {
            if points.is_empty() {
                None
            } else {
                Some((points.first().unwrap().date, points.last().unwrap().date))
            }
        })
    }

    /// Returns the overall date range across all movements.
    pub fn overall_date_range(&self) -> Option<(NaiveDate, NaiveDate)> {
        let mut min_date: Option<NaiveDate> = None;
        let mut max_date: Option<NaiveDate> = None;

        for points in self.data.values() {
            for point in points {
                min_date = Some(min_date.map_or(point.date, |d| d.min(point.date)));
                max_date = Some(max_date.map_or(point.date, |d| d.max(point.date)));
            }
        }

        min_date.zip(max_date)
    }

    /// Returns all movements that have data.
    #[allow(dead_code)] // Used in Phase 3 for web server
    pub fn movements_with_data(&self) -> Vec<Movement> {
        self.data
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(m, _)| *m)
            .collect()
    }

    /// Returns an iterator over all movement data.
    #[allow(dead_code)] // Used in Phase 2/3
    pub fn iter(&self) -> impl Iterator<Item = (&Movement, &Vec<DataPoint>)> {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_movement_from_str_lowercase() {
        assert_eq!(Movement::from_str("squat").unwrap(), Movement::Squat);
        assert_eq!(Movement::from_str("bench").unwrap(), Movement::Bench);
        assert_eq!(Movement::from_str("deadlift").unwrap(), Movement::Deadlift);
        assert_eq!(Movement::from_str("snatch").unwrap(), Movement::Snatch);
        assert_eq!(Movement::from_str("cj").unwrap(), Movement::CleanAndJerk);
        assert_eq!(
            Movement::from_str("bodyweight").unwrap(),
            Movement::Bodyweight
        );
    }

    #[test]
    fn test_movement_from_str_uppercase() {
        assert_eq!(Movement::from_str("SQUAT").unwrap(), Movement::Squat);
        assert_eq!(Movement::from_str("BENCH").unwrap(), Movement::Bench);
    }

    #[test]
    fn test_movement_from_str_mixed_case() {
        assert_eq!(Movement::from_str("Squat").unwrap(), Movement::Squat);
        assert_eq!(Movement::from_str("DeadLift").unwrap(), Movement::Deadlift);
    }

    #[test]
    fn test_movement_from_str_with_whitespace() {
        assert_eq!(Movement::from_str("  squat  ").unwrap(), Movement::Squat);
    }

    #[test]
    fn test_movement_from_str_invalid() {
        assert!(Movement::from_str("invalid").is_err());
        assert!(Movement::from_str("").is_err());
    }

    #[test]
    fn test_training_data_from_observations() {
        let obs = vec![
            Observation::new(
                NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
                100.0,
                Some(5),
                Movement::Squat,
            ),
            Observation::new(
                NaiveDate::from_ymd_opt(2024, 1, 10).unwrap(),
                95.0,
                Some(5),
                Movement::Squat,
            ),
            Observation::new(
                NaiveDate::from_ymd_opt(2024, 1, 12).unwrap(),
                80.0,
                None,
                Movement::Bodyweight,
            ),
        ];

        let data = TrainingData::from_observations(obs);

        // Check counts
        assert_eq!(data.count(Movement::Squat), 2);
        assert_eq!(data.count(Movement::Bodyweight), 1);
        assert_eq!(data.count(Movement::Bench), 0);

        // Check sorting (earlier date should be first)
        let squats = data.get(Movement::Squat).unwrap();
        assert!(squats[0].date < squats[1].date);
    }
}
