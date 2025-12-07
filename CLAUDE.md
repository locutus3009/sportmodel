# sportmodel

Personal strength training analytics tool for Olympic weightlifting and powerlifting. Reads training data from Excel files, applies Gaussian Process regression for trend analysis, and serves a web interface with graphs.

## Project Structure

```
sportmodel/
├── Cargo.toml
├── src/
│   ├── main.rs         # CLI entry point
│   ├── domain.rs       # Domain types (Movement, Observation, DataPoint, TrainingData)
│   ├── error.rs        # Custom error types (ParseError, FormulaError)
│   ├── excel.rs        # Excel file parsing with calamine
│   └── formulas.rs     # Strength calculation formulas (e1RM, IPF GL, Sinclair)
├── scripts/
│   └── generate_test_data.py  # Test data generator
└── test_data.xlsx      # Sample test file
```

## Module Responsibilities

### domain.rs
- `Movement` enum: Bodyweight, Squat, Bench, Deadlift, Snatch, CleanAndJerk
- `Observation`: Raw data from Excel (date, weight, reps, movement)
- `DataPoint`: Processed data ready for GP regression (date, value)
- `TrainingData`: Container organizing data by movement, sorted by date

### error.rs
- `ParseError`: File/Excel parsing errors with row context
- `FormulaError`: Calculation validation errors (Phase 3)

### excel.rs
- `load_training_data`: Main entry point for Excel parsing
- Handles Excel dates, validates columns, skips invalid rows with warnings
- Converts observations to e1RM values during loading

### formulas.rs
- `calculate_e1rm`: Average of Epley, Brzycki, Lander formulas (capped at 15 reps)
- `calculate_ipf_gl`: IPF GoodLift score for powerlifting (Phase 3)
- `calculate_sinclair`: Sinclair score for Olympic lifting (Phase 3)

## Build and Run

```bash
# Build
cargo build

# Run tests
cargo test

# Run with test data
cargo run -- test_data.xlsx 8080

# Run with logging
RUST_LOG=warn cargo run -- test_data.xlsx 8080

# Generate new test data
.venv/bin/python scripts/generate_test_data.py
```

## Data Flow

1. **Input**: Excel file with columns Date, Weight, Repetitions, Movement
2. **Parsing**: `excel::load_training_data` reads file, validates, creates Observations
3. **Processing**: `TrainingData::from_observations` converts to DataPoints (e1RM values)
4. **Output**: Sorted data per movement, ready for GP regression

## Input Format

Excel file (.xlsx) with columns:
- **Date**: Excel date or ISO format string
- **Weight**: Numeric weight in kg
- **Repetitions**: Integer (empty defaults to 1 for lifts, ignored for bodyweight)
- **Movement**: One of: bodyweight, squat, bench, deadlift, snatch, cj

## Phase 2 Preparation (GP Regression)

The following are ready for Phase 2:

1. **Data Access**: `TrainingData::get(movement)` returns sorted `&[DataPoint]`
2. **Data Structure**: `DataPoint { date: NaiveDate, value: f64 }` - value is e1RM
3. **Iterator**: `TrainingData::iter()` for processing all movements

GP regression will need:
- Convert dates to numeric (days since epoch or similar)
- Implement kernel functions (RBF/squared exponential recommended)
- Matrix operations for GP inference (consider nalgebra or ndarray)
- Prediction grid (daily points from min_date to max_date + 6 months)

## Phase 3 Preparation (Web Server)

Ready for Phase 3:
- `calculate_ipf_gl` and `calculate_sinclair` functions defined
- `Movement::all()` for iterating all movement types
- `TrainingData::overall_date_range()` for visualization bounds

Web server will need:
- axum or actix-web for HTTP
- Chart.js or similar for frontend graphs
- JSON serialization for API responses
