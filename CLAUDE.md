# sportmodel

Personal strength training analytics tool for Olympic weightlifting and powerlifting. Reads training data from Excel files, applies Gaussian Process regression for trend analysis, and serves a web interface with graphs.

## Project Structure

```
sportmodel/
├── Cargo.toml
├── src/
│   ├── main.rs         # CLI entry point with GP analysis output
│   ├── domain.rs       # Domain types (Movement, Observation, DataPoint, TrainingData)
│   ├── error.rs        # Custom error types (ParseError, FormulaError)
│   ├── excel.rs        # Excel file parsing with calamine
│   ├── formulas.rs     # Strength calculation formulas (e1RM, IPF GL, Sinclair)
│   ├── gp.rs           # Gaussian Process regression implementation
│   └── analysis.rs     # Analysis orchestration and higher-level functions
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

### gp.rs
- `GpHyperparameters`: Length scale, signal variance, noise variance
- `GpPrediction`: Mean prediction with standard deviation and 95% CI
- `GpModel`: Fitted GP model with `fit()`, `predict()`, `predict_range()`
- `GpError`: Error types for insufficient data, singular matrix, invalid params
- Uses squared exponential (RBF) kernel with Cholesky decomposition for stability

### analysis.rs
- `MovementAnalysis`: Predictions and data points for a single movement
- `analyze_training_data()`: Fits GP models for all movements
- `PredictionSummary`: Summary stats with CI and trend (for Phase 3)
- Staleness calculation functions for composite indices (Phase 3)

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
4. **GP Regression**: `analyze_training_data()` fits GP models per movement
5. **Output**: Daily predictions with 95% confidence intervals

## Input Format

Excel file (.xlsx) with columns:
- **Date**: Excel date or ISO format string
- **Weight**: Numeric weight in kg
- **Repetitions**: Integer (empty defaults to 1 for lifts, ignored for bodyweight)
- **Movement**: One of: bodyweight, squat, bench, deadlift, snatch, cj

## GP Regression

### Hyperparameters
- **Length scale**: 90 days (strength changes over months)
- **Signal variance**: Estimated from data variance (min floor of 1.0)
- **Noise variance**: 5% of signal variance (measurement noise + daily variation)

### Kernel
Squared exponential (RBF): `k(x, x') = σ² × exp(-0.5 × (x - x')² / l²)`

### Numerical Stability
- Cholesky decomposition for matrix inversion (more stable than direct inverse)
- Automatic jitter (1e-6 to 1e-4) added to diagonal if decomposition fails
- Minimum 2 observations required per movement

### Prediction Output
For each movement with sufficient data:
- Prediction at last observation (validation)
- Prediction for today (if in range)
- Prediction 3 months into future
- 95% confidence intervals on all predictions

## Phase 3 Preparation (Web Server)

Ready for Phase 3:
- `calculate_ipf_gl` and `calculate_sinclair` functions defined
- `Movement::all()` for iterating all movement types
- `TrainingData::overall_date_range()` for visualization bounds
- `MovementAnalysis` with predictions ready for JSON serialization
- `PredictionSummary` for API response formatting
- `find_most_reliable_date_*` for composite index staleness tracking

Web server will need:
- axum or actix-web for HTTP
- serde for JSON serialization
- Chart.js or similar for frontend graphs
- Static file serving for HTML/JS assets
