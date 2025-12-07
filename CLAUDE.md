# sportmodel

Personal strength training analytics tool for Olympic weightlifting and powerlifting. Reads training data from Excel files, applies Gaussian Process regression for trend analysis, and serves a web interface with graphs.

## Project Structure

```
sportmodel/
├── Cargo.toml
├── src/
│   ├── main.rs         # Entry point: loads data, runs GP analysis, starts web server
│   ├── domain.rs       # Domain types (Movement, Observation, DataPoint, TrainingData)
│   ├── error.rs        # Custom error types (ParseError, FormulaError)
│   ├── excel.rs        # Excel file parsing with calamine
│   ├── formulas.rs     # Strength calculation formulas (e1RM, IPF GL, Sinclair)
│   ├── gp.rs           # Gaussian Process regression implementation
│   ├── analysis.rs     # Analysis orchestration and higher-level functions
│   ├── server.rs       # Web server (axum) with REST API, WebSocket, and static serving
│   └── watcher.rs      # File watching with debouncing for live reload
├── static/
│   ├── index.html      # Dashboard HTML with tab navigation
│   ├── style.css       # Dark theme styles
│   └── app.js          # Chart.js visualization, tab logic, and WebSocket client
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
- `FormulaError`: Calculation validation errors

### excel.rs
- `load_training_data`: Main entry point for Excel parsing
- Handles Excel dates, validates columns, skips invalid rows with warnings
- Converts observations to e1RM values during loading

### formulas.rs
- `calculate_e1rm`: Average of Epley, Brzycki, Lander formulas (capped at 15 reps)
- `calculate_ipf_gl`: IPF GoodLift score for powerlifting
- `calculate_sinclair`: Sinclair score for Olympic lifting

### gp.rs
- `GpHyperparameters`: Length scale, signal variance, noise variance
- `GpPrediction`: Mean prediction with standard deviation and 95% CI
- `GpModel`: Fitted GP model with `fit()`, `predict()`, `predict_range()`
- `GpError`: Error types for insufficient data, singular matrix, invalid params
- Uses squared exponential (RBF) kernel with Cholesky decomposition for stability

### analysis.rs
- `MovementAnalysis`: Predictions and data points for a single movement
- `analyze_training_data()`: Fits GP models for all movements
- `PredictionSummary`: Summary stats with CI and trend
- `find_most_reliable_date_*`: Staleness calculation for composite indices

### server.rs
- `AnalysisData`: Mutable data container (training data, analyses, composites)
- `AppState`: Shared state with `RwLock<AnalysisData>`, file path, and WebSocket broadcast
- `WsMessage`: WebSocket message types (`DataUpdated`, `Error`)
- `create_router()`: Configures axum routes, WebSocket endpoint, and static file serving
- `run_server()`: Starts the HTTP server
- `ws_handler()`: WebSocket upgrade handler for live updates
- REST API handlers for movements and composite indices

### watcher.rs
- `WatcherConfig`: Configuration (debounce duration, retry attempts, retry delay)
- `watch_file()`: Watches a file for modifications with debouncing
- `Debouncer`: Collapses rapid successive events into single callbacks
- Uses notify crate for cross-platform file system watching

## Build and Run

```bash
# Build
cargo build

# Run tests
cargo test

# Run with test data (starts web server)
cargo run -- test_data.xlsx 8080

# Run with logging
RUST_LOG=warn cargo run -- test_data.xlsx 8080

# Generate new test data
.venv/bin/python scripts/generate_test_data.py
```

## Web Interface

After starting the server, open http://localhost:8080 in your browser.

### Dashboard Features
- **8 tabs**: Squat, Bench, Deadlift, Snatch, C&J, IPF GL, Sinclair, Bodyweight
- **Charts**: GP regression curve with 95% confidence intervals
- **Observations**: Green dots showing actual measurements
- **Today line**: Dashed vertical line marking the current date
- **Dark theme**: Easy on the eyes
- **Live reload**: Charts automatically refresh when Excel file changes
- **Connection indicator**: Green dot shows live connection status
- **Toast notifications**: Brief popup when data updates

### API Endpoints

```bash
# List all movements with summary
curl http://localhost:8080/api/movements

# Get full data for a specific movement
curl http://localhost:8080/api/movement/squat
curl http://localhost:8080/api/movement/bench
curl http://localhost:8080/api/movement/deadlift
curl http://localhost:8080/api/movement/snatch
curl http://localhost:8080/api/movement/cj
curl http://localhost:8080/api/movement/bodyweight

# Get composite indices (IPF GL and Sinclair)
curl http://localhost:8080/api/composites

# WebSocket endpoint for live updates
# Connect to ws://localhost:8080/ws
# Receives "reload" message when data changes
```

## Live Reload

The server watches the Excel file for modifications and automatically reloads data when changes are detected.

### How It Works
1. File watcher monitors the Excel file using the notify crate
2. Changes are debounced (2 second window) to handle rapid successive writes
3. On change, data is reloaded with retry logic (3 attempts, 500ms delay)
4. WebSocket broadcasts "reload" message to all connected browsers
5. Frontend clears cache and refreshes the current chart

### Debouncing
Syncthing and Excel may trigger multiple file system events for a single save:
- File created (temp file)
- File modified (multiple writes)
- File renamed (atomic replace)

The debouncer collapses these into a single reload after activity settles.

### Error Handling
- If reload fails, retries up to 3 times with 500ms delay between attempts
- On permanent failure, WebSocket receives "error:message" and toast shows error
- Old data remains displayed until successful reload

### Connection States
- **Live** (green): WebSocket connected, receiving updates
- **Offline** (red): WebSocket disconnected, no live updates
- **Reconnecting** (orange): Attempting to reconnect with exponential backoff

## Data Flow

1. **Input**: Excel file with columns Date, Weight, Repetitions, Movement
2. **Parsing**: `excel::load_training_data` reads file, validates, creates Observations
3. **Processing**: `TrainingData::from_observations` converts to DataPoints (e1RM values)
4. **GP Regression**: `analyze_training_data()` fits GP models per movement
5. **Composite Indices**: IPF GL and Sinclair calculated from component predictions
6. **Web Server**: Serves JSON API, WebSocket, and static frontend
7. **Visualization**: Chart.js renders predictions with confidence intervals
8. **Live Updates**: File watcher detects changes, reloads data, notifies browsers via WebSocket

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

### Prediction Range
- 12 months into the past
- 6 months into the future
- Daily resolution

## Composite Indices

### IPF GoodLift (Powerlifting)
- Requires: Squat, Bench, Deadlift, Bodyweight predictions
- Formula: `GL = Total × 100 / (A - B × e^(-C × BW))`
- "Most reliable date": When all component lifts were most recently measured

### Sinclair (Olympic Weightlifting)
- Requires: Snatch, Clean & Jerk, Bodyweight predictions
- Formula: `Sinclair = Total × 10^(A × (log10(BW/B))²)` for BW < B
- "Most reliable date": When both lifts were most recently measured

### Uncertainty Propagation
Composite index uncertainty uses the maximum relative uncertainty from component lifts.

## Troubleshooting

### Live reload not working
- Check the connection indicator in the top-right corner
- If "Offline", the WebSocket connection may be blocked by a proxy
- Try refreshing the page to reconnect
- Check server logs with `RUST_LOG=info` for file watcher messages

### Charts not updating
- Verify the Excel file was actually saved (not just opened)
- Wait for debounce timeout (2 seconds after last file activity)
- Check browser console for WebSocket errors
- Clear browser cache and refresh

### Connection indicator stuck on "Reconnecting"
- Server may have crashed - check terminal for errors
- Port may be blocked - try a different port
- Browser may be blocking WebSocket - check browser security settings

### File watcher errors
- Ensure the Excel file path is valid and accessible
- On Linux, check inotify limits: `cat /proc/sys/fs/inotify/max_user_watches`
- Increase limit if needed: `sudo sysctl fs.inotify.max_user_watches=524288`
