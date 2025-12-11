# sportmodel

Personal strength training analytics tool for Olympic weightlifting and powerlifting. Reads training data from Excel files, applies Gaussian Process regression for trend analysis, and serves a web interface with graphs.

## Project Structure

```
sportmodel/
├── Cargo.toml
├── sportmodel.service  # Systemd user service unit file
├── src/
│   ├── main.rs         # Entry point: loads data, runs GP analysis, starts web server
│   ├── domain.rs       # Domain types (Movement, Observation, DataPoint, TrainingData)
│   ├── error.rs        # Custom error types (ParseError, FormulaError)
│   ├── excel.rs        # Excel file parsing with calamine
│   ├── formulas.rs     # Strength calculation formulas (e1RM, IPF GL, Sinclair)
│   ├── gp.rs           # Gaussian Process regression implementation
│   ├── analysis.rs     # Analysis orchestration and higher-level functions
│   ├── server.rs       # Web server (axum) with REST API, WebSocket, and static serving
│   ├── tdee.rs         # TDEE calculation from calorie and weight data
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
- `Movement` enum: Bodyweight, Squat, Bench, Deadlift, Snatch, CleanAndJerk, Calorie
- `Observation`: Raw data from Excel (date, weight, reps, movement)
- `DataPoint`: Processed data ready for GP regression (date, value)
- `TrainingData`: Container organizing data by movement, sorted by date

### error.rs
- `ParseError`: File/Excel parsing errors with row context
- `FormulaError`: Calculation validation errors

### excel.rs
- `load_training_data`: Main entry point for Excel parsing
- `load_observations`: Returns raw observations without conversion (used by TDEE)
- Handles Excel dates, validates columns, skips invalid rows with warnings
- Converts observations to e1RM values during loading

### formulas.rs
- `calculate_e1rm`: Average of Epley, Brzycki, Lander formulas (capped at 15 reps)
- `calculate_ipf_gl`: IPF GoodLift score for powerlifting
- `calculate_sinclair`: Sinclair score for Olympic lifting

### gp.rs
- `GpHyperparameters`: Length scale, signal variance, noise variance
- `GpPrediction`: Mean prediction with standard deviation (used for sigma bands)
- `GpModel`: Fitted GP model with `fit()`, `predict()`, `predict_range()`
- `GpError`: Error types for insufficient data, singular matrix, invalid params
- Uses squared exponential (RBF) kernel with Cholesky decomposition for stability

### analysis.rs
- `MovementAnalysis`: Predictions and data points for a single movement
- `analyze_training_data()`: Fits GP models for all movements
- `PredictionSummary`: Summary stats with CI and trend
- `find_most_reliable_date_*`: Staleness calculation for composite indices

### server.rs
- `AnalysisData`: Mutable data container (training data, analyses, composites, TDEE)
- `AppState`: Shared state with `RwLock<AnalysisData>`, file path, and WebSocket broadcast
- `WsMessage`: WebSocket message types (`DataUpdated`, `Error`)
- `create_router()`: Configures axum routes, WebSocket endpoint, and static file serving
- `run_server()`: Starts the HTTP server
- `ws_handler()`: WebSocket upgrade handler for live updates
- REST API handlers for movements, composite indices, and TDEE

### tdee.rs
- `TdeeResult`: Calculated TDEE with smoothed average, today's TDEE, EMA values, weight change, pairs count
- `TdeeError`: Detailed error types (insufficient data, span too short, etc.)
- `calculate_tdee()`: Main entry point for TDEE calculation
- `calculate_tdee_for_date()`: Internal helper for single-date TDEE calculation
- Uses 28-day window with 10-day EMA smoothing for weight trends
- TDEE smoothing: calculates TDEE for last 10 days, applies EMA to produce smoothed result
- EMA boundary handling: initializes from last available weight in window, processes backwards
- Requires minimum 3 data points in each 10-day EMA window
- Requires at least 50% valid calorie-weight pairs (14/28 days)

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

## Running as a Systemd User Service

Sportmodel can run as a systemd user daemon for automatic startup and background operation.

### Installation

```bash
# Build release binary
cargo build --release

# Create systemd user directory if needed
mkdir -p ~/.config/systemd/user

# Copy the service file
cp sportmodel.service ~/.config/systemd/user/

# Edit the service file to configure paths
# REQUIRED: Set SPORTMODEL_FILE to your Excel data file
# OPTIONAL: Adjust WorkingDirectory and ExecStart paths
nano ~/.config/systemd/user/sportmodel.service

# Reload systemd to pick up the new service
systemctl --user daemon-reload
```

### Configuration

The service uses environment variables for configuration:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SPORTMODEL_FILE` | Yes | - | Path to Excel training data file |
| `SPORTMODEL_PORT` | No | 8080 | Web server port |
| `RUST_LOG` | No | - | Log level (error, warn, info, debug) |

Edit `~/.config/systemd/user/sportmodel.service` to set these values. The `%h` placeholder expands to your home directory.

### Service Commands

```bash
# Start the service
systemctl --user start sportmodel

# Stop the service
systemctl --user stop sportmodel

# Check status
systemctl --user status sportmodel

# Enable auto-start on login
systemctl --user enable sportmodel

# Disable auto-start
systemctl --user disable sportmodel

# View logs
journalctl --user -u sportmodel

# Follow logs in real-time
journalctl --user -u sportmodel -f

# View recent logs
journalctl --user -u sportmodel --since "1 hour ago"
```

### Graceful Shutdown

The server handles SIGTERM gracefully, allowing in-flight requests to complete before shutting down. This ensures clean stops when using `systemctl --user stop sportmodel`.

### Troubleshooting Service Issues

**Service fails to start:**
```bash
# Check logs for errors
journalctl --user -u sportmodel -n 50

# Common issues:
# - SPORTMODEL_FILE path doesn't exist
# - Port already in use
# - WorkingDirectory doesn't contain 'static' folder
```

**Service starts but web interface doesn't work:**
```bash
# Verify the service is running
systemctl --user status sportmodel

# Check if port is listening
ss -tlnp | grep 8080

# Test API endpoint
curl http://localhost:8080/api/movements
```

**Logs not appearing:**
```bash
# Ensure RUST_LOG is set in the service file
# Add: Environment=RUST_LOG=info
```

## Web Interface

After starting the server, open http://localhost:8080 in your browser.

### Dashboard Features
- **8 tabs**: Squat, Bench, Deadlift, Snatch, C&J, IPF GL, Sinclair, Bodyweight
- **TDEE display**: Header shows smoothed TDEE (hover for today's raw TDEE and details)
- **Charts**: GP regression curve with three sigma bands (1σ, 2σ, 3σ)
- **Observations**: Green dots showing actual measurements
- **Today line**: Dashed vertical line marking the current date
- **Range controls**: Button groups to adjust history depth (1-5 years) and prediction length (6/12 months)
- **Dark theme**: Easy on the eyes
- **Live reload**: Charts and TDEE automatically refresh when Excel file changes
- **Connection indicator**: Green dot shows live connection status
- **Toast notifications**: Brief popup when data updates

### Tooltip Behavior
Hovering on the chart shows:
- **Prediction**: Always shows exactly one prediction value for the hovered date
- **Observation**: Only shown if an actual measurement exists for that exact date
- **Sigma bands**: Never shown in tooltip (visual only)

The tooltip uses `mode: 'x'` with a custom filter to deduplicate items. This is necessary because:
- Observations are sparse (only on measurement days)
- Predictions are dense (daily for 18 months)
- Without deduplication, nearby data points would all appear in tooltip

### API Endpoints

```bash
# List all movements with summary
curl http://localhost:8080/api/movements

# Get full data for a specific movement (default: 2 years history, 12 months prediction)
curl http://localhost:8080/api/movement/squat
curl http://localhost:8080/api/movement/bench
curl http://localhost:8080/api/movement/deadlift
curl http://localhost:8080/api/movement/snatch
curl http://localhost:8080/api/movement/cj
curl http://localhost:8080/api/movement/bodyweight

# With custom date range (history_years: 1-5, prediction_months: 6 or 12)
curl "http://localhost:8080/api/movement/squat?history_years=3&prediction_months=6"

# Get composite indices (IPF GL and Sinclair)
curl http://localhost:8080/api/composites

# With custom date range
curl "http://localhost:8080/api/composites?history_years=1&prediction_months=6"

# Get TDEE (Total Daily Energy Expenditure)
curl http://localhost:8080/api/tdee
# Success: {"average_tdee":2385.0,"tdee":2391.0,"avg_calories":2316.2,"ema_start":79.9,"ema_end":79.6,"weight_change_kg":-0.27,"pairs_used":28}
# Error: {"error":"insufficient_calorie_data","message":"Need 14 calorie entries, found 5"}

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
7. **Visualization**: Chart.js renders predictions with three sigma bands
8. **Live Updates**: File watcher detects changes, reloads data, notifies browsers via WebSocket

## Input Format

Excel file (.xlsx) with columns:
- **Date**: Excel date or ISO format string
- **Weight**: Numeric weight in kg (or calories for calorie entries)
- **Repetitions**: Integer (empty defaults to 1 for lifts, ignored for bodyweight/calorie)
- **Movement**: One of: bodyweight, squat, bench, deadlift, snatch, cj, calorie

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
- Configurable via UI: 1-5 years into the past, 6 or 12 months into the future
- Default: 2 years history, 12 months prediction
- Daily resolution

### Uncertainty Bands
The chart displays three sigma bands (1σ, 2σ, 3σ) using **predictive variance**, not posterior variance:
- **Posterior variance**: Uncertainty in the mean function (shrinks near data points)
- **Predictive variance**: Posterior variance + noise variance (where observations should fall)

The predictive std_dev is computed as: `sqrt(posterior_variance + noise_variance)`

This ensures ~68%/95%/99.7% of observations fall within 1σ/2σ/3σ bands respectively.

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

## TDEE Calculation

TDEE (Total Daily Energy Expenditure) is calculated empirically from calorie intake and weight data.

### Algorithm
1. **Weight EMA Smoothing**: 10-day exponential moving average (α=0.1) for weight trends
2. **Comparison Window**: 28-day period comparing EMA_start and EMA_end
3. **Daily TDEE Formula**: `TDEE = Avg_Calories - (weight_change_kg / 28) × 7700`
4. **TDEE Smoothing**: Calculate TDEE for last 10 days, apply EMA to get smoothed result

### Output Values
- **average_tdee**: Smoothed TDEE (EMA over last 10 daily TDEE calculations) - displayed in UI
- **tdee**: Today's raw TDEE calculation - shown in tooltip

### EMA Boundary Handling
The 10-day EMA window may not have data on every day:
- Window covers `[target_date - 9, target_date]` inclusive
- EMA initializes with the **last** available value (closest to target date)
- Processes **backwards** from last to first, giving maximum weight to most recent data
- If a day has no data, EMA carries forward unchanged
- Minimum 3 data points required in each 10-day window

### Data Requirements
| Requirement | Value | Description |
|-------------|-------|-------------|
| Data span | 48+ days | 28-day window + 10-day EMA lookback × 2 (start + TDEE smoothing) |
| Pair ratio | ≥50% | At least 14 valid calorie-weight pairs in 28 days |
| EMA points | ≥3 | Minimum weights in each 10-day EMA window |

### Constants
| Constant | Value | Description |
|----------|-------|-------------|
| TDEE_WINDOW_DAYS | 28 | Main comparison window |
| EMA_WINDOW_DAYS | 10 | Days for each EMA calculation |
| EMA_ALPHA | 0.1 | EMA smoothing factor |
| KCAL_PER_KG_FAT | 7700 | Energy density of body fat |
| MIN_PAIR_RATIO | 0.5 | Minimum fraction of valid pairs |
| MIN_EMA_DATA_POINTS | 3 | Minimum weights in EMA window |

### Error Types
- `insufficient_calorie_data`: Not enough calorie entries
- `insufficient_weight_data_ema_start`: Not enough weights in start EMA window
- `insufficient_weight_data_ema_end`: Not enough weights in end EMA window
- `insufficient_pairs`: Not enough valid calorie-weight pairs
- `data_span_too_short`: Data doesn't cover required 48-day span

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
