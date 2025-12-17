# sportmodel

Personal strength training analytics tool for Olympic weightlifting and powerlifting. Reads training data from Excel files, applies Gaussian Process regression for trend analysis, and serves a web interface with graphs.

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3). See LICENSE.txt for details.

## Recent Changes (since 3398a81)

### GPLv3 License (b6db3f2)
- Added LICENSE.txt file with full GPLv3 text
- Added license field to Cargo.toml

### Enhanced TDEE Command (94100e6, c6763cd, e911e8a)
- The `/tdee` Telegram command now displays comprehensive 28-day analysis:
  - Average TDEE (smoothed over 10 days)
  - Today's raw TDEE
  - Average calorie intake
  - Weight change in kg

### Update Existing Rows (442660e)
- Telegram bot now intelligently updates existing spreadsheet entries
- When recording data for today's date, searches for existing entry with matching date and movement type
- Updates the existing row instead of creating duplicates
- Confirmation messages include " [updated]" suffix when updating
- Prevents accumulation of duplicate daily entries from corrections or multiple submissions

### Fix Excel Phantom Cell Creation (a6381ad)
- **Critical Bug Fix**: Telegram bot `append_excel()` was creating phantom cells during row search
- **Root Cause**: Using `get_cell_mut()` to read cells during search loop created XML entries for non-existent rows
- **Symptoms**:
  - LibreOffice warning: "maximum number of rows per sheet was exceeded"
  - calamine failed to load files immediately after bot updates
  - Files worked after LibreOffice re-save (which cleaned phantom cells)
- **Solution**: Replace `get_cell_mut()` with read-only `get_cell()` during search phase
  - `get_cell()` returns `Option<&Cell>` - reads existing cells without creating new ones
  - `get_cell_mut()` still used for actual writes (after finding/creating target row)
- **Added File Sync**: Explicit `sync_all()` and 50ms delay after write to ensure file watcher sees complete data
- **Result**: Clean Excel files with no phantom cells, immediate compatibility with calamine

### Fix Authorization Bypass Vulnerability (48d9613)
- **Security Fix**: Telegram bot authorization could be bypassed if `msg.from` was `None`
- **Root Cause**: Using `if let Some(ref user) = msg.from` pattern allowed execution to continue when user info was missing
- **Impact**: Commands could execute without authorization for forwarded messages from channels or other edge cases
- **Solution**: Use `map_or(false, ...)` pattern to explicitly deny access when user information is missing
  - Authorization only succeeds when BOTH conditions are met:
    - `msg.from` is present (Some)
    - User ID is in the whitelist
- **Result**: All code paths now require valid user information and whitelist membership

## Project Structure

```
sportmodel/
├── Cargo.toml
├── sportmodel.service  # Systemd user service unit file
├── src/
│   ├── main.rs              # Entry point: loads data, runs GP analysis, starts web server
│   ├── domain.rs            # Domain types (Movement, Observation, DataPoint, TrainingData)
│   ├── error.rs             # Custom error types (ParseError, FormulaError)
│   ├── excel.rs             # Excel file parsing with calamine
│   ├── formulas.rs          # Strength calculation formulas (e1RM, IPF GL, Sinclair)
│   ├── gp.rs                # Gaussian Process regression implementation
│   ├── analysis.rs          # Analysis orchestration and higher-level functions
│   ├── server.rs            # Web server (axum) with REST API, WebSocket, and static serving
│   ├── body_composition.rs  # Body fat % and lean body mass calculations
│   ├── tdee.rs              # TDEE calculation from calorie and weight data
│   ├── telegram.rs          # Telegram bot for data entry (optional feature)
│   └── watcher.rs           # File watching with debouncing for live reload
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
- `Movement` enum: Bodyweight, Squat, Bench, Deadlift, Snatch, CleanAndJerk, Calorie, Neck, Waist
- `Observation`: Raw data from Excel (date, weight, reps, movement)
- `DataPoint`: Processed data ready for GP regression (date, value)
- `TrainingData`: Container organizing data by movement, sorted by date

### body_composition.rs
- `calculate_body_fat_pct()`: US Navy formula for body fat percentage
- `calculate_lbm()`: Lean body mass from bodyweight and BF%
- `calculate_raw_body_fat_points()`: Computes exact BF% from matched neck/waist measurements
- `calculate_raw_lbm_points()`: Computes exact LBM from matched bodyweight/neck/waist
- `analyze_body_fat()`: Fits GP directly to computed BF% values
- `analyze_lbm()`: Fits GP directly to computed LBM values

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
- `GpConfig`: Per-movement length scale configuration (strength=90d, body_comp=60d, energy=60d)
- `GpHyperparameters`: Length scale, signal variance, noise variance
- `GpPrediction`: Mean prediction with standard deviation (used for sigma bands)
- `GpModel`: Fitted GP model with `fit()`, `predict()`, `predict_range()`, `log_marginal_likelihood()`
  - Caches Cholesky decomposition from `fit()` for efficient variance computation
  - Uses batch matrix operations for variance computation (single solve instead of per-point)
- `GpError`: Error types for insufficient data, singular matrix, invalid params
- `optimize_noise_with_metadata()`: Optimizes noise variance via log marginal likelihood (fixed length scale)
- Uses squared exponential (RBF) kernel with Cholesky decomposition for stability

### analysis.rs
- `MovementAnalysis`: Predictions and data points for a single movement
- `analyze_training_data()`: Fits GP models for all movements in parallel (rayon)
- `PredictionSummary`: Summary stats with CI and trend
- `find_most_reliable_date_*`: Staleness calculation for composite indices

### server.rs
- `AnalysisData`: Mutable data container (training data, analyses, composites, TDEE, body composition)
- `AppState`: Shared state with `RwLock<AnalysisData>`, file path, WebSocket broadcast, and `GpConfig`
- `WsMessage`: WebSocket message types (`DataUpdated`, `Error`)
- `create_router()`: Configures axum routes, WebSocket endpoint, and static file serving
- `run_server()`: Starts the HTTP server
- `ws_handler()`: WebSocket upgrade handler for live updates
- REST API handlers for movements, composite indices, TDEE, and body composition

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

### telegram.rs (optional, requires `telegram` feature)
- `start_bot()`: Starts the Telegram bot dispatcher
- `Command` enum: Bot commands (Help, Tdee, Bodyweight, Squat, Bench, Deadlift, Snatch, Cj, Calories, NeckAndWaist)
- `append_excel()`: Adds or updates rows in the Excel file using umya-spreadsheet
  - Searches for existing row with matching date and movement type using `get_cell()` (read-only)
  - Uses `get_cell()` during search to avoid creating phantom XML entries for non-existent cells
  - Uses `get_cell_mut()` only for actual writes to target row
  - If existing row found, updates it instead of appending
  - Returns " [updated]" suffix when updating, empty string when appending
  - Explicitly syncs file to disk with `sync_all()` and 50ms delay for file watcher compatibility
- `date_to_excel_serial()`: Converts NaiveDate to Excel serial date format
- Uses teloxide for Telegram Bot API
- Uses umya-spreadsheet for Excel writes (calamine is read-only)

## Build and Run

```bash
# Build
cargo build

# Build with Telegram bot support
cargo build --features telegram

# Run tests
cargo test

# Run with test data (starts web server)
cargo run -- test_data.xlsx 8080

# Run with logging
RUST_LOG=warn cargo run -- test_data.xlsx 8080

# Run with Telegram bot (requires TELOXIDE_TOKEN env var)
TELOXIDE_TOKEN=your_bot_token cargo run --features telegram -- test_data.xlsx 8080

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
| `TELOXIDE_TOKEN` | No | - | Telegram bot token (only if built with `telegram` feature) |

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
- **10 tabs**: Squat, Bench, Deadlift, Snatch, C&J, IPF GL, Sinclair, Body Fat %, LBM, Bodyweight
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

# Get body composition metrics (requires neck and waist data)
curl http://localhost:8080/api/bodyfat
curl http://localhost:8080/api/lbm
# Success: {"predictions":[{"date":"2024-01-15","mean":18.5,"std_dev":1.2},...], "data_points":[{"date":"2024-01-10","value":18.2},...]}
# Error: 404 if insufficient neck/waist/bodyweight data

# With custom date range
curl "http://localhost:8080/api/bodyfat?history_years=3&prediction_months=6"

# WebSocket endpoint for live updates
# Connect to ws://localhost:8080/ws
# Receives "reload" message when data changes
```

## Live Reload

The server watches the Excel file for modifications and automatically reloads data when changes are detected.

### How It Works
1. File watcher monitors the Excel file using the notify crate
2. Changes are debounced (2 second window) to handle rapid successive writes
3. Reload operations are serialized and coalesced (see below)
4. On change, data is reloaded with retry logic (3 attempts, 500ms delay)
5. WebSocket broadcasts "reload" message to all connected browsers
6. Frontend clears cache and refreshes the current chart

### Debouncing
Syncthing and Excel may trigger multiple file system events for a single save:
- File created (temp file)
- File modified (multiple writes)
- File renamed (atomic replace)

The debouncer collapses these into a single reload after activity settles.

### Reload Serialization and Coalescing
Even with debouncing, multiple events can slip through. The reload system handles this:
- **Serialization**: Only one reload runs at a time (protected by mutex)
- **Coalescing**: If events arrive during a reload, they're coalesced into a single pending reload
- **Result**: At most 2 reloads per save operation (one during save, one after to catch final state)

This prevents concurrent file reads, duplicate WebSocket broadcasts, and wasteful repeated reloads.

### Error Handling
- If reload fails, retries up to 3 times with 500ms delay between attempts
- Transient failures (file mid-save) are logged at debug level, not warn
- On permanent failure, WebSocket receives "error:message" and toast shows error
- Old data remains displayed until successful reload

### Connection States
- **Live** (green): WebSocket connected, receiving updates
- **Offline** (red): WebSocket disconnected, no live updates
- **Reconnecting** (orange): Attempting to reconnect with exponential backoff

## Data Flow

1. **Input**: Excel file with columns Date, Weight, Repetitions, Movement
   - Manual editing via spreadsheet application
   - Telegram bot commands (appends rows directly)
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
- **Weight**: Numeric weight in kg (or calories for calorie entries, or cm for neck/waist)
- **Repetitions**: Integer (empty defaults to 1 for lifts, ignored for bodyweight/calorie/neck/waist)
- **Movement**: One of: bodyweight, squat, bench, deadlift, snatch, cj, calorie, neck, waist

## GP Regression

### Hyperparameter Configuration

GP hyperparameters are configured via `GpConfig` (created once in main, passed through call chain):

| Parameter | Strength Movements | Body Composition | Energy |
|-----------|-------------------|------------------|--------|
| Length scale | 90 days | 60 days | 60 days |
| Movements | squat, bench, deadlift, snatch, cj | bodyweight, neck, waist, BF%, LBM | calorie |

**Length scale rationale**:
- Strength movements use 90 days (changes over months, weekly variation is noise)
- Body composition uses 60 days (faster response to diet/training changes)

### Hyperparameter Optimization

**Fixed parameters** (domain knowledge):
- **Length scale**: Per-movement category (see table above)
- **Signal variance**: Estimated from data variance (min floor of 1.0)

**Optimized parameter** (via log marginal likelihood):
- **Noise variance**: Grid search over noise ratios [0.01, 0.02, ..., 0.5]

The log marginal likelihood balances data fit against model complexity:
```
log p(y|X,θ) = -½ yᵀK⁻¹y - ½ log|K| - n/2 log(2π)
```
- First term: Data fit (prefers models that explain observations)
- Second term: Complexity penalty (prefers simpler covariance structures)

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

### Performance Optimizations

The GP implementation uses several optimizations for handling large datasets (1000+ observations):

1. **Cached Cholesky Decomposition**: The O(n³) Cholesky factorization is computed once during `fit()` and stored in `GpModel`. This avoids redundant factorization during variance computation.

2. **Batch Variance Computation**: Instead of solving `Lv = k*` for each test point individually (2000+ times), we solve `LV = K*ᵀ` once as a batch matrix operation, then extract column norms.

3. **Parallel Movement Analysis**: Movement GP fitting is parallelized via rayon. Each movement's analysis is independent and runs on a separate thread.

**Complexity Analysis**:
| Operation | Complexity | Notes |
|-----------|------------|-------|
| `GpModel::fit()` | O(n³) | Cholesky decomposition dominates |
| `GpModel::predict()` means | O(n × m) | Cross-kernel matrix multiplication |
| `GpModel::predict()` variance | O(n² × m) | Batch triangular solve |

Where n = training observations, m = prediction points.

**Typical Performance** (1405 bodyweight + 6 other movements, 2186 prediction days):
- Startup time: ~2 seconds (release build)

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

## Body Composition

Body Fat Percentage (BF%) and Lean Body Mass (LBM) are calculated by fitting GP regression **directly** to computed values.

### Approach: Direct GP Interpolation

The correct approach for body composition is:
1. **Compute exact values** from matched measurements (neck+waist on the same day)
2. **Fit GP regression** directly to these computed BF%/LBM values
3. **Interpolate BF%/LBM directly** — the GP learns actual BF%/LBM dynamics

This avoids the incorrect approach of interpolating inputs (neck/waist separately) then computing through the nonlinear formula, which produces artificial dynamics due to:
- GP smoothing on inputs creating artificial correlations
- Nonlinear transformation (log10) amplifying smoothing artifacts
- GP hyperparameters tuned to neck/waist dynamics, not BF% dynamics

### US Navy Formula (Men)

BF% is calculated using the US Navy method:

```
BF% = 495 / (1.0324 - 0.19077 × log10(waist - neck) + 0.15456 × log10(height)) - 450
```

**Measurements**:
- Waist: measured at navel level (in cm)
- Neck: measured at narrowest point below Adam's apple (in cm)
- Height: constant 180 cm (hardcoded)

**Constraint**: Waist must be greater than neck (waist - neck > 0)

### Lean Body Mass

LBM is calculated from bodyweight and BF%:

```
LBM = bodyweight × (1 - BF% / 100)
```

### Data Point Markers

Chart markers (green dots) show dates where ALL required actual measurements exist:
- **Body Fat %**: Requires both neck AND waist measured on the same day
- **LBM**: Requires bodyweight, neck, AND waist measured on the same day

The values displayed are the exact computed values (not interpolated).

### Data Requirements
- Minimum 2 matched neck+waist measurements for BF% GP regression
- Minimum 2 matched bodyweight+neck+waist measurements for LBM GP regression

### Uncertainty Bands

Uncertainty bands come directly from the GP model fitted to BF%/LBM values. This represents actual uncertainty in the BF%/LBM estimate based on:
- Distance from actual measurements
- Variance in the observed BF%/LBM data
- GP hyperparameters estimated from BF%/LBM dynamics

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

## Telegram Bot

The optional Telegram bot allows data entry via chat commands. When enabled, the bot runs concurrently with the web server.

### Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram
2. Copy the bot token
3. Build with the `telegram` feature and set the `TELOXIDE_TOKEN` environment variable

```bash
# Run with Telegram bot
TELOXIDE_TOKEN=123456:ABC-DEF... cargo run --features telegram -- data.xlsx 8080
```

### Authorization

The bot uses a user whitelist for access control:

- **Discovery mode**: Leave `TELEGRAM_ALLOWED_USERS` unset or empty. All requests are denied and user IDs are logged.
- **Whitelist mode**: Set `TELEGRAM_ALLOWED_USERS="123456789,987654321"` with comma-separated user IDs. Only listed users can use the bot.
- **Finding your ID**: Run in discovery mode, send a message to the bot, check server logs for `user_id=...` entries.

Unauthorized users receive "⚠️ Access denied. Contact bot administrator."

### Available Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/help` | - | Display available commands |
| `/tdee` | - | Display 28-day TDEE analysis (average TDEE, today's TDEE, avg intake, weight change) |
| `/bodyweight` | `<weight>` | Record body weight (kg) |
| `/squat` | `<weight> <reps>` | Record back squat |
| `/bench` | `<weight> <reps>` | Record bench press |
| `/deadlift` | `<weight> <reps>` | Record deadlift |
| `/snatch` | `<weight> <reps>` | Record snatch |
| `/cj` | `<weight> <reps>` | Record clean & jerk |
| `/calories` | `<calories>` | Record calorie intake |
| `/neckandwaist` | `<neck> <waist>` | Record neck and waist measurements (cm) |

### How It Works

1. Commands add or update rows in the Excel file using umya-spreadsheet
2. Date is set to the current local date automatically
3. If an entry already exists for today's date and movement type, it is updated instead of creating a duplicate
4. Confirmation messages include " [updated]" suffix when updating existing entries
5. The file watcher detects changes and triggers a reload
6. WebSocket broadcasts update to connected browsers

### Error Handling

The bot provides helpful error messages for:
- Missing arguments (shows expected usage)
- Too many arguments
- Invalid format (e.g., non-numeric values)
- Unknown commands (shows help text)

### Feature Flag

The Telegram bot is behind a feature flag to avoid pulling in extra dependencies when not needed:

```toml
[features]
telegram = ["dep:teloxide", "dep:umya-spreadsheet"]
```

Dependencies added by the `telegram` feature:
- `teloxide`: Telegram Bot API framework
- `umya-spreadsheet`: Excel file writing (calamine is read-only)
