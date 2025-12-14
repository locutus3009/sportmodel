//! Web server for serving the training analytics dashboard.
//!
//! Provides a REST API for movement data and composite indices,
//! WebSocket for live updates, and static file serving for the frontend.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    Router,
    extract::{
        Path, Query, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::get,
};
use chrono::{Duration, Local, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast};
use tower_http::services::ServeDir;

use crate::analysis::{MovementAnalysis, analyze_movement};
use crate::domain::{Movement, TrainingData};
use crate::gp::GpConfig;
use crate::tdee::{TdeeError, TdeeResult};

/// Message types for WebSocket broadcast.
#[derive(Clone, Debug)]
pub enum WsMessage {
    /// Data has been reloaded successfully.
    DataUpdated,
    /// An error occurred during reload.
    Error(String),
}

/// Mutable analysis data that can be reloaded.
pub struct AnalysisData {
    #[allow(dead_code)] // May be used for future features
    pub training_data: TrainingData,
    pub analyses: HashMap<Movement, MovementAnalysis>,
    pub ipf_gl: Option<CompositeAnalysis>,
    pub sinclair: Option<CompositeAnalysis>,
    pub tdee: Result<TdeeResult, TdeeError>,
    /// Body fat percentage analysis (GP fitted directly to BF% values)
    pub body_fat: Option<MovementAnalysis>,
    /// Lean body mass analysis (GP fitted directly to LBM values)
    pub lbm: Option<MovementAnalysis>,
    #[allow(dead_code)] // May be used for future features
    pub last_reload: chrono::DateTime<Utc>,
}

/// Shared application state with reloadable data.
pub struct AppState {
    /// The analysis data, protected by RwLock for concurrent reads.
    pub data: RwLock<AnalysisData>,
    /// Path to the Excel file for reloading.
    pub file_path: PathBuf,
    /// Broadcast channel for WebSocket notifications.
    pub ws_broadcast: broadcast::Sender<WsMessage>,
    /// GP configuration for hyperparameters.
    pub gp_config: GpConfig,
}

/// Analysis data for composite indices (IPF GL, Sinclair).
pub struct CompositeAnalysis {
    pub predictions: Vec<CompositePrediction>,
    pub most_reliable_date: Option<NaiveDate>,
    pub current_value: f64,
}

/// A single composite index prediction.
pub struct CompositePrediction {
    pub date: NaiveDate,
    pub value: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

// === JSON Response Types ===

#[derive(Serialize)]
pub struct MovementSummary {
    pub id: String,
    pub name: String,
    pub has_data: bool,
    pub observation_count: usize,
    pub last_observation: Option<String>,
}

#[derive(Serialize)]
pub struct MovementResponse {
    pub movement: String,
    pub observations: Vec<DataPointJson>,
    pub predictions: Vec<PredictionJson>,
    pub last_observation_date: Option<String>,
}

#[derive(Serialize)]
pub struct DataPointJson {
    pub date: String,
    pub value: f64,
}

#[derive(Serialize)]
pub struct PredictionJson {
    pub date: String,
    pub mean: f64,
    pub std_dev: f64,
}

#[derive(Serialize)]
pub struct CompositesResponse {
    pub ipf_gl: Option<CompositeData>,
    pub sinclair: Option<CompositeData>,
}

#[derive(Serialize)]
pub struct CompositeData {
    pub current_value: f64,
    pub predictions: Vec<PredictionJson>,
    pub most_reliable_date: Option<String>,
}

/// Response type for TDEE endpoint.
#[derive(Serialize)]
#[serde(untagged)]
pub enum TdeeResponse {
    /// Successful TDEE calculation.
    Success {
        average_tdee: f64,
        tdee: f64,
        avg_calories: f64,
        ema_start: f64,
        ema_end: f64,
        weight_change_kg: f64,
        pairs_used: usize,
    },
    /// Error response with details.
    Error { error: String, message: String },
}

/// Response type for body composition endpoints (BF%, LBM).
#[derive(Serialize)]
pub struct BodyCompositionResponse {
    pub predictions: Vec<PredictionJson>,
    pub data_points: Vec<DataPointJson>,
}

// === Query Parameters ===

/// Query parameters for movement data endpoint.
#[derive(Deserialize)]
pub struct MovementQuery {
    /// How many years of historical observations to include (1-5, default 1)
    #[serde(default = "default_history_years")]
    pub history_years: u8,
    /// How many months into the future to predict (6 or 12, default 6)
    #[serde(default = "default_prediction_months")]
    pub prediction_months: u8,
}

fn default_history_years() -> u8 {
    2
}

fn default_prediction_months() -> u8 {
    12
}

// === Router Setup ===

/// Creates the application router.
pub fn create_router(state: Arc<AppState>, static_dir: PathBuf) -> Router {
    Router::new()
        .route("/api/movements", get(get_movements))
        .route("/api/movement/{name}", get(get_movement_data))
        .route("/api/composites", get(get_composites))
        .route("/api/tdee", get(get_tdee))
        .route("/api/bodyfat", get(get_bodyfat))
        .route("/api/lbm", get(get_lbm))
        .route("/ws", get(ws_handler))
        .fallback_service(ServeDir::new(static_dir).append_index_html_on_directories(true))
        .with_state(state)
}

// === WebSocket Handler ===

/// WebSocket upgrade handler for live updates.
async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_ws_connection(socket, state))
}

/// Handles an individual WebSocket connection.
async fn handle_ws_connection(mut socket: WebSocket, state: Arc<AppState>) {
    log::info!("WebSocket client connected");

    let mut rx = state.ws_broadcast.subscribe();

    loop {
        tokio::select! {
            // Forward broadcast messages to client
            msg = rx.recv() => {
                match msg {
                    Ok(WsMessage::DataUpdated) => {
                        if socket.send(Message::Text("reload".into())).await.is_err() {
                            break;
                        }
                    }
                    Ok(WsMessage::Error(err)) => {
                        let msg = format!("error:{}", err);
                        if socket.send(Message::Text(msg.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // Missed some messages, send a reload anyway
                        if socket.send(Message::Text("reload".into())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }
            // Handle client messages (ping/pong, close)
            result = socket.recv() => {
                match result {
                    Some(Ok(Message::Ping(data))) => {
                        if socket.send(Message::Pong(data)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    log::info!("WebSocket client disconnected");
}

/// Runs the web server with graceful shutdown support.
pub async fn run_server(
    state: Arc<AppState>,
    port: u16,
    static_dir: PathBuf,
) -> anyhow::Result<()> {
    let app = create_router(state.clone(), static_dir);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    println!("Server running at http://localhost:{}", port);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    #[cfg(feature = "telegram")]
    let telegram_fut = crate::telegram::start_bot(state);
    #[cfg(not(feature = "telegram"))]
    let telegram_fut = std::future::pending::<()>();

    tokio::select! {
    result = axum::serve(listener, app) => {
        result?;
    }
    _ = telegram_fut => {}
    _ = shutdown_signal() => {
        log::info!("Shutting down...");
    }
    }

    log::info!("Server shut down gracefully");
    Ok(())
}

/// Wait for shutdown signal (SIGTERM or SIGINT).
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            log::info!("Received SIGINT, initiating graceful shutdown...");
        }
        _ = terminate => {
            log::info!("Received SIGTERM, initiating graceful shutdown...");
        }
    }
}

// === API Handlers ===

/// GET /api/movements - List all movements with summary.
async fn get_movements(State(state): State<Arc<AppState>>) -> Json<Vec<MovementSummary>> {
    let data = state.data.read().await;

    let summaries: Vec<MovementSummary> = Movement::all()
        .iter()
        .map(|m| {
            let analysis = data.analyses.get(m);
            let data_points = analysis.map(|a| &a.data_points);

            MovementSummary {
                id: movement_to_id(*m),
                name: m.display_name().to_string(),
                has_data: data_points.is_some_and(|dp| !dp.is_empty()),
                observation_count: data_points.map(|dp| dp.len()).unwrap_or(0),
                last_observation: analysis
                    .and_then(|a| a.last_observation_date)
                    .map(|d| d.to_string()),
            }
        })
        .collect();

    Json(summaries)
}

/// GET /api/movement/:name - Full data for one movement.
///
/// Query parameters:
/// - `history_years`: 1-5 (default 1) - years of historical data to show
/// - `prediction_months`: 6 or 12 (default 6) - months into the future to predict
async fn get_movement_data(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<MovementQuery>,
) -> Result<Json<MovementResponse>, StatusCode> {
    let movement = id_to_movement(&name).ok_or(StatusCode::NOT_FOUND)?;

    // Clamp parameters to valid ranges
    let history_years = params.history_years.clamp(1, 5);
    let prediction_months = if params.prediction_months >= 12 {
        12
    } else {
        6
    };

    let today = Local::now().date_naive();
    let history_start = today - Duration::days(i64::from(history_years) * 365);
    let prediction_end = today + Duration::days(i64::from(prediction_months) * 30);

    let data = state.data.read().await;

    // Get the original data points for this movement
    let original_points = data.training_data.get(movement).unwrap_or(&[]);

    // Filter observations to requested history window
    let filtered_points: Vec<_> = original_points
        .iter()
        .filter(|dp| dp.date >= history_start)
        .cloned()
        .collect();

    // Get last observation date from filtered data
    let last_date = filtered_points.last().map(|dp| dp.date);

    // Re-run GP analysis with the filtered data and custom prediction range
    let predictions = if filtered_points.len() >= 2 {
        analyze_movement(
            movement,
            &filtered_points,
            history_start,
            prediction_end,
            &state.gp_config,
        )
        .map(|a| a.predictions)
        .unwrap_or_default()
    } else {
        Vec::new()
    };

    // Convert to JSON format
    let observations: Vec<DataPointJson> = filtered_points
        .iter()
        .map(|dp| DataPointJson {
            date: dp.date.to_string(),
            value: dp.value,
        })
        .collect();

    let preds: Vec<PredictionJson> = predictions
        .iter()
        .map(|p| PredictionJson {
            date: p.date.to_string(),
            mean: p.mean,
            std_dev: p.std_dev,
        })
        .collect();

    Ok(Json(MovementResponse {
        movement: movement.display_name().to_string(),
        observations,
        predictions: preds,
        last_observation_date: last_date.map(|d| d.to_string()),
    }))
}

/// GET /api/composites - IPF GL and Sinclair data.
///
/// Query parameters:
/// - `history_years`: 1-5 (default 2) - years of historical data to show
/// - `prediction_months`: 6 or 12 (default 12) - months into the future to predict
async fn get_composites(
    State(state): State<Arc<AppState>>,
    Query(params): Query<MovementQuery>,
) -> Json<CompositesResponse> {
    // Clamp parameters to valid ranges
    let history_years = params.history_years.clamp(1, 5);
    let prediction_months = if params.prediction_months >= 12 {
        12
    } else {
        6
    };

    let today = Local::now().date_naive();
    let history_start = today - Duration::days(i64::from(history_years) * 365);
    let prediction_end = today + Duration::days(i64::from(prediction_months) * 30);

    let data = state.data.read().await;

    let ipf_gl = data.ipf_gl.as_ref().map(|c| CompositeData {
        current_value: c.current_value,
        predictions: c
            .predictions
            .iter()
            .filter(|p| p.date >= history_start && p.date <= prediction_end)
            .map(|p| PredictionJson {
                date: p.date.to_string(),
                mean: p.value,
                // Derive std_dev from CI bounds (assuming symmetric uncertainty)
                std_dev: (p.ci_upper - p.ci_lower) / 2.0,
            })
            .collect(),
        most_reliable_date: c.most_reliable_date.map(|d| d.to_string()),
    });

    let sinclair = data.sinclair.as_ref().map(|c| CompositeData {
        current_value: c.current_value,
        predictions: c
            .predictions
            .iter()
            .filter(|p| p.date >= history_start && p.date <= prediction_end)
            .map(|p| PredictionJson {
                date: p.date.to_string(),
                mean: p.value,
                // Derive std_dev from CI bounds (assuming symmetric uncertainty)
                std_dev: (p.ci_upper - p.ci_lower) / 2.0,
            })
            .collect(),
        most_reliable_date: c.most_reliable_date.map(|d| d.to_string()),
    });

    Json(CompositesResponse { ipf_gl, sinclair })
}

/// GET /api/tdee - TDEE (Total Daily Energy Expenditure) calculation.
///
/// Returns the calculated TDEE from calorie intake and weight data,
/// or an error with details if insufficient data is available.
async fn get_tdee(State(state): State<Arc<AppState>>) -> Json<TdeeResponse> {
    let data = state.data.read().await;

    let response = match &data.tdee {
        Ok(result) => TdeeResponse::Success {
            average_tdee: result.average_tdee,
            tdee: result.tdee,
            avg_calories: result.avg_calories,
            ema_start: result.ema_start,
            ema_end: result.ema_end,
            weight_change_kg: result.weight_change_kg,
            pairs_used: result.pairs_used,
        },
        Err(err) => {
            let (error_type, message) = match err {
                crate::tdee::TdeeError::InsufficientCalorieData {
                    available,
                    required,
                } => (
                    "insufficient_calorie_data",
                    format!("Need {} calorie entries, found {}", required, available),
                ),
                crate::tdee::TdeeError::InsufficientWeightDataForEmaStart {
                    available,
                    required,
                } => (
                    "insufficient_weight_data_ema_start",
                    format!(
                        "Need {} weights in EMA start window, found {}",
                        required, available
                    ),
                ),
                crate::tdee::TdeeError::InsufficientWeightDataForEmaEnd {
                    available,
                    required,
                } => (
                    "insufficient_weight_data_ema_end",
                    format!(
                        "Need {} weights in EMA end window, found {}",
                        required, available
                    ),
                ),
                crate::tdee::TdeeError::InsufficientPairs {
                    available,
                    required,
                } => (
                    "insufficient_pairs",
                    format!(
                        "Need {} calorie-weight pairs, found {}",
                        required, available
                    ),
                ),
                crate::tdee::TdeeError::DataSpanTooShort {
                    available_days,
                    required_days,
                } => (
                    "data_span_too_short",
                    format!(
                        "Data span {} days is less than required {} days",
                        available_days, required_days
                    ),
                ),
            };
            TdeeResponse::Error {
                error: error_type.to_string(),
                message,
            }
        }
    };

    Json(response)
}

/// GET /api/bodyfat - Body fat percentage predictions.
///
/// Query parameters:
/// - `history_years`: 1-5 (default 2) - years of historical data to show
/// - `prediction_months`: 6 or 12 (default 12) - months into the future to predict
async fn get_bodyfat(
    State(state): State<Arc<AppState>>,
    Query(params): Query<MovementQuery>,
) -> Result<Json<BodyCompositionResponse>, StatusCode> {
    let history_years = params.history_years.clamp(1, 5);
    let prediction_months = if params.prediction_months >= 12 {
        12
    } else {
        6
    };

    let today = Local::now().date_naive();
    let history_start = today - Duration::days(i64::from(history_years) * 365);
    let prediction_end = today + Duration::days(i64::from(prediction_months) * 30);

    let data = state.data.read().await;

    let Some(body_fat) = &data.body_fat else {
        return Err(StatusCode::NOT_FOUND);
    };

    // Filter predictions to the requested date range
    let predictions: Vec<PredictionJson> = body_fat
        .predictions
        .iter()
        .filter(|p| p.date >= history_start && p.date <= prediction_end)
        .map(|p| PredictionJson {
            date: p.date.to_string(),
            mean: p.mean,
            std_dev: p.std_dev,
        })
        .collect();

    // Data points are the actual computed BF% values from matched measurements
    let data_points: Vec<DataPointJson> = body_fat
        .data_points
        .iter()
        .filter(|dp| dp.date >= history_start && dp.date <= prediction_end)
        .map(|dp| DataPointJson {
            date: dp.date.to_string(),
            value: dp.value,
        })
        .collect();

    Ok(Json(BodyCompositionResponse {
        predictions,
        data_points,
    }))
}

/// GET /api/lbm - Lean body mass predictions.
///
/// Query parameters:
/// - `history_years`: 1-5 (default 2) - years of historical data to show
/// - `prediction_months`: 6 or 12 (default 12) - months into the future to predict
async fn get_lbm(
    State(state): State<Arc<AppState>>,
    Query(params): Query<MovementQuery>,
) -> Result<Json<BodyCompositionResponse>, StatusCode> {
    let history_years = params.history_years.clamp(1, 5);
    let prediction_months = if params.prediction_months >= 12 {
        12
    } else {
        6
    };

    let today = Local::now().date_naive();
    let history_start = today - Duration::days(i64::from(history_years) * 365);
    let prediction_end = today + Duration::days(i64::from(prediction_months) * 30);

    let data = state.data.read().await;

    let Some(lbm) = &data.lbm else {
        return Err(StatusCode::NOT_FOUND);
    };

    // Filter predictions to the requested date range
    let predictions: Vec<PredictionJson> = lbm
        .predictions
        .iter()
        .filter(|p| p.date >= history_start && p.date <= prediction_end)
        .map(|p| PredictionJson {
            date: p.date.to_string(),
            mean: p.mean,
            std_dev: p.std_dev,
        })
        .collect();

    // Data points are the actual computed LBM values from matched measurements
    let data_points: Vec<DataPointJson> = lbm
        .data_points
        .iter()
        .filter(|dp| dp.date >= history_start && dp.date <= prediction_end)
        .map(|dp| DataPointJson {
            date: dp.date.to_string(),
            value: dp.value,
        })
        .collect();

    Ok(Json(BodyCompositionResponse {
        predictions,
        data_points,
    }))
}

// === Helper Functions ===

fn movement_to_id(movement: Movement) -> String {
    match movement {
        Movement::Bodyweight => "bodyweight".to_string(),
        Movement::Squat => "squat".to_string(),
        Movement::Bench => "bench".to_string(),
        Movement::Deadlift => "deadlift".to_string(),
        Movement::Snatch => "snatch".to_string(),
        Movement::CleanAndJerk => "cj".to_string(),
        Movement::Calorie => "calorie".to_string(),
        Movement::Neck => "neck".to_string(),
        Movement::Waist => "waist".to_string(),
    }
}

fn id_to_movement(id: &str) -> Option<Movement> {
    match id.to_lowercase().as_str() {
        "bodyweight" => Some(Movement::Bodyweight),
        "squat" => Some(Movement::Squat),
        "bench" => Some(Movement::Bench),
        "deadlift" => Some(Movement::Deadlift),
        "snatch" => Some(Movement::Snatch),
        "cj" | "cleanandjerk" => Some(Movement::CleanAndJerk),
        "calorie" => Some(Movement::Calorie),
        "neck" => Some(Movement::Neck),
        "waist" => Some(Movement::Waist),
        _ => None,
    }
}
