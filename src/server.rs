//! Web server for serving the training analytics dashboard.
//!
//! Provides a REST API for movement data and composite indices,
//! plus static file serving for the frontend.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    Router,
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::get,
};
use chrono::NaiveDate;
use serde::Serialize;
use tower_http::services::ServeDir;

use crate::analysis::MovementAnalysis;
use crate::domain::{Movement, TrainingData};

/// Shared application state.
pub struct AppState {
    #[allow(dead_code)] // May be used for future features
    pub training_data: TrainingData,
    pub analyses: HashMap<Movement, MovementAnalysis>,
    pub ipf_gl: Option<CompositeAnalysis>,
    pub sinclair: Option<CompositeAnalysis>,
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
    pub ci_lower: f64,
    pub ci_upper: f64,
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

// === Router Setup ===

/// Creates the application router.
pub fn create_router(state: Arc<AppState>, static_dir: PathBuf) -> Router {
    Router::new()
        .route("/api/movements", get(get_movements))
        .route("/api/movement/{name}", get(get_movement_data))
        .route("/api/composites", get(get_composites))
        .fallback_service(ServeDir::new(static_dir).append_index_html_on_directories(true))
        .with_state(state)
}

/// Runs the web server.
pub async fn run_server(
    state: Arc<AppState>,
    port: u16,
    static_dir: PathBuf,
) -> anyhow::Result<()> {
    let app = create_router(state, static_dir);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    println!("Server running at http://localhost:{}", port);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// === API Handlers ===

/// GET /api/movements - List all movements with summary.
async fn get_movements(State(state): State<Arc<AppState>>) -> Json<Vec<MovementSummary>> {
    let summaries: Vec<MovementSummary> = Movement::all()
        .iter()
        .map(|m| {
            let analysis = state.analyses.get(m);
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
async fn get_movement_data(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<MovementResponse>, StatusCode> {
    let movement = id_to_movement(&name).ok_or(StatusCode::NOT_FOUND)?;
    let analysis = state.analyses.get(&movement);

    let (observations, predictions, last_date) = if let Some(a) = analysis {
        let obs: Vec<DataPointJson> = a
            .data_points
            .iter()
            .map(|dp| DataPointJson {
                date: dp.date.to_string(),
                value: dp.value,
            })
            .collect();

        let preds: Vec<PredictionJson> = a
            .predictions
            .iter()
            .map(|p| PredictionJson {
                date: p.date.to_string(),
                mean: p.mean,
                ci_lower: p.ci_lower(),
                ci_upper: p.ci_upper(),
            })
            .collect();

        let last = a.last_observation_date.map(|d| d.to_string());

        (obs, preds, last)
    } else {
        (Vec::new(), Vec::new(), None)
    };

    Ok(Json(MovementResponse {
        movement: movement.display_name().to_string(),
        observations,
        predictions,
        last_observation_date: last_date,
    }))
}

/// GET /api/composites - IPF GL and Sinclair data.
async fn get_composites(State(state): State<Arc<AppState>>) -> Json<CompositesResponse> {
    let ipf_gl = state.ipf_gl.as_ref().map(|c| CompositeData {
        current_value: c.current_value,
        predictions: c
            .predictions
            .iter()
            .map(|p| PredictionJson {
                date: p.date.to_string(),
                mean: p.value,
                ci_lower: p.ci_lower,
                ci_upper: p.ci_upper,
            })
            .collect(),
        most_reliable_date: c.most_reliable_date.map(|d| d.to_string()),
    });

    let sinclair = state.sinclair.as_ref().map(|c| CompositeData {
        current_value: c.current_value,
        predictions: c
            .predictions
            .iter()
            .map(|p| PredictionJson {
                date: p.date.to_string(),
                mean: p.value,
                ci_lower: p.ci_lower,
                ci_upper: p.ci_upper,
            })
            .collect(),
        most_reliable_date: c.most_reliable_date.map(|d| d.to_string()),
    });

    Json(CompositesResponse { ipf_gl, sinclair })
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
        _ => None,
    }
}
