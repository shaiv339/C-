use axum::{
    extract::State,
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc};
use tokio::task;
use uuid::Uuid;

// Hypothetical llama_cpp crate
use llama_cpp::Llama;

#[derive(Debug, Deserialize)]
struct InferRequest {
    prompt: String,
    max_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
struct InferResponse {
    id: String,
    output: String,
}

#[derive(Clone)]
struct AppState {
    llama: Arc<Llama>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let llama = Arc::new(Llama::new("models/llama.bin"));

    let app_state = AppState { llama };

    let app = Router::new()
        .route("/infer", post(handle_inference))
        .with_state(app_state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!(" Server running at http://{}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn handle_inference(
    State(state): State<AppState>,
    Json(req): Json<InferRequest>,
) -> Json<InferResponse> {
    let llama = state.llama.clone();
    let prompt = req.prompt.clone();
    let max_tokens = req.max_tokens.unwrap_or(50);

    let output = task::spawn_blocking(move || {
        llama.infer(&prompt, max_tokens)
    })
    .await
    .unwrap_or_else(|_| "Inference failed.".to_string());

    Json(InferResponse {
        id: Uuid::new_v4().to_string(),
        output,
    })
}