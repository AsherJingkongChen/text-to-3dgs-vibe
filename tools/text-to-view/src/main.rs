//! To run the program, please execute the following commands in your terminal:
//!
//! 1.  **Set the environment variable for your API key:**
//!     ```bash
//!     export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
//!     ```
//!     (Replace `"YOUR_GEMINI_API_KEY"` with your actual key.)
//! 2.  **Run the example using cargo:**
//!     ```bash
//!     cargo r -p text-to-view -- "a panda meditating"
//!     ```
//!     This will use Gemini to optimize the prompt before sending it to Veo.

use color_eyre::eyre::{eyre, Result, WrapErr};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::time::sleep;
use video_rs::Decoder;
use image::save_buffer;

// --- Data Structures ---

// --- Veo API Structures ---
#[derive(Serialize)]
struct VeoRequest<'a> {
    instances: Vec<Instance<'a>>,
    parameters: Parameters<'a>,
}

#[derive(Serialize)]
struct Instance<'a> {
    prompt: &'a str,
}

#[derive(Serialize)]
struct Parameters<'a> {
    #[serde(rename = "personGeneration")]
    person_generation: &'a str,
    #[serde(rename = "aspectRatio")]
    aspect_ratio: &'a str,
    #[serde(rename = "sampleCount")]
    sample_count: u32,
    #[serde(rename = "durationSeconds")]
    duration_seconds: u32,
}

#[derive(Deserialize, Debug)]
struct LongRunningOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct OperationStatus {
    done: Option<bool>,
    response: Option<VeoResponse>,
    error: Option<OperationError>,
}

#[derive(Deserialize, Debug)]
struct VeoResponse {
    #[serde(rename = "generateVideoResponse")]
    generate_video_response: GenerateVideoResponse,
}

#[derive(Deserialize, Debug)]
struct GenerateVideoResponse {
    #[serde(rename = "generatedSamples")]
    generated_samples: Option<Vec<GeneratedSample>>,
}

#[derive(Deserialize, Debug)]
struct GeneratedSample {
    video: Video,
}

#[derive(Deserialize, Debug)]
struct Video {
    uri: String,
}

#[derive(Deserialize, Debug)]
struct OperationError {
    code: i32,
    message: String,
}

// --- Gemini Prompt Alchemy API Structures ---
#[derive(Serialize)]
struct GeminiRequest<'a> {
    contents: Vec<Content<'a>>,
    #[serde(rename = "generationConfig")]
    generation_config: GenerationConfig<'a>,
}

#[derive(Serialize)]
struct Content<'a> {
    role: &'a str,
    parts: Vec<Part<'a>>,
}

#[derive(Serialize)]
struct Part<'a> {
    text: &'a str,
}

#[derive(Serialize)]
struct GenerationConfig<'a> {
    #[serde(rename = "responseMimeType")]
    response_mime_type: &'a str,
    temperature: f32,
    #[serde(rename = "topP")]
    top_p: f32,
}

#[derive(Deserialize, Debug)]
struct GeminiResponse {
    candidates: Vec<Candidate>,
}

#[derive(Deserialize, Debug)]
struct Candidate {
    content: ContentResponse,
}

#[derive(Deserialize, Debug)]
struct ContentResponse {
    parts: Vec<PartResponse>,
}

#[derive(Deserialize, Debug)]
struct PartResponse {
    text: String,
}

// --- Core Logic Functions ---

const META_PROMPT_TEMPLATE: &str = r#"
You are a master prompt engineer specializing in text-to-video generation.
Your task is to take a user's base prompt and enhance it to be more descriptive, dynamic, and cinematic for the Veo video generation model.
Add details about camera view movement, lighting, tracking, and composition while preserving the core subject.

Your output MUST be only the rewritten prompt text and nothing else.

**User's Base Prompt:**
"{user_prompt_here}"
"#;

/// Rewrites the user's prompt using the Gemini API.
async fn optimize_prompt(client: &reqwest::Client, api_key: &str, user_prompt: &str) -> Result<String> {
    let model_id = "gemini-2.5-flash-preview-05-20";
    let api_method = "streamGenerateContent";
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:{}?key={}",
        model_id, api_method, api_key
    );

    let meta_prompt = META_PROMPT_TEMPLATE.replace("{user_prompt_here}", user_prompt);

    let request_body = GeminiRequest {
        contents: vec![Content {
            role: "user",
            parts: vec![Part { text: &meta_prompt }],
        }],
        generation_config: GenerationConfig {
            response_mime_type: "text/plain",
            temperature: 1.4,
            top_p: 0.9,
        },
    };

    eprintln!("Asking Gemini to optimize prompt...");
    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await?
        .error_for_status()?;

    let response_bytes = response.bytes().await?;
    let response_text = std::str::from_utf8(&response_bytes)?;
    
    // The streaming API returns chunks of JSON in an array. We need to parse them.
    let gemini_responses: Vec<GeminiResponse> = serde_json::from_str(response_text)
        .wrap_err_with(|| format!("Failed to parse Gemini's streaming response: {}", response_text))?;

    let mut optimized_prompt = String::new();
    for res in gemini_responses {
        let Some(candidate) = res.candidates.get(0) else { continue };
        let Some(part) = candidate.content.parts.get(0) else { continue };
        optimized_prompt.push_str(&part.text);
    }

    if optimized_prompt.is_empty() {
        return Err(eyre!("Gemini did not return an optimized prompt. Using original."));
    }
    
    eprintln!("Successfully optimized prompt: '{}'", optimized_prompt);
    Ok(optimized_prompt)
}


/// Submits the video generation request and polls until the video URI is available.
async fn submit_and_poll(client: &reqwest::Client, api_key: &str, prompt: &str) -> Result<String> {
    let model_id = "veo-2.0-generate-001";
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:predictLongRunning?key={}",
        model_id, api_key
    );

    let request_body = VeoRequest {
        instances: vec![Instance { prompt }],
        parameters: Parameters {
            person_generation: "allow_all",
            aspect_ratio: "16:9",
            sample_count: 1,
            duration_seconds: 5,
        },
    };

    eprintln!("Submitting video generation job for the optimized prompt...");
    let initial_response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await?
        .error_for_status()?;

    let operation: LongRunningOperation = initial_response.json().await?;
    let op_name = operation.name;
    eprintln!("Job submitted. Operation name: {}", op_name);

    let status_url = format!("https://generativelanguage.googleapis.com/v1beta/{}?key={}", op_name, api_key);
    loop {
        let status_response = client.get(&status_url).send().await?.error_for_status()?;
        let status: OperationStatus = status_response.json().await?;

        if !status.done.unwrap_or(false) {
            eprintln!("Video not ready yet. Checking again in 6 seconds...");
            sleep(Duration::from_secs(6)).await;
            continue;
        }

        eprintln!("Video generation complete!");
        if let Some(error) = status.error {
            return Err(eyre!("Operation finished with an error: (Code {}) {}", error.code, error.message));
        }

        let response = status.response.ok_or_else(|| eyre!("Operation is done, but no response field was found."))?;
        let samples = response.generate_video_response.generated_samples.ok_or_else(|| eyre!("Response did not contain any generated video samples."))?;
        let sample = samples.first().ok_or_else(|| eyre!("Response contained no video samples."))?;
        
        return Ok(sample.video.uri.clone());
    }
}

/// Downloads a video from a given URL and saves it to a temporary file.
async fn download_video(client: &reqwest::Client, api_key: &str, video_url: &str) -> Result<PathBuf> {
    let download_url = format!("{}&key={}", video_url, api_key);
    let video_bytes = client.get(&download_url).send().await?.bytes().await?;
    
    let mut temp_path = env::temp_dir();
    temp_path.push("video.mp4");

    let mut file = fs::File::create(&temp_path)?;
    file.write_all(&video_bytes)?;
    
    eprintln!("Successfully downloaded video to temporary path: {}", temp_path.display());
    Ok(temp_path)
}

/// Extracts frames from a video file at specified timestamps.
fn extract_frames(video_path: &Path) -> Result<()> {
    video_rs::init().map_err(|e| eyre!(e.to_string()))?;

    let output_dir = Path::new("views");
    fs::remove_dir_all(output_dir).ok();
    fs::create_dir_all(output_dir)?;

    let frame_rate = Decoder::new(video_path)?.frame_rate();
    let timestamps = [0.3, 0.7, 2.3, 2.7, 4.3, 4.7];

    for (i, &time_sec) in timestamps.iter().enumerate() {
        let mut decoder = Decoder::new(video_path)
            .with_context(|| format!("Failed to create decoder for timestamp {}s", time_sec))?;

        let target_frame = (time_sec * frame_rate as f64) as usize;
        let Some(frame_result) = decoder.decode_raw_iter().nth(target_frame) else {
            return Err(eyre!("Failed to extract frame at {}s (frame {}): timestamp may be out of video duration.", time_sec, target_frame));
        };

        let frame = frame_result.with_context(|| format!("Failed to decode frame at position {}", target_frame))?;
        let file_name = format!("{}.jpg", i);
        let output_path = output_dir.join(file_name);

        save_buffer(
            &output_path,
            frame.data(0),
            frame.width(),
            frame.height(),
            image::ColorType::Rgb8,
        ).with_context(|| format!("Failed to save frame to {}", output_path.display()))?;
        
        eprintln!("Saved frame at {}s (frame {}) to {}", time_sec, target_frame, output_path.display());
    }

    Ok(())
}

/// Main entry point for the application.
#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    // --- 1. Setup ---
    let api_key = env::var("GEMINI_API_KEY").wrap_err("GEMINI_API_KEY environment variable not set")?;
    let client = reqwest::Client::new();

    let prompt_parts: Vec<String> = env::args().skip(1).collect();
    if prompt_parts.is_empty() {
        return Err(eyre!("Please provide a prompt as a command-line argument."));
    }
    let user_prompt = prompt_parts.join(" ");

    // --- 2. Prompt Alchemy ---
    let optimized_prompt = match optimize_prompt(&client, &api_key, &user_prompt).await {
        Ok(prompt) => prompt,
        Err(e) => {
            eprintln!("Could not optimize prompt, using original: {}", e);
            user_prompt
        }
    };

    // --- 3. Generate Video ---
    let video_url = submit_and_poll(&client, &api_key, &optimized_prompt).await?;
    eprintln!("Video is available at: {}", video_url);

    // --- 4. Download Video ---
    let video_path = download_video(&client, &api_key, &video_url).await?;

    // --- 5. Extract Frames ---
    eprintln!("Starting frame extraction...");
    extract_frames(&video_path)?;
    eprintln!("Frame extraction successful!");

    // --- 6. Clean up ---
    fs::remove_file(&video_path)?;
    eprintln!("Cleaned up temporary video file: {}", video_path.display());

    Ok(())
}
