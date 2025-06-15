use color_eyre::eyre::{eyre, Result, WrapErr};
use reqwest::{multipart, Body};
use std::env;
use std::fs;
use std::process::Command;
use tokio::fs::File;
use tokio_util::codec::{BytesCodec, FramedRead};

async fn run_text_to_view(prompt: &str) -> Result<()> {
    eprintln!("--- Step 1: Running text-to-view ---");
    let status = Command::new("cargo")
        .args([
            "run",
            "-p",
            "text-to-view",
            "--",
            prompt,
        ])
        .status()
        .wrap_err("Failed to execute text-to-view command")?;

    if !status.success() {
        return Err(eyre!("text-to-view process exited with non-zero status"));
    }
    eprintln!("--- Step 1: text-to-view completed successfully ---\n");
    Ok(())
}

async fn run_view_to_3dgs() -> Result<()> {
    eprintln!("--- Step 2: Running view-to-3dgs (peropero) ---");
    
    // For now, we assume the peropero server is already running locally.
    // A more robust implementation would handle starting/stopping the server.
    
    let client = reqwest::Client::new();
    let url = "http://localhost:8888/reconstruction";

    let image_paths: Vec<_> = glob::glob("views/*.jpg")?
        .filter_map(Result::ok)
        .collect();

    if image_paths.is_empty() {
        return Err(eyre!("No images found in the 'views' directory. Did text-to-view run correctly?"));
    }

    let mut form = multipart::Form::new();
    let image_count = image_paths.len();
    for path in image_paths {
        let file_name = path.file_name().unwrap().to_str().unwrap().to_string();
        let file = File::open(&path).await?;
        let stream = FramedRead::new(file, BytesCodec::new());
        let body = Body::wrap_stream(stream);
        let part = multipart::Part::stream(body)
            .file_name(file_name)
            .mime_str("image/jpeg")?;
        form = form.part("images", part);
    }

    eprintln!("Uploading {} images to reconstruction server...", image_count);

    let response = client
        .post(url)
        .multipart(form)
        .send()
        .await
        .wrap_err("Failed to send request to reconstruction server. Is it running at http://localhost:8888?")?;

    if !response.status().is_success() {
        let error_body = response.text().await.unwrap_or_else(|_| "Could not read error body".to_string());
        return Err(eyre!("Reconstruction server returned an error: {}", error_body));
    }

    let ply_data = response.bytes().await?;
    fs::write("output.ply", &ply_data)?;

    eprintln!("--- Step 2: Reconstruction successful! Model saved to output.ply ---\n");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let prompt_parts: Vec<String> = env::args().skip(1).collect();
    if prompt_parts.is_empty() {
        return Err(eyre!("Please provide a prompt as a command-line argument."));
    }
    let user_prompt = prompt_parts.join(" ");

    // Step 1: Generate views from text
    run_text_to_view(&user_prompt).await?;

    // Step 2: Reconstruct 3DGS model from views
    run_view_to_3dgs().await?;

    eprintln!("🎉🎉🎉 Hooray! The entire pipeline is complete. Your 3DGS model is ready in 'output.ply'!");
    eprintln!("--- Step 3: Launching brush viewer ---");

    let brush_executable = "./tools/brush/target/release/brush_app";

    if !std::path::Path::new(brush_executable).exists() {
        eprintln!("'brush_app' not found, compiling it first...");
        let build_status = Command::new("cargo")
            .arg("build")
            .arg("--release")
            .arg("--bin")
            .arg("brush_app")
            .current_dir("./tools/brush")
            .status()
            .wrap_err("Failed to build brush viewer")?;

        if !build_status.success() {
            return Err(eyre!("Failed to compile brush viewer"));
        }
        eprintln!("'brush_app' compiled successfully.");
    }

    let status = Command::new(brush_executable)
        .arg("output.ply")
        .arg("--with-viewer")
        .status()
        .wrap_err("Failed to execute brush viewer")?;

    if !status.success() {
        return Err(eyre!("brush viewer process exited with non-zero status"));
    }

    Ok(())
}
