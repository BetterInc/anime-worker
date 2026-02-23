/// Test to verify hf-hub can download .gitattributes without errors

#[test]
fn test_hf_hub_download_gitattributes() {
    use hf_hub::api::sync::Api;

    println!("Testing HF Hub API with Wan-AI/Wan2.2-TI2V-5B-Diffusers...");

    // Create API client
    let api = Api::new().expect("Failed to create HF API client");
    let repo = api.model("Wan-AI/Wan2.2-TI2V-5B-Diffusers".to_string());

    // Get repo info
    let info = repo.info().expect("Failed to get repo info");
    println!("Found {} files in repo", info.siblings.len());

    // Find .gitattributes
    let gitattributes = info
        .siblings
        .iter()
        .find(|f| f.rfilename == ".gitattributes");

    if let Some(file) = gitattributes {
        println!("Found .gitattributes, attempting download...");

        // This is where the bug occurred in 0.3.2
        match repo.get(&file.rfilename) {
            Ok(path) => {
                println!("✓ SUCCESS: Downloaded to {:?}", path);
                assert!(path.exists(), "Downloaded file should exist");
            }
            Err(e) => {
                panic!("✗ FAILED: {}", e);
            }
        }
    } else {
        println!("Warning: .gitattributes not found in repo");
    }
}
