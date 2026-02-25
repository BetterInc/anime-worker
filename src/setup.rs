//! Interactive setup wizard for anime-worker.
//!
//! Guides the user through creating a `config.toml` with sensible defaults
//! derived from detected hardware, then optionally bootstraps the Python
//! virtual environment.
//!
//! The wizard is split into small, individually-testable functions so that
//! the core config-building logic can be covered by unit tests without
//! spawning a real TTY.

use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

use anyhow::Context;
use console::style;
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Password};

use crate::config::{self, WorkerConfig, WorkerConstraints};
use crate::hardware;

// ---------------------------------------------------------------------------
// Public entry-point
// ---------------------------------------------------------------------------

/// Run the full interactive setup wizard.
///
/// `config_path` is the target file that will be written (e.g. `./config.toml`).
/// `scripts_dir`  is the directory that contains `setup_env.py`.
pub fn run(config_path: &Path, scripts_dir: &Path) -> anyhow::Result<()> {
    let theme = ColorfulTheme::default();

    print_banner();

    // --- Step 1: existing config detection ---------------------------------
    let skip_setup = if config_path.exists() {
        println!(
            "\n{} Config found at {}",
            style("»").cyan(),
            style(config_path.display()).yellow()
        );
        Confirm::with_theme(&theme)
            .with_prompt("Use existing config?")
            .default(true)
            .interact()
            .context("Failed to read confirmation")?
    } else {
        false
    };

    if skip_setup {
        println!("{} Using existing config — nothing changed.", style("✓").green());
        offer_python_setup(&theme, scripts_dir)?;
        print_done();
        return Ok(());
    }

    // --- Step 2: gather settings interactively -----------------------------
    let cfg = gather_config(&theme)?;

    // --- Step 3: persist ---------------------------------------------------
    cfg.save(config_path)
        .with_context(|| format!("Failed to write config to {}", config_path.display()))?;

    println!("\n{} Config saved to {}", style("✓").green(), style(config_path.display()).yellow());

    // --- Step 4: optional Python setup -------------------------------------
    offer_python_setup(&theme, scripts_dir)?;

    print_done();
    Ok(())
}

// ---------------------------------------------------------------------------
// Config gathering — separated so it can be unit-tested
// ---------------------------------------------------------------------------

/// Prompt the user for every configuration value and return a completed
/// [`WorkerConfig`].  All I/O goes through `dialoguer`; no file operations
/// are performed here.
pub fn gather_config(theme: &ColorfulTheme) -> anyhow::Result<WorkerConfig> {
    println!("\n{}", style("── Worker connection ──────────────────────────────").dim());

    // Server URL
    let server_url: String = Input::with_theme(theme)
        .with_prompt("API server URL")
        .default(DEFAULT_SERVER_URL.to_string())
        .interact_text()
        .context("Failed to read server URL")?;

    // Worker name (default: hostname)
    let default_name = hostname();
    let worker_name: String = Input::with_theme(theme)
        .with_prompt("Worker name")
        .default(default_name)
        .interact_text()
        .context("Failed to read worker name")?;

    // API key — use Password so it is not echoed in the terminal
    let api_key: String = Password::with_theme(theme)
        .with_prompt("API key (from worker registration)")
        .allow_empty_password(false)
        .interact()
        .context("Failed to read API key")?;

    // Worker ID — derived from API key prefix or entered separately.
    // The server encodes the worker UUID in the key as "aw_<uuid>_<secret>".
    // We parse it out; if the format does not match we ask explicitly.
    let worker_id = derive_worker_id_from_key(&api_key, theme)?;

    // --- Resource limits ---------------------------------------------------
    println!("\n{}", style("── Resource limits (optional) ─────────────────────").dim());

    let configure_limits = Confirm::with_theme(theme)
        .with_prompt("Configure resource limits?")
        .default(false)
        .interact()
        .context("Failed to read confirmation")?;

    let constraints = if configure_limits {
        gather_constraints(theme)?
    } else {
        WorkerConstraints::default()
    };

    // --- Paths / Python ----------------------------------------------------
    let python_path = match config::discover_python() {
        Some(p) => {
            println!(
                "{} Auto-detected Python at '{}'",
                style("✓").green(),
                style(&p).yellow()
            );
            p
        }
        None => {
            println!(
                "{} Python 3.10+ not found on PATH. You can set it manually in config.toml.",
                style("!").yellow()
            );
            "python3".to_string()
        }
    };

    let python_scripts_dir = config::default_python_scripts_dir_pub();

    Ok(WorkerConfig {
        server_url,
        worker_id,
        api_key,
        worker_name,
        models_dir: config::config_dir().join("models"),
        python_path,
        python_scripts_dir,
        heartbeat_interval_secs: 30,
        constraints,
    })
}

/// Prompt the user for resource constraints.
pub fn gather_constraints(theme: &ColorfulTheme) -> anyhow::Result<WorkerConstraints> {
    let available_cores = hardware::cpu_cores();
    let available_ram   = hardware::total_ram_gb();
    let available_disk  = hardware::available_disk_space_gb(Path::new("."));

    println!(
        "  Detected: {} CPU cores, {:.1} GB RAM, {:.1} GB disk free",
        available_cores, available_ram, available_disk
    );

    // CPU
    let cpu_input: String = Input::with_theme(theme)
        .with_prompt(format!("CPU cores to allocate (0 = all, max {})", available_cores))
        .default("0".to_string())
        .validate_with(|s: &String| validate_non_negative_usize(s))
        .interact_text()
        .context("Failed to read CPU cores")?;
    let cpu_limit = parse_optional_usize(&cpu_input);

    // RAM
    let ram_input: String = Input::with_theme(theme)
        .with_prompt(format!("RAM limit in GB (0 = all, {:.0} GB available)", available_ram))
        .default("0".to_string())
        .validate_with(|s: &String| validate_non_negative_f64(s))
        .interact_text()
        .context("Failed to read RAM limit")?;
    let ram_limit_gb = parse_optional_f64(&ram_input);

    // Disk
    let disk_input: String = Input::with_theme(theme)
        .with_prompt(format!("Disk space limit in GB (0 = all, {:.0} GB available)", available_disk))
        .default("0".to_string())
        .validate_with(|s: &String| validate_non_negative_f64(s))
        .interact_text()
        .context("Failed to read disk limit")?;
    let disk_limit_gb = parse_optional_f64(&disk_input);

    Ok(WorkerConstraints {
        cpu_limit,
        ram_limit_gb,
        disk_limit_gb,
        ..WorkerConstraints::default()
    })
}

// ---------------------------------------------------------------------------
// Python setup offer
// ---------------------------------------------------------------------------

fn offer_python_setup(theme: &ColorfulTheme, scripts_dir: &Path) -> anyhow::Result<()> {
    println!();
    let run_python = Confirm::with_theme(theme)
        .with_prompt("Run Python environment setup now? (installs PyTorch, diffusers, …)")
        .default(false)
        .interact()
        .context("Failed to read confirmation")?;

    if run_python {
        run_python_setup(scripts_dir)?;
    }
    Ok(())
}

/// Execute the Python setup script.  This is the only place that shells out
/// during setup; kept separate so tests can skip it.
pub fn run_python_setup(scripts_dir: &Path) -> anyhow::Result<()> {
    let setup_script = scripts_dir.join("setup_env.py");

    let bootstrap = config::discover_python().ok_or_else(|| {
        anyhow::anyhow!(
            "No Python 3.10+ found on PATH.\n\
             Install Python first:\n  \
             - Linux: sudo apt install python3 python3-venv\n  \
             - Windows: https://www.python.org/downloads/"
        )
    })?;

    if !setup_script.exists() {
        anyhow::bail!(
            "setup_env.py not found at {}.\n\
             Make sure the python/ directory is next to the binary.",
            setup_script.display()
        );
    }

    println!("  Running {} {}…", bootstrap, setup_script.display());

    let status = std::process::Command::new(&bootstrap)
        .arg(&setup_script)
        .status()
        .with_context(|| format!("Failed to launch '{}'", bootstrap))?;

    if !status.success() {
        anyhow::bail!(
            "Python setup exited with code {:?}. \
             Run `anime-worker setup-python` manually for details.",
            status.code()
        );
    }

    println!("{} Python environment ready.", style("✓").green());
    Ok(())
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

fn print_banner() {
    println!();
    println!("{}", style("╔═══════════════════════════════════════╗").cyan());
    println!("{}", style("║       anime-worker  •  Setup Wizard    ║").cyan());
    println!("{}", style("╚═══════════════════════════════════════╝").cyan());
    println!();
    println!("This wizard creates a {} in the current directory.", style("config.toml").yellow());
    println!("You will need your API key from the worker registration page.");
}

fn print_done() {
    println!();
    println!("{}", style("✅  Setup complete!").green().bold());
    println!("Run {} to start processing jobs.", style("./anime-worker run").yellow().bold());
}

// ---------------------------------------------------------------------------
// Small pure helpers — easy to unit-test
// ---------------------------------------------------------------------------

/// Default API server URL shown in the prompt.
pub const DEFAULT_SERVER_URL: &str =
    "https://animestudiocontaineryaw5gajb-anime-api.functions.fnc.nl-ams.scw.cloud";

/// Return the machine hostname, falling back to `"my-worker"`.
pub fn hostname() -> String {
    hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "my-worker".to_string())
}

/// Try to extract a worker UUID from an API key of the form `aw_<uuid>_…`.
///
/// If the key does not contain a UUID segment, ask the user to enter the
/// worker ID explicitly.
pub fn derive_worker_id_from_key(
    api_key: &str,
    theme: &ColorfulTheme,
) -> anyhow::Result<String> {
    if let Some(id) = extract_uuid_from_key(api_key) {
        return Ok(id);
    }

    // Key does not contain an embedded UUID — ask explicitly.
    let worker_id: String = Input::with_theme(theme)
        .with_prompt("Worker ID (UUID from registration page)")
        .validate_with(|s: &String| {
            if s.trim().is_empty() {
                Err("Worker ID must not be empty")
            } else {
                Ok(())
            }
        })
        .interact_text()
        .context("Failed to read worker ID")?;

    Ok(worker_id)
}

/// Parse a worker UUID out of a key shaped like `aw_<uuid>_<secret>`.
///
/// Returns `None` if the key does not match the expected format.
pub fn extract_uuid_from_key(api_key: &str) -> Option<String> {
    // Expected format: aw_<uuid>_<random>
    // uuid is 36 chars: 8-4-4-4-12
    let stripped = api_key.strip_prefix("aw_")?;
    // A UUID is exactly 36 characters (including hyphens)
    if stripped.len() >= 36 {
        let candidate = &stripped[..36];
        if is_valid_uuid(candidate) {
            return Some(candidate.to_string());
        }
    }
    None
}

/// Cheap UUID format check: 8-4-4-4-12 hex groups separated by hyphens.
pub fn is_valid_uuid(s: &str) -> bool {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 5 {
        return false;
    }
    let expected_lens = [8, 4, 4, 4, 12];
    parts
        .iter()
        .zip(expected_lens.iter())
        .all(|(part, &len)| part.len() == len && part.chars().all(|c| c.is_ascii_hexdigit()))
}

/// Validate that a string represents a non-negative integer.
pub fn validate_non_negative_usize(s: &String) -> Result<(), String> {
    match s.trim().parse::<usize>() {
        Ok(_) => Ok(()),
        Err(_) => Err(format!("'{}' is not a valid non-negative integer", s)),
    }
}

/// Validate that a string represents a non-negative float.
pub fn validate_non_negative_f64(s: &String) -> Result<(), String> {
    match s.trim().parse::<f64>() {
        Ok(v) if v >= 0.0 => Ok(()),
        Ok(_) => Err(format!("'{}' must be >= 0", s)),
        Err(_) => Err(format!("'{}' is not a valid number", s)),
    }
}

/// Parse `"0"` as `None` (meaning "no limit"), anything else as `Some(n)`.
pub fn parse_optional_usize(s: &str) -> Option<usize> {
    match s.trim().parse::<usize>() {
        Ok(0) => None,
        Ok(n) => Some(n),
        Err(_) => None,
    }
}

/// Parse `"0"` or `"0.0"` as `None` (meaning "no limit"), else `Some(v)`.
pub fn parse_optional_f64(s: &str) -> Option<f64> {
    match s.trim().parse::<f64>() {
        Ok(v) if v <= 0.0 => None,
        Ok(v) => Some(v),
        Err(_) => None,
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- UUID helpers -------------------------------------------------------

    #[test]
    fn valid_uuid_accepted() {
        assert!(is_valid_uuid("550e8400-e29b-41d4-a716-446655440000"));
    }

    #[test]
    fn uuid_wrong_group_lengths_rejected() {
        assert!(!is_valid_uuid("550e840-e29b-41d4-a716-446655440000")); // first group too short
        assert!(!is_valid_uuid("550e84000-e29b-41d4-a716-446655440000")); // first group too long
    }

    #[test]
    fn uuid_non_hex_chars_rejected() {
        assert!(!is_valid_uuid("zzzzzzzz-e29b-41d4-a716-446655440000"));
    }

    #[test]
    fn uuid_wrong_number_of_groups_rejected() {
        assert!(!is_valid_uuid("550e8400-e29b-41d4"));
    }

    // --- Key parsing --------------------------------------------------------

    #[test]
    fn extracts_uuid_from_valid_key() {
        let uuid = "42676a86-383b-4ae4-ab29-a4130a0ae988";
        let key  = format!("aw_{}_CHM552Vi4XKTVDbLCPejoirkbFgtsSCWfgUZYgDUNy4", uuid);
        assert_eq!(extract_uuid_from_key(&key), Some(uuid.to_string()));
    }

    #[test]
    fn returns_none_for_key_without_uuid() {
        // Key that starts with "aw_" but no UUID segment
        assert_eq!(extract_uuid_from_key("aw_shortkey"), None);
    }

    #[test]
    fn returns_none_for_key_without_prefix() {
        assert_eq!(extract_uuid_from_key("sk-somethingelse"), None);
    }

    #[test]
    fn returns_none_for_empty_key() {
        assert_eq!(extract_uuid_from_key(""), None);
    }

    // --- Number validators --------------------------------------------------

    #[test]
    fn non_negative_usize_accepts_zero() {
        assert!(validate_non_negative_usize(&"0".to_string()).is_ok());
    }

    #[test]
    fn non_negative_usize_accepts_positive() {
        assert!(validate_non_negative_usize(&"8".to_string()).is_ok());
    }

    #[test]
    fn non_negative_usize_rejects_float() {
        assert!(validate_non_negative_usize(&"3.5".to_string()).is_err());
    }

    #[test]
    fn non_negative_usize_rejects_text() {
        assert!(validate_non_negative_usize(&"all".to_string()).is_err());
    }

    #[test]
    fn non_negative_f64_accepts_zero() {
        assert!(validate_non_negative_f64(&"0".to_string()).is_ok());
        assert!(validate_non_negative_f64(&"0.0".to_string()).is_ok());
    }

    #[test]
    fn non_negative_f64_accepts_positive_float() {
        assert!(validate_non_negative_f64(&"16.5".to_string()).is_ok());
    }

    #[test]
    fn non_negative_f64_rejects_negative() {
        assert!(validate_non_negative_f64(&"-1".to_string()).is_err());
    }

    #[test]
    fn non_negative_f64_rejects_text() {
        assert!(validate_non_negative_f64(&"lots".to_string()).is_err());
    }

    // --- Optional parsers ---------------------------------------------------

    #[test]
    fn parse_zero_usize_gives_none() {
        assert_eq!(parse_optional_usize("0"), None);
    }

    #[test]
    fn parse_positive_usize_gives_some() {
        assert_eq!(parse_optional_usize("4"), Some(4));
    }

    #[test]
    fn parse_zero_f64_gives_none() {
        assert_eq!(parse_optional_f64("0"), None);
        assert_eq!(parse_optional_f64("0.0"), None);
    }

    #[test]
    fn parse_positive_f64_gives_some() {
        assert_eq!(parse_optional_f64("32.0"), Some(32.0));
    }

    #[test]
    fn parse_negative_f64_gives_none() {
        // Negative values are treated as "no limit" (defensive)
        assert_eq!(parse_optional_f64("-5"), None);
    }

    // --- Config round-trip --------------------------------------------------

    #[test]
    fn config_built_with_no_limits_has_none_constraints() {
        let cfg = WorkerConfig {
            server_url: "https://example.com".to_string(),
            worker_id: "42676a86-383b-4ae4-ab29-a4130a0ae988".to_string(),
            api_key: "aw_42676a86-383b-4ae4-ab29-a4130a0ae988_secret".to_string(),
            worker_name: "test-worker".to_string(),
            models_dir: PathBuf::from("/tmp/models"),
            python_path: "python3".to_string(),
            python_scripts_dir: PathBuf::from("/tmp/python"),
            heartbeat_interval_secs: 30,
            constraints: WorkerConstraints {
                cpu_limit: parse_optional_usize("0"),
                ram_limit_gb: parse_optional_f64("0"),
                disk_limit_gb: parse_optional_f64("0"),
                ..WorkerConstraints::default()
            },
        };

        assert!(cfg.constraints.cpu_limit.is_none());
        assert!(cfg.constraints.ram_limit_gb.is_none());
        assert!(cfg.constraints.disk_limit_gb.is_none());
    }

    #[test]
    fn config_built_with_explicit_limits_stores_values() {
        let constraints = WorkerConstraints {
            cpu_limit: parse_optional_usize("4"),
            ram_limit_gb: parse_optional_f64("16"),
            disk_limit_gb: parse_optional_f64("200"),
            ..WorkerConstraints::default()
        };

        assert_eq!(constraints.cpu_limit, Some(4));
        assert_eq!(constraints.ram_limit_gb, Some(16.0));
        assert_eq!(constraints.disk_limit_gb, Some(200.0));
    }

    #[test]
    fn config_save_and_reload_preserves_constraints() {
        use crate::config::WorkerConfig;

        let tmp = std::env::temp_dir()
            .join(format!("anime-worker-setup-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("config.toml");

        let cfg = WorkerConfig {
            server_url: DEFAULT_SERVER_URL.to_string(),
            worker_id: "42676a86-383b-4ae4-ab29-a4130a0ae988".to_string(),
            api_key: "aw_42676a86-383b-4ae4-ab29-a4130a0ae988_secret".to_string(),
            worker_name: "unit-test-worker".to_string(),
            models_dir: tmp.join("models"),
            python_path: "python3".to_string(),
            python_scripts_dir: tmp.join("python"),
            heartbeat_interval_secs: 30,
            constraints: WorkerConstraints {
                cpu_limit: Some(8),
                ram_limit_gb: Some(32.0),
                disk_limit_gb: Some(500.0),
                ..WorkerConstraints::default()
            },
        };

        cfg.save(&path).expect("save should succeed");
        let loaded = WorkerConfig::load(&path).expect("load should succeed");

        assert_eq!(loaded.server_url, DEFAULT_SERVER_URL);
        assert_eq!(loaded.worker_name, "unit-test-worker");
        assert_eq!(loaded.constraints.cpu_limit, Some(8));
        assert_eq!(loaded.constraints.ram_limit_gb, Some(32.0));
        assert_eq!(loaded.constraints.disk_limit_gb, Some(500.0));

        std::fs::remove_dir_all(&tmp).ok();
    }

    // --- hostname -----------------------------------------------------------

    #[test]
    fn hostname_returns_non_empty_string() {
        let h = hostname();
        assert!(!h.is_empty());
    }
}
