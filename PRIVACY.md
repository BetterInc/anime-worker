# Privacy & Data Usage

This worker connects to the Anime Studio API to process video generation jobs.

## What We Don't Do

No IP logging, no location tracking, no analytics, no data selling, no content analysis.

---

## What Your Worker Sends

### 1. Hardware Stats (every 30 seconds)

```json
{
  "cpu": {"cores": 10, "usage_percent": 17},
  "ram": {"total_gb": 125.0, "used_gb": 66.2, "free_gb": 58.8},
  "disk": {"total_gb": 1769.6, "used_gb": 826.9, "free_gb": 942.7},
  "gpus": [
    {"name": "NVIDIA GeForce RTX 3090", "vram_total_mb": 24576, "vram_free_mb": 12507}
  ]
}
```

**Why:** Assign jobs that fit your hardware.

### 2. Job Progress

```json
{
  "type": "task_progress",
  "progress": 75.0,
  "message": "Diffusion step 15/20"
}
```

**Why:** Show progress in dashboard.

### 3. Generated Videos

Uploads video files (MP4) and images (PNG) to S3 storage.

### 4. Logs (Optional - Off by Default)

Worker logs and GPU metrics. **Not sent unless you enable it.**

**To enable:** Add to `~/.anime-worker/config.toml`:
```toml
enable_log_streaming = true
enable_metrics_collection = true
```

**Retention:** 3 days, then deleted (or 30 days in archive).

---

## What the Service Can See

Hardware specs (CPU/RAM/Disk/GPU), cached models, job progress, and generated videos. If you enable logs: worker logs and GPU metrics.

---

## Private vs Public Workers

**Private Worker (Default):**
- Only processes YOUR jobs (your account)
- Other users cannot see your worker
- Your videos stay private

**Public Worker:**
- Processes jobs from ANY user
- You generate videos for other people
- Contributes GPU power to help the community

---

## What You Receive

Scene prompts, project settings, and model configuration for video generation.

**Private mode:** Only your jobs.
**Public mode:** Jobs from any user.

---

## Security

Encrypted connection (WSS/HTTPS), API key authentication, outbound only (no open ports on your machine).

**Your control:** Disconnect anytime, delete worker, choose private/public, set limits.

---

## Open Source

Audit the code on GitHub: `src/protocol.rs`, `src/hardware.rs`, `src/client.rs`

No hidden tracking.
