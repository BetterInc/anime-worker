# Privacy & Data Usage

**Important:** This worker connects to an Anime Studio API server. The API server operator can see the data described below.

## What We Don't Do

No IP logging, no location tracking, no analytics, no data selling, no content analysis.

---

## What Your Worker Sends to the API Server

### 1. Hardware Information

**Sent on connection and every 30 seconds:**

```json
{
  "type": "heartbeat",
  "hardware_stats": {
    "cpu": {"cores": 10, "usage_percent": 17},
    "ram": {"total_gb": 125.0, "used_gb": 66.2, "free_gb": 58.8},
    "disk": {"total_gb": 1769.6, "used_gb": 826.9, "free_gb": 942.7},
    "gpus": [
      {"name": "NVIDIA GeForce RTX 3090", "vram_total_mb": 24576, "vram_free_mb": 12507},
      {"name": "NVIDIA GeForce RTX 3090", "vram_total_mb": 24576, "vram_free_mb": 12537}
    ]
  }
}
```

**What the server receives:**
CPU core count and usage, total RAM and disk space (free/used), GPU model names and VRAM amounts.

**Why:** Server needs this to assign jobs that fit your hardware.

---

### 2. Job Progress

**While generating videos:**

```json
{
  "type": "task_progress",
  "progress": 75.0,
  "message": "Diffusion step 15/20"
}
```

**What the server receives:** Current task progress percentage and status messages.

---

### 3. Generated Videos

**What gets uploaded:**
Video files (MP4) and lastframe images (PNG) to the API operator's S3 bucket.

**Important:** The API operator can access all videos your worker generates.

---

### 4. Logs (Optional - Disabled by Default)

**Only if you enable log streaming:**

```json
{
  "type": "log",
  "level": "INFO",
  "message": "Diffusion complete in 1551.8s"
}
```

**What the server receives:** Worker logs, Python inference logs, and GPU/RAM utilization metrics.

**To enable:** Edit `~/.anime-worker/config.toml`
**Retention:** 3 days in database, 30 days in S3, then deleted.

---

## What You Receive From the Server

### Job Assignments

**The server sends you:**

```json
{
  "type": "job_batch_assign",
  "tasks": [{
    "scene": {
      "prompt": "A beautiful digital girl with flowing hair",
      "duration": 3
    }
  }],
  "project": {
    "story_context": "A modern love story...",
    "visual_style": {...}
  }
}
```

**What you receive:** Scene prompts, project context/style, and model configuration.

**Private mode:** Only YOUR jobs. **Public mode:** Jobs from any user.

---

## Security

**Connection:** Encrypted (TLS/WSS), authenticated (API key), outbound only (no open ports).

**Your Control:** Disconnect anytime, delete worker registration, choose private/public mode, set resource limits.

---

## What the API Operator Can See

Hardware specs (CPU/RAM/Disk/GPU), cached models, job progress, and all generated videos. If you enable logs: worker logs and GPU metrics.

### Public vs Private Workers

**Private Worker:**
- ✅ Only processes YOUR jobs (your account)
- ✅ Other users cannot see your worker
- ✅ Your videos stay private

**Public Worker:**
- ⚠️ Processes jobs from ANY user
- ⚠️ You generate videos for other people
- ⚠️ Server operator can see all generated content

**Default:** Private

---

## Trust Model

**If you trust the API operator:** Run the worker. They'll see your hardware specs and videos you generate.

**If you don't trust them:** Don't run the worker, or run your own API server (it's open source).

---

## Open Source

**Audit the code yourself:**
- `src/protocol.rs` - All message formats defined here
- `src/hardware.rs` - How hardware stats are collected
- `src/client.rs` - WebSocket communication
- `src/upload.rs` - File uploads

**No hidden tracking. What you see is what you get.**

---

## Questions?

Open an issue on GitHub.

**Remember:** You're connecting YOUR GPU to SOMEONE ELSE'S server. Only run workers for API servers you trust.
