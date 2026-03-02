# Anime Worker - Privacy & Data Usage

## Overview

The Anime Worker is a distributed GPU worker that connects to the Anime Studio API to process video generation jobs. This document describes **exactly what data is sent and received**, and **why it's needed**.

---

## Data Flow Summary

```
┌─────────────┐                    ┌─────────────┐
│   Worker    │ ←──── Jobs ──────  │     API     │
│  (Your GPU) │ ────── Results ──→ │   Server    │
└─────────────┘                    └─────────────┘
     ↑                                    ↑
     │                                    │
  Your machine                      Cloud (Scaleway)
  Private                           You control this!
```

---

## 1. Data Sent FROM Worker TO Server

### A. On Connection (Hello Message)

**What:**
```json
{
  "type": "hello",
  "worker_id": "uuid-here",
  "api_key": "your-api-key-hash",
  "name": "Your Worker Name",
  "platform": "linux",
  "hardware_stats": {
    "cpu": {
      "cores": 10,
      "usage_percent": 17
    },
    "ram": {
      "total_gb": 125.0,
      "used_gb": 45.2,
      "free_gb": 79.8
    },
    "disk": {
      "total_gb": 1769.6,
      "used_gb": 826.3,
      "free_gb": 943.3
    },
    "gpus": [
      {
        "name": "NVIDIA GeForce RTX 3090",
        "vram_total_mb": 24576,
        "vram_free_mb": 12507
      }
    ]
  },
  "models_cached": ["wan22_t2v_14b"],
  "constraints": {
    "max_model_size_gb": 100
  }
}
```

**Why needed:**
- `worker_id`: Identify your worker in the system
- `api_key`: Authentication (proves you own this worker)
- `hardware_stats`: Server needs to know your capabilities to assign appropriate jobs
  - **CPU/RAM**: For job scheduling (don't assign 120GB RAM job to 32GB worker)
  - **Disk**: Prevent downloading models that won't fit
  - **GPUs**: Assign jobs to workers with sufficient VRAM
- `models_cached`: Avoid re-downloading models you already have
- `platform`: For compatibility (Linux vs Windows paths, commands)

**What is NOT sent:**
- ❌ Your IP address (WebSocket connection, not logged)
- ❌ File paths on your machine
- ❌ Personal data
- ❌ Other processes running on your machine

---

### B. Every 30 Seconds (Heartbeat)

**What:**
```json
{
  "type": "heartbeat",
  "worker_id": "uuid",
  "hardware_stats": {
    // Same as above - updated stats
  },
  "models_cached": ["wan22_t2v_14b", "wan22_ti2v_5b"]
}
```

**Why needed:**
- **Detect if worker is still alive** - If no heartbeat for 60s, mark offline
- **Track resource usage** - Is RAM/disk filling up? (prevent OOM)
- **Update model cache** - Did you download a new model?
- **Capacity planning** - How busy are your workers?

**Frequency:** Every 30 seconds
**Privacy:** Only hardware stats, no personal data

---

### C. During Model Download (Model Progress)

**What:**
```json
{
  "type": "model_progress",
  "task_id": "uuid",
  "model_id": "wan22_t2v_14b",
  "downloaded_gb": 8.5,
  "total_gb": 28.0
}
```

**Why needed:**
- **Show progress in dashboard** - User sees "Downloading 8.5/28.0 GB"
- **Estimate completion time** - Calculate ETA
- **Detect stuck downloads** - If progress stops for 10 min, retry

**Frequency:** When changed by >100MB
**Privacy:** Only download progress, no file content

---

### D. During Generation (Task Progress)

**What:**
```json
{
  "type": "task_progress",
  "task_id": "uuid",
  "progress": 75.0,
  "message": "Diffusion step 15/20",
  "phase": "generating"
}
```

**Why needed:**
- **Show real-time progress** - User sees job isn't stuck
- **Estimate completion** - Calculate ETA for remaining scenes
- **Detect failures** - If progress stops, worker might have crashed

**Frequency:** Every inference step (~1-2 minutes)
**Privacy:** Only progress info, no video content

---

### E. On Completion (Task Complete)

**What:**
```json
{
  "type": "task_complete",
  "task_id": "uuid",
  "result_filename": "scene_001_preview.mp4",
  "metadata": {
    "duration": 3.0,
    "frames": 73,
    "resolution": "832x480"
  }
}
```

**Why needed:**
- **Mark job as done** - Update database, show success to user
- **Track which file** - Link result to correct scene
- **Analytics** - How long did generation take? (for future estimates)

**Privacy:** Only metadata (duration, resolution), **NOT the video content itself**

**Important:** The actual video file is uploaded separately via HTTPS (see section 2 below)

---

### F. Logs (Optional - New Feature)

**What:**
```json
{
  "type": "log",
  "level": "INFO",
  "message": "Diffusion complete in 1551.8s",
  "source": "python",
  "timestamp": "2026-03-02T13:23:54Z",
  "metadata": {
    "gpu_id": 0,
    "sm_util": 98,
    "mem_util": 95
  }
}
```

**Why needed:**
- **Debugging** - When jobs fail, logs show exactly what went wrong
- **Performance monitoring** - Is your GPU being used efficiently?
- **Support** - You can share logs instead of describing the problem

**Privacy:**
- ✅ Logs are tied to YOUR user account (only you can see them)
- ✅ Deleted after 3 days (or archived to S3 for 30 days)
- ❌ No personal data in logs (API keys/paths are sanitized)

**Optional:** This feature can be disabled in worker config if preferred

---

## 2. Data Sent VIA Worker (File Uploads)

### Video Files

**What:**
- Generated video files (MP4)
- Lastframe images (PNG)

**Where:**
- Uploaded to **S3 bucket** (Scaleway Object Storage)
- **NOT stored on API server** (just passes through)

**Why needed:**
- **Deliver results to user** - You generated the video, user needs to download it
- **Last-frame continuity** - Next scene uses previous scene's last frame

**Privacy:**
- ✅ Files stored in **YOUR S3 bucket** (you control access)
- ✅ Only accessible by **authenticated users** (your account)
- ✅ Pre-signed URLs expire after download
- ❌ API server never stores video content (just metadata)

**Flow:**
```
Worker → Upload to S3 → User downloads from S3
         ↑
     API server just coordinates, doesn't store
```

---

## 3. Data Received BY Worker FROM Server

### A. Job Assignments

**What:**
```json
{
  "type": "job_batch_assign",
  "job_id": "uuid",
  "tasks": [
    {
      "task_id": "uuid",
      "scene": {
        "id": 1,
        "prompt": "A beautiful digital girl with flowing hair",
        "duration": 3
      }
    }
  ],
  "project": {
    "story_context": "A modern love story...",
    "visual_style": {...}
  },
  "model_config": {
    "model_path": "/app/models/wan22_t2v_14b",
    "num_inference_steps": 20
  }
}
```

**Why needed:**
- **Tell worker what to generate** - Scene prompts, settings
- **Provide configuration** - Which model, how many steps, resolution
- **Context** - Story context for coherent generation

**Privacy:**
- ✅ Only **your jobs** are sent to **your workers**
- ✅ Private workers only receive jobs from their owner
- ✅ Public workers can receive jobs from any user (opt-in setting)

---

### B. Task Cancellation

**What:**
```json
{
  "type": "task_cancel",
  "task_id": "uuid",
  "reason": "Cancelled by user"
}
```

**Why needed:**
- **Stop wasted work** - User cancelled job, free up GPU immediately
- **Resource management** - Don't generate videos nobody wants

---

### C. Model Download Instructions

**What:**
```json
{
  "model_id": "wan22_t2v_14b",
  "hf_repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  "size_gb": 28
}
```

**Why needed:**
- **Download correct model** - From HuggingFace
- **Verify integrity** - Check size matches expected
- **Resume failed downloads** - Know where to resume from

**Privacy:**
- ✅ Downloads from **HuggingFace CDN** (public repos)
- ✅ Stored **locally on your worker** (not on server)
- ❌ Server never touches model weights

---

## 4. What Data is Stored in the Database

### Worker Registration

**Stored:**
- Worker ID (UUID)
- Worker name (you choose)
- API key hash (bcrypt, not reversible)
- Owner user ID (your account)
- Hardware stats (latest snapshot)
- Status (online/offline/busy)
- Last heartbeat timestamp

**NOT stored:**
- ❌ Your IP address
- ❌ Your actual API key (only hash)
- ❌ File paths on your machine
- ❌ Personal information

### Job History

**Stored (per job):**
- Job ID, type (preview/render)
- Status, progress
- Start/end timestamps
- Model used
- Scene count

**NOT stored:**
- ❌ Generated video content (goes to S3)
- ❌ Your prompts after job completes (deleted with project)
- ❌ IP addresses

### Worker Logs (Optional)

**Stored (if enabled):**
- Log level, message, timestamp
- GPU/RAM metrics
- Job/task IDs (for linking)

**Retention:**
- 3 days in database
- 30 days in S3 archive
- Then permanently deleted

**NOT stored:**
- ❌ Sensitive data (API keys sanitized)
- ❌ File system paths
- ❌ Personal information

---

## 5. Network Security

### WebSocket Connection

**Protocol:** WSS (WebSocket Secure) over TLS 1.3
**Authentication:** API key verified on every connection
**Direction:** Outbound only (worker connects TO server, not reverse)

**Why outbound?**
- ✅ No open ports on your machine
- ✅ Works behind NAT/firewall
- ✅ You control the connection (can disconnect anytime)

### File Uploads

**Protocol:** HTTPS with multipart/form-data
**Authentication:** API key header
**Encryption:** TLS 1.3 in transit

**Retry logic:**
- 3 attempts with exponential backoff
- 5 minute timeout per attempt
- Prevents data loss from network hiccups

---

## 6. What You Control

### Worker Configuration

**You control:**
- ✅ Which server to connect to
- ✅ Worker name and ID
- ✅ Resource constraints (max model size, etc.)
- ✅ When to start/stop the worker
- ✅ Log streaming (enable/disable)
- ✅ Which models to cache

**File:** `~/.anime-worker/config.toml`

### Data Deletion

**You can:**
- ✅ Delete your worker (removes all logs, history)
- ✅ Disconnect anytime (worker goes offline, jobs requeued)
- ✅ Clear model cache (free up disk space)
- ✅ Request data export (all your logs/history)

---

## 7. Privacy Best Practices

### What We Do

✅ **Minimal data collection** - Only what's needed for functionality
✅ **Short retention** - Logs deleted after 3 days (33 days with archive)
✅ **No tracking** - No analytics, no telemetry beyond job metrics
✅ **Encryption** - TLS for all network traffic
✅ **Authentication** - API keys, not passwords
✅ **Isolation** - Your workers only see your jobs (if private)

### What We Don't Do

❌ **No IP logging** - We don't store or log your IP address
❌ **No browser fingerprinting** - No tracking cookies or scripts
❌ **No third-party analytics** - No Google Analytics, Mixpanel, etc.
❌ **No data selling** - Your data is never sold or shared
❌ **No content analysis** - We don't analyze generated videos

---

## 8. Data Minimization

### What Could Be Sent But Isn't

We deliberately **don't** collect:
- ❌ Worker machine hostname
- ❌ OS version beyond "linux"/"windows"
- ❌ CPU model name
- ❌ GPU serial numbers
- ❌ Network information (MAC address, local IP)
- ❌ User login name
- ❌ File system paths
- ❌ Running processes
- ❌ Installed software

**We only collect the minimum needed for job assignment.**

---

## 9. Comparison to Alternatives

### Commercial Services (RunPod, VastAI, etc.)

**They collect:**
- ❌ Your credit card info
- ❌ Detailed telemetry
- ❌ Usage analytics
- ❌ They control the servers (you don't)

**Anime Worker:**
- ✅ Self-hosted (you control the API server)
- ✅ Minimal data (only hardware stats)
- ✅ No payment info needed (you own the hardware)
- ✅ Open source (audit the code yourself)

---

## 10. Self-Hosting Privacy

### You Control Everything

Since you self-host the API:
- ✅ **Your database** - All data stored in YOUR Postgres
- ✅ **Your S3 bucket** - All files in YOUR object storage
- ✅ **Your logs** - Stored in YOUR infrastructure
- ✅ **No third parties** - Data never leaves your control

**This is as private as it gets!**

---

## 11. Worker Code Transparency

### Open Source

The worker is **fully open source**:
- ✅ Inspect exactly what data is sent (src/protocol.rs)
- ✅ See how hardware stats are collected (src/hardware.rs)
- ✅ Audit network calls (src/client.rs)
- ✅ No telemetry or hidden data collection

**You can fork and modify** - Remove any data collection you don't want!

---

## 12. Sensitive Data Handling

### API Keys

**Storage:** `~/.anime-worker/config.toml`
**Permissions:** `chmod 600` (only you can read)
**Transmission:** Only sent during authentication
**Server storage:** Bcrypt hash (not reversible)

### Generated Content

**On worker:** Stored temporarily in `~/.anime-worker/output/`
**After upload:** **Deleted locally** (unless you configure retention)
**On server:** **NOT stored** - goes directly to S3
**On S3:** Encrypted at rest, accessible only by you

---

## 13. GDPR Compliance

### Your Rights

✅ **Right to access** - Download all your data anytime
✅ **Right to deletion** - Delete worker = delete all data
✅ **Right to portability** - Export logs/history as JSON
✅ **Right to be forgotten** - Delete account = purge all data

### Data Retention

| Data Type | Retention | Why |
|-----------|-----------|-----|
| Worker info | Until deleted | Needed while worker exists |
| Hardware stats | Latest only | For capacity planning |
| Job history | 30 days | For debugging recent issues |
| Logs | 3+30 days | For debugging, then deleted |
| Videos | Until deleted | User's content |

---

## 14. Security Measures

### Worker Security

✅ **No inbound connections** - Worker connects OUT, not IN
✅ **Firewall friendly** - Works behind NAT
✅ **API key authentication** - Not username/password
✅ **TLS 1.3** - Modern encryption
✅ **Code signing** - GitHub releases are signed

### Data in Transit

✅ **WebSocket over TLS** (WSS)
✅ **HTTPS for uploads**
✅ **Certificate validation**
✅ **No plaintext transmission**

### Data at Rest

✅ **API keys hashed** (bcrypt, not reversible)
✅ **Database encrypted** (Scaleway RDS encryption)
✅ **S3 encrypted** (AES-256)
✅ **Logs archived** (encrypted in S3)

---

## 15. Opting Out

### Disable Features

You can disable data collection:

**Disable log streaming:**
```toml
# ~/.anime-worker/config.toml
[logging]
enabled = false  # Don't send logs to server
```

**Disable metrics:**
```toml
[metrics]
enabled = false  # Don't send GPU/RAM metrics
```

**Reduce heartbeat frequency:**
```toml
heartbeat_interval_secs = 300  # Every 5 minutes instead of 30s
```

**Run completely offline:**
- Don't connect to server at all
- Use local subprocess mode instead
- No data sent anywhere

---

## 16. Questions & Concerns

### "Can the server see my generated videos?"

**No.** Videos go directly to S3. The API server only sees:
- ✅ Filename
- ✅ File size
- ✅ Duration/resolution
- ❌ NOT the actual video content

### "Can someone else see my worker stats?"

**No.** Workers are private by default:
- ✅ Only you can see your worker stats
- ✅ Only you can see jobs assigned to your worker
- ✅ Admins can see aggregate stats (total workers, avg usage)
- ❌ Other users cannot see your hardware info

### "What if I want to delete everything?"

Easy! Delete your worker:
```bash
# Via API or dashboard
DELETE /workers/{worker_id}

# This deletes:
# - Worker registration
# - All logs
# - All history
# - API keys
```

Your S3 files remain (you control S3 separately).

### "Can I use this on a VPN/Tor?"

**Yes!** The worker works fine behind:
- ✅ VPN
- ✅ Tor (might be slow)
- ✅ Corporate proxy
- ✅ NAT/firewall

Just make sure outbound WebSocket (WSS) connections are allowed.

---

## 17. Trust But Verify

### How to Audit

**Check what data is sent:**
```bash
# Inspect network traffic
tcpdump -i any -A 'host YOUR-API-SERVER'

# See WebSocket messages
# Worker logs show every message sent (info level)
```

**Check the code:**
```bash
# What data is sent?
cat src/protocol.rs  # All message types defined here

# How is hardware detected?
cat src/hardware.rs  # GPU/RAM/CPU detection logic

# Where are network calls?
cat src/client.rs    # WebSocket client
cat src/upload.rs    # HTTP upload logic
```

**No hidden data collection!**

---

## 18. Summary

### What IS Sent (and Why)

| Data | Why Needed | Frequency | Retention |
|------|-----------|-----------|-----------|
| Hardware stats | Job assignment | 30s | Latest only |
| Progress | User visibility | Per step | Until job done |
| Logs | Debugging | Real-time | 3+30 days |
| Videos | Results delivery | Per job | Until deleted |

### What is NOT Sent

❌ Personal data (name, email - only stored on signup)
❌ IP addresses
❌ File paths
❌ Other processes
❌ Network info
❌ Anything not needed for functionality

### Your Control

✅ Self-hosted (you own the infrastructure)
✅ Open source (audit the code)
✅ Configurable (disable features you don't want)
✅ Deletable (remove all data anytime)

---

## 19. Contact

**Questions about privacy?**
- GitHub Issues: https://github.com/YOUR-REPO/anime-worker/issues
- Email: privacy@your-domain.com (if you set one up)

**Want to improve privacy?**
- Pull requests welcome!
- Suggest features that reduce data collection
- Report any privacy concerns

---

## 20. Changes to This Document

**Last updated:** 2026-03-02
**Version:** 1.0.0

**We'll notify you if this changes:**
- Major changes = email notification
- Minor updates = shown in dashboard
- All changes tracked in git history

---

**TL;DR:** We only collect hardware stats and job progress needed for the system to work. No tracking, no personal data, no analytics. You control everything since you self-host!
