# Worker Testing Guide

## 🧪 Local CPU Testing (Development)

For testing the system locally WITHOUT a GPU:

### Setup

```bash
cd python

# Create venv
python3 -m venv venv

# Install CPU-only PyTorch (faster, smaller)
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements-cpu.txt
```

**Why CPU-only?**
- ✅ Smaller download (~500MB vs ~5GB)
- ✅ No CUDA required
- ✅ Works on any machine
- ✅ Good for testing system flow
- ⚠️ Slower inference (but OK for tiny test videos)

---

## 🚀 GPU Production Setup

For real GPU workers:

```bash
cd python

# Create venv
python3 -m venv venv

# Install GPU PyTorch (with CUDA)
source venv/bin/activate
pip install -r requirements.txt
```

**Why GPU version?**
- ✅ 100x faster inference
- ✅ Can handle full resolution
- ✅ Production-ready
- ⚠️ Requires NVIDIA GPU
- ⚠️ Larger download (~5GB)

---

## 🎯 Test Model Configuration

### CPU Test Model

**In models_config.yaml:**
```yaml
test_cpu:
  name: "CPU Test Model"
  size_gb: 20        # Model size
  # Ultra-small settings in config/models/test_cpu.yaml:
  width: 64          # Tiny!
  height: 64
  num_frames: 5      # Just 5 frames
  num_inference_steps: 5  # Fast
```

**Performance:**
- CPU (Intel i7): ~1-2 minutes
- GPU (RTX 3090): ~5-10 seconds

**Output:**
- 64x64 pixel video
- 5 frames @ 8fps = 0.625 seconds
- File size: ~100KB

---

## 📝 Configuration Files

### requirements.txt (GPU - Production)
```txt
torch>=2.1.0        # With CUDA
diffusers>=0.31.0
...
```

### requirements-cpu.txt (CPU - Local Testing)
```txt
# Install PyTorch CPU separately
diffusers>=0.31.0
...
```

---

## 🔧 Worker Config for Testing

```toml
server_url = "http://localhost:8000"
worker_id = "..."
api_key = "aw_..."
worker_name = "Local CPU Test"

[constraints]
supported_models = ["test_cpu"]  # Only test model
max_model_size_gb = 25           # Enough for Wan 5B
```

---

## ✅ Testing Checklist

- [ ] Python venv created
- [ ] CPU PyTorch installed
- [ ] Dependencies installed
- [ ] Worker config created
- [ ] Worker starts successfully
- [ ] Worker connects to API
- [ ] Worker shows "Online" in UI
- [ ] Model downloads
- [ ] Test video generates
- [ ] Video uploads to S3
- [ ] Video appears in UI

---

## 🎉 Success Criteria

Test passes if you can:
1. Create a project
2. Generate a preview
3. See the generated 64x64 video in the Previews page
4. Download and play the video

**Time to complete:** ~30 minutes total
