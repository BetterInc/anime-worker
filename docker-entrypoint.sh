#!/bin/bash
set -e

CONFIG_DIR="/root/.anime-worker"
CONFIG_FILE="$CONFIG_DIR/config.toml"

mkdir -p "$CONFIG_DIR"

# Check if config exists
if [ -f "$CONFIG_FILE" ]; then
    echo "✅ Using existing worker config"

    # Ensure python_scripts_dir is set
    if ! grep -q "python_scripts_dir" "$CONFIG_FILE" 2>/dev/null; then
        echo "python_scripts_dir = \"${PYTHON_SCRIPTS_DIR:-/app/python}\"" >> "$CONFIG_FILE"
        echo "✅ Added python_scripts_dir to config"
    fi
else
    echo "⚠️  No worker config found - auto-registering..."

    # Wait for API to be ready
    until curl -sf "${SERVER_URL:-http://api:8000}/health" > /dev/null; do
        echo "⏳ Waiting for API..."
        sleep 2
    done

    # Auto-register via public endpoint
    RESPONSE=$(curl -s -X POST "${SERVER_URL:-http://api:8000}/workers/auto-register" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"${WORKER_NAME:-Docker Worker}\"}")

    WORKER_ID=$(echo "$RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    API_KEY=$(echo "$RESPONSE" | grep -o '"api_key":"[^"]*"' | cut -d'"' -f4)

    if [ -z "$WORKER_ID" ] || [ -z "$API_KEY" ]; then
        echo "❌ Failed to auto-register. Response: $RESPONSE"
        exit 1
    fi

    echo "✅ Worker auto-registered: $WORKER_ID"

    # Create config
    cat > "$CONFIG_FILE" <<EOF
server_url = "${SERVER_URL:-http://api:8000}"
worker_id = "$WORKER_ID"
api_key = "$API_KEY"
worker_name = "${WORKER_NAME:-Docker Worker}"
models_dir = "${MODELS_DIR:-/app/models}"
python_path = "${PYTHON_PATH:-python3}"
python_scripts_dir = "${PYTHON_SCRIPTS_DIR:-/app/python}"
heartbeat_interval_secs = ${HEARTBEAT_INTERVAL_SECS:-30}

[constraints]
max_model_size_gb = ${MAX_MODEL_SIZE_GB:-50}
EOF

    echo "✅ Config created"
fi

echo "🚀 Starting worker..."
exec anime-worker run
