FROM rust:latest as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Cargo files
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src/ ./src/

# Build the release binary
RUN cargo build --release

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary from builder
COPY --from=builder /build/target/release/anime-worker /usr/local/bin/anime-worker

# Copy Python scripts
COPY python/ ./python/

# Install Python dependencies
RUN pip install --no-cache-dir -r python/requirements.txt

# Install HuggingFace CLI
RUN pip install --no-cache-dir "huggingface_hub[cli]"

# Create directories
RUN mkdir -p /app/models /root/.anime-worker

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Expose for health checks if needed
EXPOSE 8080

ENTRYPOINT ["docker-entrypoint.sh"]
