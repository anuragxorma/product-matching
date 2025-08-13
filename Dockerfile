FROM python:3.11-slim

# Minimal system deps: git for installing CLIP from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install core deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch wheels explicitly (smaller)
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy code
COPY pm_lib.py app.py ./ 
# Include sample data so the container runs out-of-the-box
COPY data ./data

# Expose API port
EXPOSE 9000

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
