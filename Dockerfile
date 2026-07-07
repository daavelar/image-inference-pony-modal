# Use a runtime image instead of devel to save space (~6-8GB reduction)
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Define working directory
WORKDIR /app

# Install only necessary system dependencies (removed git as it's not needed for execution)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy local requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH="/runpod-volume/models/untitledpony_b10.safetensors"

# Ensure the models and loras directories exist
RUN mkdir -p models loras outputs

# Execute the handler
CMD ["python", "-u", "handler.py"]
