# Use NVIDIA CUDA 12.1 as the base
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3-pip python3-dev && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# Copy all scripts
COPY . .

# Expose Port 8080
EXPOSE 8080

# Start the AI server
CMD ["python3", "main.py"]
