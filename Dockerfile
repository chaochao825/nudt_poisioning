FROM python:3.10-slim

WORKDIR /workspace

# Copy only what's needed for installation first to leverage cache
# But here we'll just copy everything for simplicity since it's small
COPY . /workspace

# Install Python dependencies (using wheels to avoid compilation)
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    numpy \
    Pillow

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python3", "main.py"]
