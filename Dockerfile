# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

# Set work directory
WORKDIR /app

# Install system dependencies if required for some python packages (like building faiss or similar)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create directories that might be needed at runtime
RUN mkdir -p data/pdf_files faiss_store

# Expose the Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
