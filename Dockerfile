# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Create workspace directories
RUN mkdir -p workspace/uploads workspace/audio workspace/boards workspace/models

# Expose the port the app runs on
EXPOSE 7860

# Run the application
# We use uvicorn for the FastAPI part. 
# If you want to use the Gradio standalone, replace with 'python app.py'
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
