# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies required for PDF processing (e.g., UnstructuredPDFLoader)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
# This includes app.py, index.html, etc.
COPY . .

# Create the uploads directory where PDFs will be temporarily stored
RUN mkdir -p uploads

# Expose the port that the Flask app will run on
EXPOSE 5000

# Define environment variables (optional, can also be in docker-compose.yml)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Command to run the Flask application
CMD ["flask", "run"]