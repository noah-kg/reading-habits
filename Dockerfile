# Use a maintained base image
FROM python:3.10-slim

# Metadata
LABEL author="Noah Gampe"
LABEL email="noah.gampe@gmail.com"

# Optional: argument for OS-level dependencies
ARG DEPS="make"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends $DEPS && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ADD . /app
