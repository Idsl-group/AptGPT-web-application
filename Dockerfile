# Use the official Python image as the base image
FROM python:3.10.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ARNIEFILE=arnie_config.txt

# Install gcc and other build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    git \
    cmake \
    libncurses5-dev \
    libreadline-dev \
    zlib1g-dev \
    wget \
    vienna-rna \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory
WORKDIR /AptGpt-web-application

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8080

# Set Streamlit configuration options (optional)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEWEBRTC=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create a non-root user for better security
RUN useradd -m streamlituser
RUN chown -R streamlituser:streamlituser /AptGpt-web-application
USER streamlituser

# Command to run when starting the container
CMD ["streamlit", "run", "app.py"]
