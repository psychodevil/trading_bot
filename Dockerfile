# Containerized Quantitative Trading & Backtesting Environment
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies in container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY . .

# Expose port for reports and live dashboard
EXPOSE 8088

# Default command runs the fast walk-forward simulation
CMD ["python", "run_walkforward_portfolio.py"]
