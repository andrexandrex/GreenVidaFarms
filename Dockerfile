# Use a lightweight Python base image
FROM python:3.10-slim

# 1. System dependencies (adjust as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements and install Python dependencies
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of your app code
COPY . /app

# 4. Expose Shiny for Python default port
EXPOSE 3838

# 5. Default command to run the Shiny app
CMD ["python", "-m", "shiny", "run", "--host", "0.0.0.0", "--port", "3838", "app.py"]
