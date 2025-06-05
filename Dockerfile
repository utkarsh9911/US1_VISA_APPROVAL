FROM python:3.9.19-slim

WORKDIR /app

# Install system packages required for building Python packages
RUN apt-get update && \
    apt-get install -y gcc g++ build-essential python3-dev libffi-dev libssl-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /app

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]
