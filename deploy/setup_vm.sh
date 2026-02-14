#!/bin/bash
set -e

# Define directories
APP_DIR=$(pwd)

echo "----------------------------------------------------------------"
echo "Starting AI Portrait Evaluator Deployment on GCP (e2-micro)"
echo "----------------------------------------------------------------"

# 1. System Updates & Dependencies
echo "[1/4] Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1

# 2. Swap Setup (Critical for 1GB RAM)
echo "[2/4] Checking swap configuration..."
if ! swapon --show | grep -q '/swapfile'; then
    echo "Creating 2GB swap file to prevent OOM errors..."
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "Swap created successfully."
else
    echo "Swap file already exists. Skipping."
fi

# 3. Python Environment Setup
echo "[3/4] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Install dependencies in venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# 4. Systemd Service Configuration
echo "[4/4] Configuring systemd service..."

SERVICE_FILE="ai-portrait.service"

# Generate service file
# Note: Running as root to bind to port 80. For production, consider using Nginx + non-root user.
cat <<EOF > $SERVICE_FILE
[Unit]
Description=AI Portrait Evaluator Web Server
After=network.target

[Service]
User=root
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/gunicorn app:app --bind 0.0.0.0:80 --workers 1 --threads 4 --timeout 120 --log-level info
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Install and start service
sudo mv $SERVICE_FILE /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-portrait
sudo systemctl restart ai-portrait

echo "----------------------------------------------------------------"
echo "Deployment Complete!"
echo "----------------------------------------------------------------"
echo "Your app should be live on the external IP of this VM."
echo "If you cannot access it, ensure 'Allow HTTP traffic' is checked in GCP VM settings."
