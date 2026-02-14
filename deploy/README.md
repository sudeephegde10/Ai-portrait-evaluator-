# AI Portrait Evaluator - GCP Deployment Guide

## Prerequisites
1.  **GCP Account**: A Google Cloud Platform account.
2.  **e2-micro Instance**: 
    - OS: Debian 11/12 or Ubuntu 20.04/22.04 (Recommended).
    - Firewall: Check "Allow HTTP traffic" when creating the VM.

## Deployment Steps

1.  **SSH into your VM**:
    Use the GCP Console "SSH" button or your terminal.

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/sudeephegde10/Ai-portrait-evaluator-.git
    cd Ai-portrait-evaluator-
    ```

3.  **Run the Setup Script**:
    ```bash
    chmod +x deploy/setup_vm.sh
    ./deploy/setup_vm.sh
    ```
    *This script will:*
    - Install necessary system libraries (fixing the `libxcb` error).
    - Create a 2GB swap file (essential for 1GB RAM VMs).
    - Set up the Python environment.
    - Install and start the web server on port 80.

4.  **Access your App**:
    Visit `http://YOUR_VM_EXTERNAL_IP` in your browser.

## Troubleshooting
- **If the site doesn't load**: Check if your VM has an External IP and if "Allow HTTP traffic" is enabled in the VM details.
- **View Logs**:
    ```bash
    sudo journalctl -u ai-portrait -f
    ```
