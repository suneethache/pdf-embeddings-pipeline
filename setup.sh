# This script installs all the required libraries and downloads the model weights required for parser

# Install python dependencies
pip3 install -r requirements.txt

# Download the helper script
wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py

# Run the helper script

python download_models_hf.py
