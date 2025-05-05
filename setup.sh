# This script installs all the required libraries and downloads the model weights required for parser

python -m venv env
source env/bin/activate

pip install -U "magic-pdf[full]"
# Install python dependencies
pip install -r requirements.txt

# Downloading the weights for the parser

pip install huggingface_hub
wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py
python download_models_hf.py

