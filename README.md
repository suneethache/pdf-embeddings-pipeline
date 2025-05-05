# PDF to Markdown Embedding Pipeline

This project extracts content from complex documents (text,tables, images, different layout) from PDF documents,
converts it to Markdown, and generates sentence-level embeddings for semantic search using FAISS and multilingual models.

## Features

- Extracts structured text and tables using [Magic-PDF (MinerU)](https://github.com/opendatalab/MinerU)
- Converts extracted content into clean Markdown
- Splits Markdown into semantic chunks
- Embeds chunks using Sentence Transformers
- Stores embeddings in a FAISS vector index

## Folder Structure
```
├── embeddings.py # Embeds markdown content and stores in FAISS
├── parser.py # Parses PDF and generates markdown using MinerU
├── setup.sh # Shell script to install dependencies and download models
├── outputs/ # Output folder for markdown and images
├── samples/ # Folder to place sample PDFs
└── requirements.txt # Python dependencies
```

## Setup Instructions

By running the following command, 
- it downloads the weights needed for the parser
- creates python virtual environment
- installs all the dependencies 

```bash
bash setup.sh
```

### Run Pipeline
Extract content from PDF and generate .md
- creates outputs folder a
- dumps all the parsing outputs and images

``` bash
python parser.py
```
### Generate embeddings from .md and store the embeddings in Faiss database

- takes the md file from the parser
- preprocess it and create sentence embeddings
- store the embeddings in Faiss vector database

``` bash
python embeddings.py
```
