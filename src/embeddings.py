"""
Input: .md file
Process: Clean .md file, create chunks, embed chunks, store in vector database
"""
from typing import List
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

def clean_content(content: str) -> str:
    """Cleans markdown content by removing headers, extra spaces, and newlines."""
    content = re.sub(r"#+\s", "", content)
    content = re.sub(r"\n{2,}", "\n\n", content)
    content = re.sub(r"([.!?]) {2,}", r"\1 ", content)
    content = "\n".join(line.strip() for line in content.splitlines())
    return content.strip()

def create_embeddings(md_file_loc: str) -> faiss.IndexFlatL2:
    """Reads a markdown file, cleans it, creates embeddings, and saves to FAISS index."""
    with open(md_file_loc, 'r', encoding='utf-8') as file:
        raw_content = file.read()
        print(f"Raw content length: {len(raw_content)}")

    content = clean_content(raw_content)

    splitter = SentenceTransformersTokenTextSplitter(
        model_name="all-MiniLM-L6-v2",
        chunk_size=100,
        chunk_overlap=20
    )
    chunks = splitter.split_text(content)

    embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))

    print("Data added to the vector DB")
    print(f"Number of vectors: {index.ntotal}")

    return index

if __name__ == "__main__":
    FILE_LOC: str = "output/sample_final.md"
    create_embeddings(FILE_LOC)
