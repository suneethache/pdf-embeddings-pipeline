"""
    Input: .md file
    process: clean .md file, create chunks, embed chunks, store in vector database
"""
from langchain.text_splitter import  SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

def clean_content(content):
    
    # Remove headings (##, ###, etc.)
    content = re.sub(r"#+\s", "", content)
    # Remove extra whitespace
    content = re.sub(r"\n{2,}", "\n\n", content)
    # Remove extra spaces after sentence-ending punctuation
    content = re.sub(r"([.!?]) {2,}", r"\1 ", content)

    # Remove multiple newlines
    content = re.sub(r"\n{2,}", "\n\n", content)

    # Optional: strip leading/trailing spaces from each line
    content = "\n".join(line.strip() for line in content.splitlines())

    return content.strip()

def create_embeddings(md_file_loc):

    # read the mark down file
    with open(md_file_loc, 'r') as f:
        raw_content = f.read()
        #print(raw_content)
        print(len(raw_content))

    # preprocess, cleaning
    content = clean_content(raw_content)

    # chunk the data
    splitter = SentenceTransformersTokenTextSplitter(
        model_name="all-MiniLM-L6-v2",
        chunk_size=100,
        chunk_overlap=20
        )
    
    chunks = splitter.split_text(content)

    # Embeddings for each chunk data
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-base") # best model for multilanguage content
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)

    # Save the embeddings  in to a vector database
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))

    print("Data added to the vector DB")
    print(index.ntotal)

    return index


if __name__ == "__main__":
    file_loc = "output/sample_final.md"
    create_embeddings(file_loc)