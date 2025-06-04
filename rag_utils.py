"""
Utils for reading and writing to the vector store during RAG operations.
"""

import configparser
import os
import glob
import torch
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_config(config_path: str="config.ini", profile: str="DEFAULT") -> dict:
    """
    Read the configuration file and return a dictionary of the values.
    """
    print(f"Reading config from {config_path}...")
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    config.read(config_path)

    # convert to dictionary for ease of use
    config_dict = {
        "vault_path": config.get(profile, "vault_path"),
        "persist_dir": config.get(profile, "persist_dir"),
        "embedding_model": config.get(profile, "embedding_model"),
        "collection_name": config.get(profile, "collection_name"),
        "normalize_embeddings": config.getboolean(profile, "normalize_embeddings"),
        "chunk_size": int(config.get(profile, "chunk_size")),
        "overlap": int(config.get(profile, "overlap")),
        "language_model": config.get(profile, "language_model"),
        "top_k": int(config.get(profile, "top_k"))
    }

    return config_dict

def get_markdown_documents(vault_path: str) -> list[Document]:
    """
    Recursively load all markdown files from the vault and return a list of Document objects.
    """
    print(f"Loading markdown documents from {vault_path}...")
    mds = []
    search_path = os.path.join(vault_path, "**", "*.md")
    for md_file in glob.glob(search_path, recursive=True):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            rel_path = os.path.relpath(md_file, vault_path)
            metadata = {
                "source": rel_path,
                "full_path": md_file,
            }
            mds.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            print(f"Error reading {md_file}: {e}")
    return mds
             
def chunk_documents(documents: list[Document], chunk_size: int, overlap: int) -> list[Document]:
    """
    Take a list of documents and chunk them into smaller documents for retrieval.
    """
    print(f"Chunking {len(documents)} documents into chunks of {chunk_size} with {overlap} overlap...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    return chunks

def load_embedding_model(model_name: str, normalize_embeddings: bool) -> HuggingFaceEmbeddings:
    """
    Load the embedding model from HuggingFace.
    """
    print(f"Loading embedding model {model_name}...")
    model_kwargs = {
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    encode_kwargs = {
        "normalize_embeddings": normalize_embeddings
    }
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def build_db(
    documents: list[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_dir: str,
    collection_name: str,
) -> Chroma:
    """
    Create or load the vector store from the documents.
    """
    print(f"Building vector store at {persist_dir}...")

    # delete existing db if it exists
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    os.makedirs(persist_dir, exist_ok=True)

    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    return db

def load_db(persist_dir: str, embedding_model: HuggingFaceEmbeddings, collection_name: str) -> Chroma:
    print(f"Loading vector store from {persist_dir}...")
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"Vector store not found at {persist_dir}")
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=collection_name
    )
    return db

# test if ran as main
if __name__ == "__main__":
    config = read_config()
    mds = get_markdown_documents(config["vault_path"])
    chunks = chunk_documents(mds, config["chunk_size"], config["overlap"])
    model = load_embedding_model(config["embedding_model"], config["normalize_embeddings"])
    db = build_db(chunks, model, config["persist_dir"], config["collection_name"])
    print("Created new persistent vector store with {} items.".format(db._collection.count()))
    db = load_db(config["persist_dir"], model, config["collection_name"])
    print("Loaded existing vector store with {} items.".format(db._collection.count()))