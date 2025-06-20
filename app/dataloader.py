"""

Dataloading wrapper class for grabbing docs and processing.

"""
import logging
import os
import shutil

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import ObsidianLoader
from langchain_chroma import Chroma
from typing_extensions import List
from tqdm import tqdm

from app.DEFAULTS import *
from app.logging import start_logger

PERSIST_DIR = "./data"

class DataLoader:
    def __init__(self, vault_path: str = DEFAULT_VAULT_PATH, embedding_modelname: str = DEFAULT_EMBEDDING_MODELNAME, logger: logging.Logger = start_logger("dataloading")):
        self.loader = ObsidianLoader(path=vault_path, collect_metadata=False)
        self.embedding_fn = HuggingFaceEmbeddings(model_name=embedding_modelname)
        self.logger = logger
    
    def gather_chunks(self) -> List[Document]:
        """
        Load the chunks from the Obsidian vault and return a list of Document objects.
        """
        self.chunks = self.loader.load_and_split()
        return self.chunks
    
    def get_vstore(self):
        """
        Rebuild the vector store from the loaded chunks using the specified embedding model.
        """
        if not hasattr(self, 'chunks'):
            raise ValueError("Chunks not loaded. Please call gather_chunks() first.")

        # check if the persist directory exists and wipe if it does
        if os.path.exists(PERSIST_DIR):
            self.logger.warning(f"Persist directory {PERSIST_DIR} exists. Wiping it clean.")
            shutil.rmtree(PERSIST_DIR)

        # build the next vector store from the chunks (iterative to avoid OOM)
        self.logger.info("Building vector store from loaded chunks...")
        self.vstore = Chroma(
            collection_name="OBRAG",
            persist_directory=PERSIST_DIR,
            embedding_function=self.embedding_fn,
        )
        for chunk in tqdm(self.chunks):
            self.vstore.add_documents([chunk])
        
        self.logger.info("Vector store built successfully. Returning.")
        return self.vstore
