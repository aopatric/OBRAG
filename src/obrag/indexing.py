"""
Building and maintaing the vector store for your local OBRAG instance.
"""
import logging

from typing import Optional
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import ObsidianLoader
from langchain.indexes import SQLRecordManager, index, IndexingResult
from .config import Config

COLLECTION_NAME = "vault"
DATABASE_URL = "sqlite:///data/cache.sql"

class Indexer:
    def __init__(self, cfg: Config, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.vstore : Optional[Chroma] = None
        self.record_manager : Optional[SQLRecordManager] = None
        self.embed_fn : Optional[HuggingFaceEmbeddings] = None
        self._setup()
    
    # initilize relevant objects for the indexing process
    def _setup(self) -> None:
        self.logger.info("Beginning Indexer setup...")
        self.embed_fn = HuggingFaceEmbeddings(model_name=self.cfg.embedding_model)

        # make sure the data directory exists
        if not Path("./data").exists():
            Path("./data").mkdir(parents=True, exist_ok=True)

        self.record_manager = SQLRecordManager(
            namespace=f"OBRAG/{COLLECTION_NAME}",
            db_url=DATABASE_URL
        )
        self.record_manager.create_schema()
        self.vstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
            persist_directory=str(self.cfg.persist_dir),
        )
        self.logger.info("Indexer setup complete.")

    # helper to clear existing vector store
    def _clear_vstore(self) -> None:
        # make sure these both exist
        assert self.vstore is not None, "Vector store is not initialized."
        assert self.record_manager is not None, "Record manager is not initialized."

        # clear by indexing an empty list with overwrite
        self.logger.info("Hard clearing vector store...")
        index([], self.record_manager, self.vstore, cleanup="full", source_id_key="path")
        self.logger.info("Vector store cleared.")
    
    # index the vault by loading all docs and using the indexer
    def _index_vault(self) -> tuple[IndexingResult, Chroma]:
        # ensure both vstore and record manager are initialized
        assert self.vstore is not None, "Vector store is not initialized."
        assert self.record_manager is not None, "Record manager is not initialized."

        # get document chunks
        chunks = ObsidianLoader(
            path=self.cfg.vault_path,
            collect_metadata=False
        ).load()

        result = index(
            chunks,
            self.record_manager,
            self.vstore,
            cleanup="incremental",
            source_id_key="path",
            batch_size=1,
        )
        return result, self.vstore

    
    # get all documents, update vector store using indexer
    def get_vstore(self, force_refresh: bool = False) -> tuple[IndexingResult, Chroma]:
        if force_refresh:
            self._clear_vstore()
        return self._index_vault()