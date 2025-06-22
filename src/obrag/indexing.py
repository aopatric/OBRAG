"""
Building and maintaing the vector store for your local OBRAG instance.
"""

from __future__ import annotations
from pathlib import Path
import time, hashlib, shutil, logging

from tqdm import tqdm
from typing import List
from collections import defaultdict
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import ObsidianLoader

from .config import Config

_META = ".meta.txt"


class Indexer:
    def __init__(self, cfg: Config, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.meta_file = cfg.persist_dir / _META
        self.logger.info(f"Indexer initialized.")

    # hash a markdown file
    def _md5(self, path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()
    
    # read a metadata file
    def _read_meta(self) -> set[str]:
        if not self.meta_file.exists():
            self.logger.warning(f"Metadata file {self.meta_file} does not exist. Attempted to read.")
            return set()
        try:
            return {line.strip() for line in self.meta_file.read_text().splitlines() if line.strip()}
        except Exception as e:
            self.logger.error(f"Failed to read metadata file {self.meta_file}: {e}")
            return set()
    
    # write to the metadata file
    def _write_meta(self, ids: set[str]) -> None:
        lines = "\n".join(sorted(ids))
        self.meta_file.write_text(lines, encoding="utf-8")

    # load existing valid vector store
    def _load_vstore(self) -> Chroma:
        embed_fn = HuggingFaceEmbeddings(
            model_name=self.cfg.embedding_model,
        )
        return Chroma(
            persist_directory=str(self.cfg.persist_dir),
            embedding_function=embed_fn,
            collection_name="OBRAG",
        )
    
    # turn documents into a list of chunk, id pairs
    def _get_ids(self, chunks: List[Document]) -> List[str]:
        ids : List[str] = []
        counters : dict[str, int] = defaultdict(int)
        for chunk in chunks:
            file_checksum = self._md5(Path(chunk.metadata["path"])) # type: ignore
            idx = counters[file_checksum]
            counters[file_checksum] += 1
            chunk_id = f"{file_checksum}_{idx}"
            ids.append(chunk_id) # type: ignore
        return ids

    # build a vector store from exisiting obsidian vault
    def _rebuild_vstore(self) -> Chroma:
        self.logger.info(f"Rebuilding vector store from Obsidian vault at {self.cfg.vault_path}")
        start_time = time.time()
        self.logger.info(f"Rebuild started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # delete existing vector store if necessary
        if self.cfg.persist_dir.exists():
            self.logger.warning(f"Vector store exists, but rebuild was requested. Deleting existing vector store.")
            shutil.rmtree(self.cfg.persist_dir)
        
        # build vector store
        chunks : List[Document] = ObsidianLoader(
            path=self.cfg.vault_path,
            collect_metadata=False
        ).load_and_split()

        embed_fn = HuggingFaceEmbeddings(
            model_name=self.cfg.embedding_model,
        ) # using only HuggingFaceEmbeddings for now, but can be extended later

        vstore = Chroma(
            persist_directory=str(self.cfg.persist_dir),
            embedding_function=embed_fn,
            collection_name="OBRAG",
        )

        ids = self._get_ids(chunks)

        for id, chunk in tqdm(zip(ids, chunks), desc="Adding documents to rebuilt vector store"):
            self.logger.debug(f"Adding document with ID {id} to vector store.")
            vstore.add_documents([chunk], ids=[id])

        self._write_meta(set(ids))  # write the set of ids to the metadatafile

        self.logger.info(f"Vector store rebuilt successfully with {len(chunks)} chunks.")

        self.logger.info(f"Rebuild completed at {time.strftime('%Y-%m-%d %H:%M:%S')}, total time taken: {time.time() - start_time:.2f} seconds")
        return vstore
    
    # update the vector store with any changes
    def _update_vstore(self, vstore: Chroma, old_ids: set[str]) -> Chroma:
        self.logger.info("Beginning vector store update...")

        chunks = ObsidianLoader(
            path=self.cfg.vault_path,
            collect_metadata=False
        ).load_and_split()

        id_list = self._get_ids(chunks)
        id_set = set(id_list)

        new = id_set - old_ids
        removed = old_ids - id_set

        self.logger.info(f"Found {len(new)} new documents and {len(removed)} removed documents.")

        if not new and not removed:
            self.logger.info("No changes detected in the vector store. Returning existing vector store.")
            return vstore
        
        if removed:
            self.logger.info(f"Removing {len(removed)} documents from vector store.")
            vstore.delete(ids=list(removed))
        
        if new:
            self.logger.info(f"Adding {len(new)} new documents to vector store.")
            for id, chunk in tqdm(zip(id_list, chunks), desc="Adding updated documents to vector store"):
                if id in new:
                    self.logger.debug(f"Adding updated document with ID {id} to vector store.")
                    vstore.add_documents([chunk], ids=[id])
        
        # write updated metadata
        self._write_meta(id_set)

        self.logger.info("Vector store update completed successfully.")
        return vstore




    # exposed function to get the vstore
    def get_vstore(self, force_refresh: bool = False) -> Chroma:
        self.logger.info("Getting vector store...")

        # if we get the refresh flag, start the rebuild right away
        if force_refresh:
            self.logger.info("Force refresh flag set. Rebuilding vector store.")
            return self._rebuild_vstore()
        
        # if nothing exists or the metadata is missing, rebuild
        if not self.cfg.persist_dir.exists() or not self.meta_file.exists():
            self.logger.warning(f"Vector store corrupted or missing at {self.cfg.persist_dir}. Rebuilding vector store.")
            return self._rebuild_vstore()

        # otherwise, load and update existing vector store
        old_vstore = self._load_vstore()
        old_ids = self._read_meta()
        return self._update_vstore(old_vstore, old_ids)