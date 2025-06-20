"""
Building and maintaing the vector store for your local OBRAG instance.
"""

from __future__ import annotations
from pathlib import Path
import json, time, hashlib, shutil, logging

from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import ObsidianLoader

from .config import Config

_META = ".meta.json"

# hash a markdown file
def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

class Indexer:
    def __init__(self, cfg: Config, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.vstore = None
        self.logger.info(f"Indexer initialized with config: {self.cfg}")

    # build a vector store
    def build_vstore(self, force: bool = False):
        self.logger.info("Starting vector store build process...")
        persist_dir = self.cfg.persist_dir
        meta_file = persist_dir / _META

        # if we already have a vector store and we don't force it, just load it
        if persist_dir.exists() and meta_file.exists() and not force:
            self.logger.info(f"Vector store already exists at {persist_dir}, loading...")
            return self.load_vstore()
        
        # otherwise, we build it
        self.logger.info(f"Building vector store at {persist_dir}...")
        start = time.time()
        if force and persist_dir.exists():
            self.logger.info(f"Force rebuild requested, removing existing vector store at {persist_dir}...")
            shutil.rmtree(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)


        # gather notes
        self.logger.info(f"Gathering notes from Obsidian vault at {self.cfg.vault_path}...")
        chunks = ObsidianLoader(
            path=self.cfg.vault_path,
            collect_metadata=False
        ).load_and_split()
        self.logger.info(f"Gathered {len(chunks)} chunks from vault at {self.cfg.vault_path}")
        embed_fn = HuggingFaceEmbeddings(model_name=self.cfg.embedding_model)

        self.logger.info(f"Using embedding model: {self.cfg.embedding_model}")
        self.logger.info(f"Building vector store at {persist_dir}...")
        vstore = Chroma(
            collection_name="OBRAG",
            persist_directory=str(persist_dir),
            embedding_function=embed_fn,
        )

        # add chunks while avoiding OOM
        self.logger.info("Adding chunks to vector store...")
        for i in tqdm(range(len(chunks)), desc="Adding chunks", unit="chunk"):
            vstore.add_documents([chunks[i]])
        self.logger.info(f"Added {len(chunks)} chunks to vector store.")

        # save metadata
        self.logger.info(f"Saving metadata to {meta_file}...")
        checksums = {  # type: ignore
            _md5(Path(chunk.metadata["path"])): chunk.metadata["path"] # type: ignore
            for chunk in chunks
        }
        meta_file.write_text(json.dumps(checksums, indent=2))
        self.logger.info(f"Vector store built in {time.time() - start:.2f} seconds.")
        return vstore


    # load an existing vector store
    def load_vstore(self) -> Chroma:
        embed_fn = HuggingFaceEmbeddings(model_name=self.cfg.embedding_model)
        return Chroma(
            collection_name="OBRAG",
            persist_directory=str(self.cfg.persist_dir),
            embedding_function=embed_fn,
        ) 