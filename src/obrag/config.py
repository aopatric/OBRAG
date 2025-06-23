from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any
from getpass import getpass
from langchain_community.document_loaders import ObsidianLoader
from langchain_huggingface import HuggingFaceEmbeddings
from logging import Logger
from openai import OpenAI
from time import sleep

from .utils import *

import os
import tomllib
import tomli_w

@dataclass(slots=True)
class Config:
    vault_path: Path = Path("/home/saafetensors/Documents/ollin/")
    persist_dir: Path = Path("./data")
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    chat_model: str = "gpt-4o-mini"
    top_k: int = 5
    temperature: float = 0.1
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    first_run: bool = True

    @property
    def vault_exists(self) -> bool:
        return self.vault_path.exists()

def _path_fixing_factory(items: Any) -> dict[str, Any]:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in items}


# helpers to merge from sources into one config object
_CFG_FILE = Path.home() / ".config" / "obrag" / "config.toml"

def _from_toml() -> dict[Any, Any]:
    if _CFG_FILE.is_file():
        try:
            with open(_CFG_FILE, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            print(f"Error reading config file {_CFG_FILE}: {e}")
    return {}

def _from_env() -> dict[Any, Any]:
    env_map = {
        "vault_path": os.getenv("OBRAG_VAULT_PATH"),
        "persist_dir": os.getenv("OBRAG_PERSIST_DIR"),
        "embedding_model": os.getenv("OBRAG_EMBEDDING_MODEL"),
        "chat_model": os.getenv("OBRAG_CHAT_MODEL"),
        "top_k": os.getenv("OBRAG_TOP_K"),
        "temperature": os.getenv("OBRAG_TEMPERATURE"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
    }
    clean_values : dict[Any, Any] = {}
    for k, v in env_map.items():
        if v is None:
            continue
        if k in ["top_k", "temperature"]:
            clean_values[k] = float(v) if "." in v else int(v)
        else:
            clean_values[k] = v
    return clean_values

def get_config(**overrides: Any) -> Config:
    cfg_dict = asdict(Config())
    cfg_dict |= _from_toml()
    cfg_dict |= _from_env()
    cfg_dict |= overrides
    cfg_dict["vault_path"] = Path(cfg_dict["vault_path"])
    cfg_dict["persist_dir"] = Path(cfg_dict["persist_dir"])
    return Config(**cfg_dict)

def save_config(cfg: Config) -> None:
    path = _CFG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(asdict(cfg, dict_factory=_path_fixing_factory), f) # type: ignore
    
def startup_wizard(init_cfg: Config, logger: Logger) -> None:
    print("An issue has been detected with your configuration. Welcome to the OBRAG startup wizard!")
    cfg = init_cfg

    # ask for vault path until ObsidianLoader can find documents
    while True:
        print(f"Vault directory currently set at {cfg.vault_path}. Enter a new one or leave empty to try the existing setting.")
        vault_path = input(f"Enter a new vault path [{cfg.vault_path}]: ").strip(' \'\"')

        if not vault_path:
            vault_path = cfg.vault_path
        
        # check if the given vault path is valid
        try:
            print(f"Trying to verify path {vault_path}...")
            vault_path = Path(vault_path)

            if not vault_path.exists():
                raise FileNotFoundError("The specified vault location does not exist. Please try again.")
            
            loader = ObsidianLoader(path=vault_path, collect_metadata=False)
            docs = loader.load()
            print(f"Vault location {vault_path} valid. Found {len(docs)} files.")
            cfg = replace(cfg, vault_path=vault_path)
            break
        except Exception as e:
            print(f"Error with vault selection: {e}")
            continue
    
    print(f"Vault path valid and set to {vault_path}.")
    print(f"\n\n")
    
    # set persist directory, shouldn't cause issues since we create it later
    print(f"Persist directory is set to [{cfg.persist_dir}]. Leave empty to use default or enter a new path.")
    path = input(f"Persist directory [{cfg.persist_dir}]: ").strip()
    if path:
        cfg = replace(cfg, persist_dir=Path(path))
    print(f"Persist directory set to {cfg.persist_dir}")
    print("\n\n")
    
    # check for embedding model
    print(f"Embedding model is set to '{cfg.embedding_model}'. Leave empty to use current setting or enter a new model.")
    print(f"Note that for now, we only support HuggingFace embedding models.")

    while True:
        embed_model = input(f"Enter a new embedding model [{cfg.embedding_model}]:").strip(' \'"')

        if not embed_model:
            embed_model = cfg.embedding_model
        
        print(f"Attempting to load model {embed_model}...")
        # check if HuggingFaceEmbeddings works
        try:
            _ = HuggingFaceEmbeddings(model_name=embed_model)
            print(f"Model sucessfully loaded.")
            cfg = replace(cfg, embedding_model=embed_model)
            break
        except Exception as e:
            print(f"Error loading model: {e}")
            continue
    
    print(f"Using model '{embed_model}' for embeddings.")
    print(f"\n\n")
    
    # check for chat model
    print(f"Chat model is set to {cfg.chat_model}. Leave empty to use default or enter a new model.")
    print(f"Note that for now, we only support OpenAI chat models. This step is not verified! Enusre your model name is correct or reconfigure.")

    # leaving in a loop for future verification logic
    while True:
        chat_model = input(f"Enter a chat model to use [{cfg.chat_model}]: ").strip(' \'"')

        if not chat_model:
            chat_model = cfg.chat_model
        
        cfg = replace(cfg, chat_model=chat_model)
        break
    print(f"Chat model set to {chat_model}.")
    print(f"\n\n")

    # get API key if not specified
    print(f"The OpenAI API key is set in the environment at runtime, and stored in your config file.")
    print(f"Enter a new API key or leave blank to attempt to load existing.")
    while True:
        api_key = getpass("Enter your OpenAI API key (hidden) [default]: ").strip(' \'"')

        if not api_key:
            api_key = cfg.openai_api_key
        
        # try to ping the api with this key
        try:
            os.environ["OPENAI_API_KEY"] = api_key
            tester = OpenAI()
            _ = tester.responses.create(
                model="gpt-4.1",
                input="test"
            )
            cfg = replace(cfg, openai_api_key=api_key)
            break
        except Exception as e:
            print(f"Error testing API key: {e}")
            continue

        # note: temperature and top-k moved to a generation wizard for later.
    print(f"API key set successfully.")
    print("\n\n")
    
    cfg = replace(cfg, first_run=False)
    save_config(cfg)
    print(f"Configuration saved successfully at {_CFG_FILE}... continuing in 3 seconds")
    sleep(3)
    clear_console()