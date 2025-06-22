from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any
from getpass import getpass
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
    
def startup_wizard(init_cfg: Config) -> None: # change this later
    print("An issue has been detected with your configuration. Welcome to the OBRAG startup wizard!")
    cfg = init_cfg

    # check vault path
    if not cfg.vault_exists:
        print(f"Provided vault directory {cfg.vault_path} does not exist.")
        p = input("Please enter a valid Obsidian vault path: ").strip()
        if p:
            cfg = replace(cfg, vault_path=Path(p))

        # make the parent directory if it doesnt exist
        cfg.vault_path.mkdir(parents=True, exist_ok=True)
    
    # check for persist directory
    print(f"Persist directory is set to {cfg.persist_dir}. Leave empty to use default or enter a new path.")
    path = input(f"Persist directory [{cfg.persist_dir}]: ").strip()
    if path:
        cfg = replace(cfg, persist_dir=Path(path))
    
    # check for embedding model
    print(f"Embedding model is set to {cfg.embedding_model}. Leave empty to use default or enter a new model.")
    print(f"Note that for now, we only support HuggingFace embedding models.")
    model = input(f"Embedding model [{cfg.embedding_model}]: ").strip()
    if model:
        cfg = replace(cfg, embedding_model=model)
    
    # check for chat model
    print(f"Chat model is set to {cfg.chat_model}. Leave empty to use default or enter a new model.")
    print(f"Note that for now, we only support OpenAI chat models.")
    chat_model = input(f"Chat model [{cfg.chat_model}]: ").strip()
    if chat_model:
        cfg = replace(cfg, chat_model=chat_model)
    
    # check for top_k
    print(f"Top K is set to {cfg.top_k}. Leave empty to use default or enter a new value.")
    top_k = input(f"Top K [{cfg.top_k}]: ").strip()
    if top_k:
        try:
            cfg = replace(cfg, top_k=int(top_k))
        except ValueError:
            print("Invalid value for Top K. Using default value.")
    
    # check for temperature
    print(f"Temperature is set to {cfg.temperature}. Leave empty to use default or enter a new value.")
    temperature = input(f"Temperature [{cfg.temperature}]: ").strip()
    if temperature:
        try:
            cfg = replace(cfg, temperature=float(temperature))
        except ValueError:
            print("Invalid value for temperature. Using default value.")

    # check for API key(s) (only OpenAI for now)
    print(f"Checking for OpenAI API key...")
    if cfg.openai_api_key:
        print("OpenAI API key found in environment variables.")
    else:
        print("No OpenAI API key found. You will need it to use the chat model.")
    if not os.getenv("OPENAI_API_KEY"):
        key = getpass("Enter your OpenAI API key: ").strip()
        os.environ["OPENAI_API_KEY"] = key
        persist = input("Do you want to save this API key in the config file? (y/n): ").strip().lower() == "y"

        if persist:
            cfg = replace(cfg, openai_api_key=key)
    
    cfg = replace(cfg, first_run=False)
    save_config(cfg)
    print(f"Configuration saved successfully at {_CFG_FILE}")