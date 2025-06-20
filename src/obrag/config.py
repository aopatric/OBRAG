from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
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

    @property
    def vault_exists(self) -> bool:
        return self.vault_path.exists()

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
    cfg_dict["vault_path"] = str(Path(cfg_dict["vault_path"]))
    cfg_dict["persist_dir"] = str(Path(cfg_dict["persist_dir"]))
    return Config(**cfg_dict)

def save_config(cfg: Config) -> None:
    path = _CFG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(asdict(cfg), f) # type: ignore
    
def startup_wizard(init_cfg: Config) -> None: # change this later
    cfg = init_cfg # type: ignore
    pass