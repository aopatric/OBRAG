from .config import get_config, startup_wizard
from .logging import start_logger
from .utils import *
from .indexing import Indexer

def main():
    print_banner()
    cfg = get_config()
    logger = start_logger()
    logger.info("OBRAG CLI started.")

    logger.info(f"Verifying configuration...")
    valid_cfg = cfg.vault_exists and cfg.openai_api_key != ""

    if not valid_cfg:
        print("Configuration is incomplete. Starting setup wizard...")
        logger.warning("Configuration is incomplete. Starting setup wizard...")
        startup_wizard(cfg)
    else:
        logger.info("Configuration is valid. Proceeding with OBRAG CLI...")

    # move on to loading vstore, building if necessary
    indexer = Indexer(cfg, logger)
    vstore = indexer.build_vstore(force=False) # type: ignore
    logger.info("Vector store loaded successfully.")
    
    


if __name__ == "__main__":
    main()