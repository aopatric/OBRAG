"""
Entry point for the OBRAG app.
"""
import os
import getpass
import json

from langchain.chat_models import init_chat_model

from app.logging import start_logger
from app.DEFAULTS import *
from app.dataloader import DataLoader
from app.RAG import RAGWrapper

def main():
    """
    Main CLI loop logic for OBRAG CLI.
    """
    logger = start_logger()
    logger.info("OBRAG CLI started.")

    # welcome ascii and version info
    print(DEFAULT_WELCOME_BANNER)
    print(f"Version {VERSION_NUMBER}")
    logger.info(f"Displaying welcome banner and version info for version {VERSION_NUMBER}...")

    # if a config file doesnt exist, prompt for basic info and make it
    if not os.path.exists("config.json"):
        logger.info("First run detected. Prompting for vault path and embed model...")
        print("\n\n")
        print("Welcome to OBRAG! This is your first run, so please provide the following information:\n")

        # get the vault path
        try:
            while True:
                vault_path = input("Enter the path to your Obsidian vault (leave blank for default): ").strip()

                if not vault_path:
                    vault_path = DEFAULT_VAULT_PATH
                    logger.info(f"No vault path provided, using default: {vault_path}")
                    break
                
                elif os.path.exists(vault_path):
                    logger.info(f"Vault path provided: {vault_path}")
                    break
                
                else:
                    print(f"Path '{vault_path}' does not exist. Please try again.")
                    logger.warning(f"Invalid vault path provided: {vault_path}")

        except Exception as e:
            logger.error(f"Error gathering vault path: {e}")
            print(f"Error: {e}")
            vault_path = DEFAULT_VAULT_PATH
        
        print("\n\n")
        print(f"Thank you. Using vault path: {vault_path}")

        # get the name of the embedding model
        print("\n\n")
        try:
            while True:
                embedding_modelname = input("Enter the name of the HuggingFace embedding model (leave blank for default): ").strip()

                if not embedding_modelname:
                    embedding_modelname = DEFAULT_EMBEDDING_MODELNAME
                    logger.info(f"No embedding model name provided, using default: {embedding_modelname}")
                    break
                
                else:
                    logger.info(f"Embedding model name provided: {embedding_modelname}")
                    break
        except Exception as e:
            logger.error(f"Error gathering embedding model name: {e}")
            print(f"Error: {e}")
            embedding_modelname = DEFAULT_EMBEDDING_MODELNAME
        print("\n\n")
        print(f"Thank you. Using embedding model: {embedding_modelname}")

        # get the name of the chat model
        print("\n\n")
        try:
            while True:
                chat_modelname = input("Enter the name of the desired chat model (leave blank for default, see README for supported models): ").strip()

                if not chat_modelname:
                    chat_modelname = DEFAULT_CHAT_MODELNAME  # Using embedding model as default for simplicity
                    logger.info(f"No chat model name provided, using default: {chat_modelname}")
                    break
                
                else:
                    logger.info(f"Chat model name provided: {chat_modelname}")
                    break
        except Exception as e:
            logger.error(f"Error gathering chat model name: {e}")
            print(f"Error: {e}")
            chat_modelname = DEFAULT_CHAT_MODELNAME
        print("\n\n")
        print(f"Thank you. Using chat model: {chat_modelname}")

        # get openai API key for now
        print("\n\n")
        try:
            if not os.environ.get("OPENAI_API_KEY"):
                api_key = getpass.getpass("Enter your OpenAI API key: ").strip()
                os.environ["OPENAI_API_KEY"] = api_key
                logger.info("OpenAI API key set from user input.")
                print("OpenAI API key has been set. It will be used for the chat model.")
            else:
                api_key = os.environ["OPENAI_API_KEY"]
                logger.info("OpenAI API key already set in environment variables.")
                print("Using existing OpenAI API key from environment variables.")
        except Exception as e:
            logger.error(f"Error gathering OpenAI API key: {e}")
            print(f"Error: {e}")
            api_key = ""

        # create config based on user input
        config = {
            "vault_path": vault_path,
            "embedding_modelname": embedding_modelname,
            "chat_modelname": chat_modelname,
            "openai_api_key": api_key
        }

        with open("config.json", "w") as config_file:
            json.dump(config, config_file, indent=4)
        logger.info("Configuration file created with user input, saved to config.json")
    else:
        logger.info("Configuration file found, loading existing settings...")
        print(f"Loading existing configuration from config.json...")
        with open("config.json", "r") as config_file:
            config = json.load(config_file)
            vault_path = config.get("vault_path", DEFAULT_VAULT_PATH)
            embedding_modelname = config.get("embedding_modelname", DEFAULT_EMBEDDING_MODELNAME)
            chat_modelname = config.get("chat_modelname", DEFAULT_CHAT_MODELNAME)
            logger.info(f"Loaded vault path: {vault_path} and embedding model: {embedding_modelname} and chat model: {chat_modelname}")
    
    print(f"\n\nStarting OBRAG...")

    print(f"\nLoading vault files...")
    dataloader = DataLoader(
        vault_path=vault_path,
        embedding_modelname=embedding_modelname,
        logger=logger
    )

    try:
        logger.info("Gathering chunks from Obsidian vault...")
        chunks = dataloader.gather_chunks()
        logger.info(f"Loaded {len(chunks)} chunks from the vault.")
        
        logger.info("Building vector store...")
        vstore = dataloader.get_vstore()
        logger.info("Vector store built successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred during data loading: {e}")
        print(f"Error: {e}")
        exit(1)

    print("\nInitializing LLM and RAG pipeline...")
    # assert the api key is set
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = config.get("openai_api_key", "")
    llm = init_chat_model(
        chat_modelname,
        model_provider="openai"
    )

    rag = RAGWrapper(
        vstore=vstore,
        llm=llm,
        logger=logger
    )

    rag.build_graph()
    print("RAG pipeline initialized successfully.")


    # start chat loop
    print("\n\nOBRAG is ready! You can now start asking questions.")

    while True:
        try:
            prompt = input("You ('exit' to quit): ").strip()
            if prompt.lower() == "exit":
                print("Exiting OBRAG. Goodbye!")
                logger.info("User exited the OBRAG CLI.")
                exit(0)
            else:
                logger.info(f"User prompt received: {prompt}")
                state = {"question": prompt}
                result = rag.invoke(state)
                print(f"OBRAG: {result['answer']}")
                logger.info(f"OBRAG response: {result['answer']}")
        except Exception as e:
            logger.error(f"Error reading input: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()