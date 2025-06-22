"""
Utils for OBRAG.
"""

import os
import platform
import argparse

ASCII_BANNER = """
      ___                         ___           ___           ___     
     /\\  \\         _____         /\\  \\         /\\  \\         /\\__\\    
    /::\\  \\       /::\\  \\       /::\\  \\       /::\\  \\       /:/ _/_   
   /:/\\:\\  \\     /:/\\:\\  \\     /:/\\:\\__\\     /:/\\:\\  \\     /:/ /\\  \\  
  /:/  \\:\\  \\   /:/ /::\\__\\   /:/ /:/  /    /:/ /::\\  \\   /:/ /::\\  \\ 
 /:/__/ \\:\\__\\ /:/_/:/\\:|__| /:/_/:/__/___ /:/_/:/\\:\\__\\ /:/__\\/\\:\\__\\
 \\:\\  \\ /:/  / \\:\\/:/ /:/  / \\:\\/:::::/  / \\:\\/:/  \\/__/ \\:\\  \\ /:/  /
  \\:\\  /:/  /   \\::/_/:/  /   \\::/~~/~~~~   \\::/__/       \\:\\  /:/  / 
   \\:\\/:/  /     \\:\\/:/  /     \\:\\~~\\        \\:\\  \\        \\:\\/:/  /  
    \\::/  /       \\::/  /       \\:\\__\\        \\:\\__\\        \\::/  /   
     \\/__/         \\/__/         \\/__/         \\/__/         \\/__/    
"""

VERSION_NUMBER = "0.1.0"


def print_banner() -> None:
    """Print the ASCII banner and version number."""
    print(ASCII_BANNER)
    print(f"Version {VERSION_NUMBER}")

def clear_console() -> None:
    """Clear the console based on the operating system."""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OBRAG CLI")
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Reconfigure OBRAG settings and rebuild the vector store.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector store from the Obsidian vault.",
    )
    parser.add_argument(
        "--ask",
        type=str,
        help="Ask a question directly from the command line, shortcut for CLI."
    )
    return parser.parse_args()

def print_help() -> None:
    """Print the help message."""
    print("OBRAG CLI")
    print("Usage: obrag [options]")
    print("Options:")
    print("  --reconfigure    Reconfigure OBRAG settings and rebuild the vector store.")
    print("  --rebuild        Rebuild the vector store from the Obsidian vault.")
    print("  --ask            Ask a question directly from the command line, shortcut for CLI.")
    print("  --help           Print this help message.")