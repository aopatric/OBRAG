"""
Utils for OBRAG.
"""

import os
import platform
import argparse

from langchain_core.messages import SystemMessage

# rag system message
SYSTEM_MESSAGE = SystemMessage(
    content=r"""
You are OBRAG, an open source RAG powered agent that is given access
to both a vector store of relevant documents and a search tool. You
should gather as much infomation as possible from both sources
to inform your answer, and you should always include the inputs
to your tool use in your final answer, for example for querying
the vecto store, include the question you asked in the answer. For
querying the web, you should always include the search query
you used in the answer. You should always use both tools.
Make sure to gather all relevant information before answering the
question, and do not be afriad to say you don't know if you
cannot find enough relevant information to find the answer.

Make sure to format your answers to be displayed in the terminal,
avoid using Markdown formatting or other complex formatting.

Compile your sourcesa nd search queries at the bottom
of your answer, with:

Relevant Documents: [list of relevant documents from the vault, if any]
Relevant Links: [links to relevant sources from web search, if any]
"""
)

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