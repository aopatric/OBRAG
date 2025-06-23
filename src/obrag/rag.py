"""
Functions to perform RAG operations.
"""

from langchain_chroma import Chroma
from langchain_core.tools import tool # type: ignore
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.chat_models import init_chat_model
from typing import List, Any
from logging import Logger

from .config import Config

import os

"""
Dev notes: the interface should look like:

rag = OBRAG(vstore, cfg, logger)
OBRAG.compile() # sets up all necessary tooling
OBRAG.generate(question: str) -> dict[str, Any])
where the dict contains:
answer: str
tools_used: List[str] # list of tool names used in the answer generation, for interpretability
"""

class OBRAG:
    def __init__(self, vstore: Chroma, cfg: Config, logger: Logger) -> None:
        self.vstore = vstore
        self.cfg = cfg
        self.logger = logger
        self._setup()
    
    def _setup(self) -> None:
        self.logger.info("Setting up generation pipeline...")

        # ensure API key is set
        assert "OPENAI_API_KEY" in os.environ, "OpenAI API key is not set."

        # ensure search can be initiated
        self.search = DuckDuckGoSearchResults(output_format="json")

        # create base chat model
        llm = init_chat_model(
            self.cfg.chat_model,
            model_provider="openai"
        )

        tools = self._define_tools()
        self.tool_map = {tool.name: tool for tool in tools}
        self.llm_with_tools = llm.bind_tools(tools) # type: ignore
    
    # define the tool functions to be added to the pipeline
    def _define_tools(self) -> List[Any]:
        assert self.search is not None, "Search tool is not initialized."

        # querying vector store
        @tool
        def search_vault(query: str) -> Any:
            """Search the local Obsidian Vault for relevant documents."""
            docs = self.vstore.similarity_search(query)
            return "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant documents found in the vault."

        @tool
        def search_web(query: str) -> Any:
            """Search the web for information."""
            return self.search.invoke(query) # type: ignore
        
        return [
            search_vault,
            search_web
        ]
    
    # define generation logic
    def generate(self, query: str, message_history : List[Any]) -> Any:
        """Generate an answer to the query using the configured pipeline."""
        self.logger.info(f"Generating answer for query: {query}")
    
        messages = message_history + [HumanMessage(query)]
        
        # get the tool calls, invoke them, and regeneate with their content
        tool_call_msg : AIMessage = self.llm_with_tools.invoke(messages) # type: ignore
        self.logger.info(f"Tool calls generated: {tool_call_msg.tool_calls}")
        messages.append(tool_call_msg) # type: ignore
        
        tools_called = []

        for tool_call in tool_call_msg.tool_calls:
            assert tool_call["name"].lower() in self.tool_map, f"Tool {tool_call['name']} not found in tool map."
            tool_fn = self.tool_map.get(tool_call["name"].lower())
            response : ToolMessage = tool_fn.invoke(tool_call) # type: ignore
            tools_called.append(tool_call["name"]) # type: ignore
            messages.append(response) # type: ignore
            self.logger.info(f"Tool {tool_call['name']} invoked successfully.")
        
        # get the response and return relevent items
        generated = self.llm_with_tools.invoke(messages) # type: ignore
        messages.append(generated) # type: ignore

        return generated.content, messages, tools_called # type: ignore