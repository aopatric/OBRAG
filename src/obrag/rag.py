"""
Functions to perform RAG operations.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain import hub
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from typing import TypedDict, List, Any
from logging import Logger

from .config import Config

import os


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RAGPipeline:
    def __init__(self, vstore: Chroma, cfg: Config, logger: Logger) -> None:
        self.vstore = vstore
        self.cfg = cfg
        self.logger = logger
        self.graph = None
        self.prompt = hub.pull("rlm/rag-prompt")
    
    def _start_llm(self) -> None:
        self.logger.info("Initializing LLM...")

        # assert the api key is set
        if not os.environ.get("OPENAI_API_KEY"):
            self.logger.error("OPENAI_API_KEY environment variable is not set.")
            os.environ["OPENAI_API_KEY"] = self.cfg.openai_api_key
        
        self.llm = init_chat_model(
            self.cfg.chat_model,
            model_provider="openai" # fixed for now
        )
        self.logger.info("LLM initialized successfully.")
    
    def compile(self) -> None:
        self.logger.info("Compiling RAG pipeline...")
        self._start_llm()
        # define action functions
        def retrieve(state: State):
            docs = self.vstore.similarity_search(state["question"], self.cfg.top_k)
            return {"context": docs}
        
        # generation helper
        def generate(state: State): # type: ignore
            content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = self.prompt.invoke({"question": state["question"], "context": content})
            response = self.llm.invoke(messages)
            return {"answer": response.content} # type: ignore
        
        # define and compile the state graph
        graph_builder = StateGraph(State).add_sequence([retrieve, generate]) # type: ignore
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile() # type: ignore
        self.logger.info("RAG pipeline compiled successfully.")
    
    def generate(self, question: str) -> dict[str, Any]:
        assert self.graph is not None, "Graph must be compiled before generating answers."

        self.logger.info(f"Generating answer for question: {question}")
        answer = self.graph.invoke({"question": question})
        self.logger.info(f"Answer generated successfully.")
        return {
            "context": answer["context"], # type: ignore
            "answer": answer["answer"] # type: ignore
        }