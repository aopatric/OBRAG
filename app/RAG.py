"""
RAG agent wrapper for the LangChain logic.
"""

from langchain import hub
from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import START, StateGraph
from logging import Logger
from app.utils import State

class RAGWrapper():
    def __init__(self, vstore : Chroma, llm: BaseChatModel, logger: Logger):
        self.vstore = vstore
        self.llm = llm
        self.logger = logger
    
    def build_graph(self):
        prompt = hub.pull("rlm/rag-prompt")

        def retrieve(state: State):
            docs = self.vstore.similarity_search(state["question"])
            return {"context": docs}
        
        def generate(state: State):
            docs_content = "\n\n".join([doc.page_content for doc in state["context"]])
            messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = self.llm.invoke(messages)
            return {"answer": response.content}
        
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()
    
    def invoke(self, state):
        """
        Invoke the RAG pipeline with the given state.
        """
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not built. Please call build_graph() first.")
        
        self.logger.info(f"Invoking RAG pipeline with question: {state['question']}")
        result = self.graph.invoke(state)
        self.logger.info(f"RAG pipeline completed. Answer: {result['answer']}")
        return result