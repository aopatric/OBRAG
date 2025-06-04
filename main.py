import argparse
import os
import rag_utils
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# --- Existing RAG Prompt (for knowledge base queries) ---
rag_prompt_template_str = """You are a helpful assistant with access to a personal knowledge base.
Your primary goal is to answer the user's question concisely based on the provided context snippets.

Follow these instructions carefully:
1.  Analyze the user's QUESTION.
2.  Review the provided CONTEXT snippets.
3.  If the CONTEXT directly and clearly answers the QUESTION, formulate a concise answer using ONLY the information from the CONTEXT.
4.  If the CONTEXT seems irrelevant to the QUESTION, or if the QUESTION is a matter of general common knowledge that is not contradicted by the CONTEXT, you may answer from your general knowledge. However, clearly state if you are using general knowledge.
5.  If the CONTEXT does not provide an answer and it's not a common knowledge question, state "I don't have specific information on that in my notes."
6.  CRITICAL: DO NOT simply rephrase or regurgitate the CONTEXT snippets. Synthesize the information.
7.  Your answer should be direct and to the point. Avoid verbosity.
8.  DO NOT repeat the question in your answer.
9.  DO NOT make up information or hallucinate.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

KNOWLEDGE_BASE_QA_PROMPT = PromptTemplate(
    template=rag_prompt_template_str,
    input_variables=["context", "question"]
)

# --- New Prompt for Query Classification (Router) ---
ROUTER_PROMPT_TEMPLATE_STR = """Classify the user's query. Your response should be ONLY one of the following two options:
- 'knowledge_base_query' (if the query likely requires information from a personal knowledge base, notes, or specific documents)
- 'general_query' (if the query is a greeting, a common sense question, a request for general information not specific to personal notes, or a simple conversational turn)

User Query: "{query}"
Classification:"""

ROUTER_PROMPT = PromptTemplate(
    template=ROUTER_PROMPT_TEMPLATE_STR,
    input_variables=["query"]
)

# --- New Prompt for General Queries (Direct LLM) ---
GENERAL_QUERY_PROMPT_TEMPLATE_STR = """You are a helpful conversational assistant.
User: {query}
Assistant:"""

GENERAL_QUERY_PROMPT = PromptTemplate(
    template=GENERAL_QUERY_PROMPT_TEMPLATE_STR,
    input_variables=["query"]
)

def refresh_db(config_path: str="config.ini", profile: str="DEFAULT"):
    print("Refreshing database...")
    try:
        conf = rag_utils.read_config(config_path, profile)
        # Expand user paths
        vault_path = os.path.expanduser(conf["vault_path"])
        persist_dir = os.path.expanduser(conf["persist_dir"])

        mds = rag_utils.get_markdown_documents(vault_path)
        chunks = rag_utils.chunk_documents(mds, conf["chunk_size"], conf["overlap"])
        model = rag_utils.load_embedding_model(conf["embedding_model"], conf["normalize_embeddings"])
        db = rag_utils.build_db(chunks, model, persist_dir, conf["collection_name"])
        print("Created new persistent vector store with {} items.".format(db._collection.count()))
    except Exception as e:
        print(f"Error refreshing database: {e}")
        raise

def start_llm(model_name: str):
    print("Starting LLM with name {}...".format(model_name))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if 'cuda' in device else torch.float32
            )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        print("LLM started successfully.")
        return llm
    except Exception as e:
        print(f"Error starting LLM: {e}")
        raise

def do_chat_loop(config_path: str="config.ini", profile: str="DEFAULT"):
    try:
        conf = rag_utils.read_config(config_path, profile)
        # Expand user paths
        persist_dir = os.path.expanduser(conf["persist_dir"])

        embeddings = rag_utils.load_embedding_model(conf["embedding_model"], conf["normalize_embeddings"])
        db = rag_utils.load_db(persist_dir, embeddings, conf["collection_name"])
        print(f"Loaded {db._collection.count()} documents from the database.")
        
        llm = start_llm(conf["language_model"])
        if not llm:
            print("LLM failed to initialize. Exiting.")
            return

        # --- Setup Chains ---
        # 1. Router Chain (for classifying the query)
        router_chain = LLMChain(llm=llm, prompt=ROUTER_PROMPT, output_key="classification")

        # 2. RAG Chain (for knowledge base queries)
        knowledge_base_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": conf.get("top_k",3)}, search_type="similarity"),
            return_source_documents=True,
            chain_type_kwargs={"prompt": KNOWLEDGE_BASE_QA_PROMPT}
        )

        # 3. General Query Chain (for direct LLM answers)
        general_query_chain = LLMChain(llm=llm, prompt=GENERAL_QUERY_PROMPT, output_key="answer")

        print("\nReady to chat! Type 'exit' or 'quit' to close.")
        agent_name = conf.get('name', 'Ollin')
        while True:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            if not query:
                continue

            try:
                # Step 1: Route the query
                print(f"{agent_name} is thinking (routing)...")
                routing_result = router_chain.invoke({"query": query})
                # The classification should be in routing_result["classification"]
                # The actual text might be in routing_result[router_chain.output_key]
                # Let's assume the LLM outputs the class name cleanly.
                # We might need to parse router_result['text'] if output_key doesn't work as expected.
                
                # Try to get classification from the expected output key, fallback to 'text'
                classification_text = routing_result.get(router_chain.output_key, routing_result.get("text", "")).strip().lower()
                print(f"Query classified as: '{classification_text}'")


                if "knowledge_base_query" in classification_text:
                    print(f"{agent_name} is searching notes...")
                    result = knowledge_base_qa_chain.invoke({"query": query})
                    answer = result['result'].strip()
                    print(f"\n{agent_name}: {answer}")
                    if result.get("source_documents"):
                        print("\nSources (titles only):")
                        sources = set()
                        for doc in result["source_documents"]:
                            source_name = doc.metadata.get("source", "Unknown")
                            sources.add(source_name)
                        for i, src in enumerate(sources, 1):
                            print(f"  {i}. {src}")
                elif "general_query" in classification_text:
                    print(f"{agent_name} is answering directly...")
                    result = general_query_chain.invoke({"query": query})
                    answer = result.get(general_query_chain.output_key, result.get("text", "Sorry, I had trouble with that.")).strip()
                    print(f"\n{agent_name}: {answer}")
                else:
                    print(f"\n{agent_name}: I'm not sure how to classify that query (received: '{classification_text}'). Could you rephrase or be more specific?")
                
                print("\n" + "*" * 50)

            except Exception as e:
                print(f"Error processing query: {e}")
                continue
        print("\nGoodbye!")

    except FileNotFoundError as e:
        print(f"Error in chat loop: {e}. Ensure config and persist directory are correct.")
        print(f"If the vector store doesn't exist at '{os.path.expanduser(conf.get('persist_dir', 'MISSING_PERSIST_DIR_IN_CONFIG'))}', run --refresh.")
    except KeyError as e:
        print(f"Error in chat loop: Missing configuration key {e}. Please check '{config_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred in chat loop: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG pipeline.")
    parser.add_argument("--refresh", action="store_true", help="Refresh the database.")
    parser.add_argument("--config", default="config.ini", help="Path to the configuration file.")
    parser.add_argument("--profile", default="DEFAULT", help="Profile to use from the configuration file.")

    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
    elif args.refresh:
        refresh_db(config_path=args.config, profile=args.profile)
    else:
        do_chat_loop(config_path=args.config, profile=args.profile)