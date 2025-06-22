# OBRAG
Obsidian-linked RAG Agent.

Installation:

Requires Python >= 3.11:

```python
pip install obrag
obrag --help
```

If there are any issues, reconfiguring with `obrag --reconfigure` should solve problems.

Simple Obsidian-linked RAG CLI for querying note base; API inference and local embeddings with HuggingFace.

To rebuild the vector store, use `obrag --rebuild`.

To skip the CLI and ask a question directly from the terminal, use `obrag --ask QUESTION`.

Happy querying.