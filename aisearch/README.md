# GNAIS

## Description

**GNAIS** (GeneNetwork AI Search) is a python package that help digest metadata around GeneNetwork using language models. It allows running natural language queries against RDF data (metadata) converted to text and preprocessed locally.

**GNAIS** performs a hybrid search (keyword and semantic) through a RAG (Retrieval Augmented Generation) system. The embedding model for semantic is Qwen/Qwen3-Embedding-0.6B (open model).

We implemented **GNAIS** using [DSPy](https://dspy.ai/). Switching between LLM providers for the text generation model is as easy as changing a variable :)

## Installation

**GNAIS** is in PyPI. You can install it in your virtual environment using the following commands:

```python
python -m venv .venv
source .venv/bin/activate
pip install gnais
```

## Usage

To use **GNAIS**, you need to define a few variables in your session or script.

```python
CORPUS_PATH=<YOUR_PATH>
PCORPUS_PATH=<YOUR_PATH>
DB_PATH=<YOUR_PATH>
SEED=<YOUR_VALUE>
MODEL_NAME=<DSPY_COMPLIANT_MODEL_NAME>
API_KEY=<YOUR_API_KEY_IF_REQUIRED>
QUERY=<YOUR_QUERY>
```


Once defined, you can run your search with:

```python
from gnais.search import search
search(QUERY)
```

