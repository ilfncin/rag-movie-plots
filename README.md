# **RAG Movie Plots**

**RAG Movie Plots** is a modular, research-oriented Retrieval-Augmented Generation (RAG) project built on top of the [Wikipedia Movie Plots dataset](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots). 

The project focuses on **architectural clarity**, **empirical design decisions**, and **observability**, rather than providing a black-box RAG demo. Each stage of the pipeline is explicitly modeled, documented, and validated through exploratory analysis and controlled experiments.

It demonstrates an end-to-end RAG pipeline - from ingestion to evaluation - using **LangChain**, **ChromaDB** and **RAGAS**. It retrieves relevant movie descriptions and answers user questions using a language model (LLM) such as OpenAI’s `gpt-4o-mini`.

---

## **Project Goals**

- Design a **clean, inspectable, end-to-end RAG pipeline**
- Separate **offline ingestion** from **online retrieval and generation**
- Ground architectural decisions (e.g. chunking) in data-driven analysis
- Enable transparent debugging and evaluation of retrieval behavior
- Support controlled experimentation across chunking, retrieval, prompt, and model choices

---

## **High-Level RAG Architecture**

This project follows a **two-phase Retrieval-Augmented Generation (RAG) architecture** designed to enforce a clean separation between **offline data preparation** and **online retrieval and generation**. The pipeline is decomposed into **five modular components**, distributed across these two phases.

## **Phase 1 - Offline Ingestion Pipeline**

Phase 1 prepares all data required for retrieval and runs entirely offline. It is responsible for transforming raw tabular data into a searchable vector representation.

### **Modules in Phase 1**

1. **ETL - Data Cleaning and JSONL Generation**
    - Loads the raw CSV dataset
    - Cleans, normalizes, and standardizes individual columns
    - Generates structured documents in `docs.jsonl`

   > The ETL layer intentionally focuses on cleaning and standardizing values within individual columns, such as removing uninformative entries and normalizing text formats, without engaging in more complex structural corrections that depend on relationships across multiple fields. These more intricate transformations, such as cross-column consistency checks or semantic deduplication, are intentionally deferred to future iterations, where context-aware strategies can be applied more effectively.

2. **Chunking - Strategy Exploration and Text Segmentation**
    - Splits long movie plots into smaller, overlapping text chunks
    - Applies configurable chunking strategies
    - Produces `chunks.jsonl`

    > Chunking parameters (chunk size, overlap, separator hierarchy) are grounded in a dedicated exploratory analysis of text structure, rather than heuristic defaults.

3. **Vector Store - Embedding and Persistence**
    - Generates dense embeddings for all chunks
    - Builds and persists a ChromaDB vector store at:
     ```bash
     db/chroma/
     ```

     > The resulting vector store represents the final output of the offline ingestion phase and serves as the sole knowledge source for online retrieval.

## **Phase 2 - Online RAG Workflow**

Phase 2 handles the **online querying flow**, combining semantic retrieval with controlled language generation to answer user questions.

### **Modules in Phase 2**

4. **Retrieval - Semantic Search and Context Selection**
    - Loads the persisted vector store
    - Encodes user queries using the same embedding model as ingestion
    - Executes semantic similarity search over embedded chunks
    - Optionally applies distance-based filtering
    - Selects and assembles a ranked contextual set for generation

    > Retrieval behavior is explicitly observable through structured logs, enabling inspection and debugging before any generation occurs.

5. **Generation - Prompt Construction and Answer Synthesis**
    - Constructs structured RAG prompts from the retrieved context
    - Applies strict prompt-level constraints to prevent hallucinations
    - Generates final answers via the selected large language model (LLM)

    > Both the generated answer and the exact context used are exposed, ensuring traceability and reproducibility.


Each module is implemented independently and communicates only through **well-defined data artifacts** (e.g., JSONL files, vector stores, retrieved context). 

This design enables controlled experimentation across chunking strategies, retrieval configurations, prompt versions, and model choices, **without entangling concerns across pipeline layers**.

---

## **Project Structure**

```
rag-movie-plots/
├── data/
│   ├── raw/                        # Original unmodified datasets
│   └── processed/                  # Versioned ETL + chunking outputs (JSONL)
│
├── db/
│   └── chroma/                     # Persisted ChromaDB vector store (SQLite + index)
│
├── logs/                           # Timestamped application logs
│
├── notebooks/                      # Jupyter notebooks for EDA, pipeline tests and RAG evaluation
│   ├── 1.0-ilfn-initial-data-exploration.ipynb
│   ├── 1.1-ilfn-chunking-strategy-exploration.ipynb
│   ├── 2.0-ilfn-rag-ingestion-pipeline.ipynb
│   ├── 3.0-ilfn-rag-online-query.ipynb
│   └── notebook_setup.py          
│
├── src/
│   └── backend/
│       ├── config/                 # Centralized configuration (env-driven)
│       │   └── settings.py
│       │
│       ├── infra/                  # Infrastructure-level abstractions
│       │   └── llm_client.py
│       │
│       ├── pipelines/              # Offline orchestration pipelines
│       │   ├── etl/                # Data cleaning and JSONL generation pipeline
│       │   ├── chunking/           # Chunking pipeline (strategy selection + execution)
│       │   └── vectorstore/        # Embedding generation and vector store persistence
│       │
│       ├── runtime/                # Online / query-time execution
│       │   ├── retrieval/          # Semantic retriever
│       │   ├── chat/               # ChatRAG orchestration
│       │   └── prompts/            # Versioned RAG prompt templates
│       │
│       ├── utils/                  # Logging, helpers, shared utilities
│       │
│       └── main.py                 # Placeholder entry point (future frontend/API integration)
│
├── .env.template                           # Template for generating new env files safely
├── pyproject.toml                          # Project metadata, dependencies, and build configuration
└── uv.lock                                 # Locked dependency versions for reproducible environments


```

## **Notebooks Overview**

**1.0 - Initial Data Exploration**

Exploratory analysis of the raw dataset:
- distribution of plot lengths
- structural inconsistencies
- missing and noisy fields

**1.1 - Chunking Strategy Exploration**

A focused, data-driven study of:
- characters per plot
- number of lines and paragraphs
- maximum line length

This notebook directly informs:
- chunk size
- overlap
- choice of recursive character-based chunking

**2.0 - Ingestion Pipeline**

End-to-end offline ingestion:
- ETL → docs.jsonl
- Chunking → chunks.jsonl
- Embedding + ChromaDB persistence

**3.0 - Online RAG Queries**

Query-time experiments:
- retriever-only inspection
- RAG vs LLM-only comparison
- context inspection and debugging


## **Environment Setup**

### **Install [uv](https://github.com/astral-sh/uv)**
```bash
pipx install uv
```

### **Sync Environment**
```bash
uv sync
```

### **Configure environment variables**
 ```bash
cp .env.template .env
```

Update the variables according to your local setup.

---


##  **Note - attempt to write a readonly database error**

If you encounter the following error while running the `2.0-ilfn-rag-ingestion-pipeline.ipynb` notebook:

```bash
InternalError: Query error: Database error: (code: 1032) attempt to write a readonly database
```

Simply **restart the Jupyter kernel** and run the cell again. This clears the active `ChromaDB` connection and releases the SQLite lock on `db/chroma`.

---


## **Status**

This repository is under active development. At this stage of the project:
- All pipelines are executed directly from notebooks
- `main.py` is not the primary entry point
- The notebooks act as:
    - executable documentation,
    - experiment runners,
    - and validation tools

> The `main.py` file is reserved for a future phase, where it will serve as the integration point for a frontend or API layer.

