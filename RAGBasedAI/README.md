# RAG-Based Radicalisation Detection System

## Overview

This system uses Retrieval-Augmented Generation (RAG) with LLamaIndex to detect radicalisation indicators in social media posts.

## Key Improvements

### ✅ What Was Fixed

1. **Removed FAISS Dependency**
   - FAISS was causing `ModuleNotFoundError` and complex installation issues
   - Replaced with LLamaIndex's default vector store (in-memory + persistence)
   - Simpler, more reliable, and easier to deploy

2. **Proper Index Persistence**
   - Old code was trying to manually manage FAISS indices
   - New code uses LLamaIndex's built-in StorageContext
   - Automatic serialization and deserialization

3. **Better Error Handling**
   - Added comprehensive error checks and user-friendly messages
   - Optional rule index (graceful degradation if not available)
   - Clear logging of what's happening at each step

4. **Consistent Data Flow**
   - Evidence and Rule indices use the same architecture
   - Metadata properly stored and preserved
   - Document relationships maintained through the pipeline

## File Structure

```
RAGBasedAI/
├── build_evidence_index.py     # Build index from CSV data
├── build_rule_nodes.py         # Build index from codebook
├── detect.py                   # Detection inference
├── codebook.txt                # Codebook with indicators
├── evidence_index/             # Persisted evidence index
│   ├── docstore.json
│   ├── vector_store.json
│   ├── metadata.json
│   └── ...
└── rule_index/                 # Persisted rule index
    ├── docstore.json
    ├── vector_store.json
    └── ...
```

## Usage

### Step 1: Build Evidence Index

```bash
cd RAGBasedAI
python build_evidence_index.py
```

This will:
- Read CSV data from `data/Fighter and sympathiser/coded_samples.csv` (or legacy `Fighter and sympathiser/coded_samples.csv`)
- Create documents with content and metadata
- Build embeddings using BAAI/bge-m3
- Persist the index to `./evidence_index/`

### Step 2: Build Rule Index

```bash
python build_rule_nodes.py
```

This will:
- Read indicators from `codebook.txt`
- Create structured indicator documents
- Build embeddings for rules
- Persist the index to `./rule_index/`

### Step 3: Run Detection

```bash
python detect.py
```

This will:
- Load both indices
- Query an example post
- Return detection results in JSON format

### Using in Your Code

```python
from detect import detect_radicalisation

# Analyze a post
result = detect_radicalisation(
    post_text="Your post content here",
    image_description="Description of any images"
)

print(result)
```

## Architecture

### Evidence Index
- **Source**: CSV data from actual posts (coded=1)
- **Documents**: Post content + image descriptions
- **Purpose**: Provide context and examples for LLM
- **Query**: Returns similar posts for comparison

### Rule Index  
- **Source**: Codebook with carefully crafted indicators
- **Documents**: Indicator definitions with examples
- **Purpose**: Guide LLM with specific detection rules
- **Query**: Returns relevant indicators for the post

### Detection Pipeline

```
Post Text + Image
     ↓
[Evidence Retrieval]  → Find 15 similar posts (reranked to top 5)
     ↓
[Rule Retrieval]      → Find 5 relevant indicators
     ↓
[LLM Analysis]        → Query Qwen model with context
     ↓
JSON Output           → Indicators, reasoning, confidence
```

## Configuration

### Embedding Model
- **Current**: `BAAI/bge-m3` (multilingual)
- **Location**: Used in both build and detect scripts
- **Quality**: High-quality embeddings for semantic search

### LLM Model
- **Current**: `Ollama qwen2.5:7b` (local)
- **Settings**: Temperature=0.3 (more deterministic)
- **Requirement**: Ollama must be running locally

### Reranker
- **Model**: `BAAI/bge-reranker-large`
- **Purpose**: Re-rank evidence to get most relevant cases
- **Top K**: Returns top 5 after reranking

## Troubleshooting

### Error: "Index files not found"
```
Run: python build_evidence_index.py
Then: python build_rule_nodes.py
```

### Error: "Ollama connection refused"
```
Make sure Ollama is running:
ollama serve

Or check if qwen2.5:7b is pulled:
ollama pull qwen2.5:7b
```

### Error: "Module not found (embeddings/llms)"
```
Install dependencies:
pip install llama-index-embeddings-huggingface
pip install llama-index-llms-ollama
pip install llama-index-postprocessors-flag-embedding-reranker
```

## Performance Notes

- **First build**: Takes ~5-10 minutes depending on dataset size
- **Subsequent queries**: ~10-30 seconds (embedding + LLM)
- **Memory usage**: ~2-4 GB for full pipeline
- **Storage**: Index files typically 500MB-1GB

## Future Improvements

1. Add streaming output for LLM responses
2. Implement batch processing
3. Add confidence scoring from similarity metrics
4. Support for multiple languages
5. Fine-tuned models for radicalisation detection
6. User feedback loop for continuous improvement

## References

- LLamaIndex Documentation: https://docs.llamaindex.ai/
- BAAI Models: https://huggingface.co/BAAI
- Ollama: https://ollama.ai/

