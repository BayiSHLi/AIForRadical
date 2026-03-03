import pandas as pd
import os
import json
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ===== Configuration =====
OUTPUT_DIR = "./evidence_index"
CSV_PATH = "/home/user/workspace/SHLi/AI for radicalisation/Fighter and sympathiser/coded_samples.csv"

# ===== Setup =====
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Embedding Model =====
print("Loading embedding model (BAAI/bge-m3)...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.embed_model = embed_model

# ===== Read CSV and Create Documents =====
print(f"Reading CSV from {CSV_PATH}...")
try:
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Loaded {len(df)} records from CSV")
except FileNotFoundError:
    print(f"❌ Error: CSV file not found at {CSV_PATH}")
    exit(1)
except Exception as e:
    print(f"❌ Error reading CSV: {e}")
    exit(1)

# ===== Create Documents =====
print("Creating documents from CSV records...")
documents = []
metadata_list = []

for idx, row in df.iterrows():
    try:
        # Build document text
        content = str(row.get('content', '')).strip()
        image_desc = str(row.get('image_description', '')).strip()
        
        text = f"""POST CONTENT:
{content}

IMAGE DESCRIPTION:
{image_desc}"""
        
        # Store metadata
        metadata = {
            'id': idx,
            'category': str(row.get('category', 'unknown')),
            'person': str(row.get('person', 'unknown')),
            'handle': str(row.get('handle', 'unknown')),
            'date': str(row.get('date', 'unknown')),
            'coded': int(row.get('coded', 0)),
        }
        
        # Create document with metadata
        doc = Document(
            text=text,
            metadata=metadata,
            doc_id=f"doc_{idx}"
        )
        
        documents.append(doc)
        metadata_list.append(metadata)
        
    except Exception as e:
        print(f"⚠ Warning: Error processing row {idx}: {e}")
        continue

print(f"✓ Created {len(documents)} documents")

# ===== Create Nodes =====
print("Parsing documents into nodes...")
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)
print(f"✓ Created {len(nodes)} nodes from documents")

# ===== Build Vector Store Index =====
print("Building vector store index (this may take a while)...")
try:
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    print("✓ Vector store index built successfully")
except Exception as e:
    print(f"❌ Error building index: {e}")
    exit(1)

# ===== Persist Index =====
print(f"Persisting index to {OUTPUT_DIR}...")
try:
    index.storage_context.persist(OUTPUT_DIR)
    print(f"✓ Index persisted to {OUTPUT_DIR}")
except Exception as e:
    print(f"❌ Error persisting index: {e}")
    exit(1)

# ===== Save Metadata as JSON =====
metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
try:
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")
except Exception as e:
    print(f"⚠ Warning: Could not save metadata: {e}")

# ===== Summary =====
print("\n" + "="*80)
print("EVIDENCE INDEX BUILD COMPLETE")
print("="*80)
print(f"Total documents: {len(documents)}")
print(f"Total nodes: {len(nodes)}")
print(f"Index directory: {OUTPUT_DIR}")
print(f"Files created:")
for file in os.listdir(OUTPUT_DIR):
    file_path = os.path.join(OUTPUT_DIR, file)
    file_size = os.path.getsize(file_path) / (1024*1024)  # Convert to MB
    print(f"  - {file} ({file_size:.2f} MB)")
print("="*80)
