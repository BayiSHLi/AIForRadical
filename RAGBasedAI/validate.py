#!/usr/bin/env python3
"""
Quick validation script to test the RAG system setup
Run this to verify everything is working correctly
"""

import sys
import os
from pathlib import Path

print("="*80)
print("RAG SYSTEM VALIDATION TEST")
print("="*80)
print()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def resolve_coded_csv() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "Fighter and sympathiser" / "coded_samples.csv",
        PROJECT_ROOT / "Fighter and sympathiser" / "coded_samples.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join([f"  - {p}" for p in candidates])
    print("❌ coded_samples.csv not found. Checked:")
    print(checked)
    sys.exit(1)

# Test 1: Check file existence
print("[Test 1] Checking required files...")
coded_csv_path = resolve_coded_csv()
required_files = [
    coded_csv_path,
    BASE_DIR / "codebook.txt",
    BASE_DIR / "build_evidence_index.py",
    BASE_DIR / "detect.py",
    BASE_DIR / "build_rule_nodes.py",
]

all_exist = True
for file in required_files:
    exists = Path(file).exists()
    status = "✓" if exists else "✗"
    display = os.path.relpath(file, start=PROJECT_ROOT)
    print(f"  {status} {display}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ Some required files are missing!")
    sys.exit(1)

print("✓ All required files present\n")

# Test 2: Check Python imports
print("[Test 2] Checking Python dependencies...")
dependencies = {
    "pandas": "Data processing",
    "llama_index.core": "LLamaIndex core",
    "llama_index.embeddings.huggingface": "HuggingFace embeddings",
    "llama_index.llms.ollama": "Ollama LLM",
}

missing_deps = []
for import_name, description in dependencies.items():
    try:
        __import__(import_name)
        print(f"  ✓ {description} ({import_name})")
    except ImportError:
        print(f"  ✗ {description} ({import_name})")
        missing_deps.append(import_name)

if missing_deps:
    print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
    print("Install them with:")
    for dep in missing_deps:
        pkg_name = dep.replace("_", "-")
        print(f"  pip install {pkg_name}")
    sys.exit(1)

print("✓ All dependencies installed\n")

# Test 3: Check index status
print("[Test 3] Checking index status...")
evidence_dir = BASE_DIR / "evidence_index"
rule_dir = BASE_DIR / "rule_index"
evidence_index_exists = evidence_dir.is_dir() and len(os.listdir(evidence_dir)) > 0
rule_index_exists = rule_dir.is_dir() and len(os.listdir(rule_dir)) > 0

if evidence_index_exists:
    print("  ✓ Evidence index found")
else:
    print("  ⚠ Evidence index not found (run: python3 build_evidence_index.py)")

if rule_index_exists:
    print("  ✓ Rule index found")
else:
    print("  ⚠ Rule index not found (run: python3 build_rule_nodes.py)")

if not (evidence_index_exists and rule_index_exists):
    print("\n⚠ Indices need to be built first")
    print("Run:")
    print("  python3 build_evidence_index.py")
    print("  python3 build_rule_nodes.py")

print()

# Test 4: Quick functionality test
if evidence_index_exists and rule_index_exists:
    print("[Test 4] Quick functionality test...")
    try:
        from llama_index.core import StorageContext, load_index_from_storage, Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        print("  Loading embedding model...")
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
        Settings.embed_model = embed_model
        
        print("  Loading indices...")
        evidence_storage = StorageContext.from_defaults(persist_dir=str(evidence_dir))
        evidence_index = load_index_from_storage(evidence_storage, embed_model=embed_model)
        print("  ✓ Evidence index loaded")
        
        rule_storage = StorageContext.from_defaults(persist_dir=str(rule_dir))
        rule_index = load_index_from_storage(rule_storage, embed_model=embed_model)
        print("  ✓ Rule index loaded")
        
        print("✓ Indices loaded successfully\n")
    except Exception as e:
        print(f"\n❌ Error loading indices: {e}\n")

print("="*80)
print("VALIDATION COMPLETE")
print("="*80)
print()

if evidence_index_exists and rule_index_exists and not missing_deps:
    print("✓ System is ready for detection!")
    print("\nTo run detection:")
    print("  python3 detect.py")
    print()
else:
    print("⚠ Some setup steps needed:")
    if missing_deps:
        print("  1. Install missing dependencies")
    if not evidence_index_exists:
        print("  2. Run: python3 build_evidence_index.py")
    if not rule_index_exists:
        print("  3. Run: python3 build_rule_nodes.py")
    print()
