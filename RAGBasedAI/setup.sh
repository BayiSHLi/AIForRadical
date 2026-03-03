#!/bin/bash

# RAG-Based Detection System - Quick Start Script
# This script sets up and runs the radicalisation detection system

set -e

echo "=========================================="
echo "RAG-Based Radicalisation Detection Setup"
echo "=========================================="
echo ""

CD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CD_DIR"

# Check Python
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi
echo "✓ Python found: $(python3 --version)"
echo ""

# Check required files
echo "[2/5] Checking required files..."
if [ ! -f "codebook.txt" ]; then
    echo "❌ codebook.txt not found"
    exit 1
fi
if [ ! -f "../Fighter and sympathiser/coded_samples.csv" ]; then
    echo "❌ coded_samples.csv not found"
    exit 1
fi
echo "✓ All required files present"
echo ""

# Build indices
echo "[3/5] Building indices..."
echo "  - Building evidence index from CSV..."
python3 build_evidence_index.py || {
    echo "❌ Failed to build evidence index"
    exit 1
}
echo ""

echo "  - Building rule index from codebook..."
python3 build_rule_nodes.py || {
    echo "❌ Failed to build rule index"
    exit 1
}
echo ""

# Check Ollama
echo "[4/5] Checking Ollama setup..."
if ! command -v ollama &> /dev/null; then
    echo "⚠ Ollama not found. Installing guide:"
    echo "  1. Download from https://ollama.ai/"
    echo "  2. Install and run: ollama serve"
    echo "  3. Pull model: ollama pull qwen2.5:7b"
    echo ""
fi
echo ""

# Ready to use
echo "[5/5] Setup complete!"
echo ""
echo "=========================================="
echo "✓ System Ready for Detection"
echo "=========================================="
echo ""
echo "To run detection:"
echo "  python3 detect.py"
echo ""
echo "To use in your code:"
echo "  from detect import detect_radicalisation"
echo "  result = detect_radicalisation('Your post text here')"
echo ""
