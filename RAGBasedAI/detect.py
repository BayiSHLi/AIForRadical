import os
import json
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

# ===== Configuration =====
BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_INDEX_PATH = BASE_DIR / "evidence_index"
RULE_INDEX_PATH = BASE_DIR / "rule_index"
INDICATOR_MAPPING_PATH = BASE_DIR / "indicator_mapping.json"

# ===== Load Indicator Mapping =====
print("Loading indicator mapping...")
try:
    with open(INDICATOR_MAPPING_PATH, "r", encoding="utf-8") as f:
        indicator_mapping = json.load(f)
    all_indicators = indicator_mapping.get("indicators", [])
    print(f"✓ Loaded {len(all_indicators)} indicators from codebook")
except Exception as e:
    print(f"⚠ Warning: Could not load indicator mapping: {e}")
    all_indicators = []

# ===== Setup Embedding Model =====
print("Loading embedding model...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.embed_model = embed_model

# ===== Load Evidence Index =====
print(f"Loading evidence index from {EVIDENCE_INDEX_PATH}...")
try:
    evidence_storage = StorageContext.from_defaults(persist_dir=str(EVIDENCE_INDEX_PATH))
    evidence_index = load_index_from_storage(
        evidence_storage,
        embed_model=embed_model,
    )
    print("✓ Evidence index loaded successfully")
except Exception as e:
    print(f"❌ Error loading evidence index: {e}")
    print("Please run build_evidence_index.py first")
    exit(1)

# ===== Load Rule Index (optional) =====
print(f"Loading rule index from {RULE_INDEX_PATH}...")
try:
    rule_storage = StorageContext.from_defaults(persist_dir=str(RULE_INDEX_PATH))
    rule_index = load_index_from_storage(
        rule_storage,
        embed_model=embed_model,
    )
    print("✓ Rule index loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load rule index: {e}")
    print("  Rule-based detection will be skipped")
    rule_index = None

# ===== Setup LLM =====
print("Setting up LLM (Ollama qwen2.5:7b)...")
llm = Ollama(model="qwen2.5:7b", temperature=0.3)
Settings.llm = llm

# ===== Setup Reranker =====
print("Setting up reranker...")
try:
    # Use parameters to avoid multiprocessing issues
    # use_fp16=True: Enable half-precision for memory efficiency
    # top_n: Limit reranking output to reduce resource usage
    reranker = FlagEmbeddingReranker(
        model="BAAI/bge-reranker-large",
        top_n=5,
        use_fp16=True
    )
    print("✓ Reranker initialized successfully")
except Exception as e:
    print(f"⚠ Warning: Could not initialize reranker: {e}")
    print("  Continuing without reranking...")
    reranker = None

# ===== Create Query Engines =====
print("Creating query engines...")
if reranker is not None:
    evidence_engine = evidence_index.as_query_engine(
        similarity_top_k=15,
        node_postprocessors=[reranker]
    )
else:
    evidence_engine = evidence_index.as_query_engine(
        similarity_top_k=5
    )

rule_engine = None
if rule_index is not None:
    rule_engine = rule_index.as_query_engine(
        similarity_top_k=5
    )

# ===== Detection Function =====
def detect_radicalisation(post_text, image_description=""):
    """
    Detect radicalisation indicators in a post using RAG-based LLM.
    
    Args:
        post_text (str): The post content to analyze
        image_description (str): Description of any images in the post
    
    Returns:
        dict: Detection result with indicators, reasoning, and confidence
    """
    
    query_text = f"""POST:
{post_text}

IMAGE:
{image_description}"""

    print("\n" + "="*80)
    print("DETECTING RADICALISATION INDICATORS")
    print("="*80)

    # Step 1: Evidence retrieval
    print("\n[1/3] Retrieving similar evidence cases...")
    try:
        evidence_result = evidence_engine.query(query_text)
        evidence_text = str(evidence_result)
        print(f"✓ Found {len(evidence_result.source_nodes) if hasattr(evidence_result, 'source_nodes') else '?'} relevant cases")
    except Exception as e:
        print(f"⚠ Error retrieving evidence: {e}")
        evidence_text = "(No evidence retrieved)"

    # Step 2: Rule retrieval (if available)
    rules_text = ""
    if rule_engine is not None:
        print("[2/3] Retrieving relevant rules...")
        try:
            rules_result = rule_engine.query(query_text)
            rules_text = str(rules_result)
            print(f"✓ Found {len(rules_result.source_nodes) if hasattr(rules_result, 'source_nodes') else '?'} relevant rules")
        except Exception as e:
            print(f"⚠ Error retrieving rules: {e}")
    else:
        print("[2/3] (Skipping rule retrieval - rule index not available)")

    # Step 3: Build prompt and query LLM
    print("[3/3] Querying LLM for detection...")
    
    # Format available indicators for the prompt
    indicators_list = "\n".join([f"  - {ind}" for ind in all_indicators])
    if not indicators_list:
        indicators_list = "  (No indicators available)"
    
    separator = "="*60
    prompt = f"""You are an expert analyst detecting signs of radicalisation in social media posts.

TASK: Analyze the target post for indicators of radicalisation from the EXACT list provided below.

{separator}
AVAILABLE INDICATORS (from codebook):
{separator}
{indicators_list}

{separator}
SIMILAR EVIDENCE CASES (from known radical content):
{separator}
{evidence_text}

{separator}
RELEVANT INDICATOR DEFINITIONS:
{separator}
{rules_text if rules_text else '(No specific indicators available)'}

{separator}
TARGET POST TO ANALYZE:
{separator}
Content: {post_text}
Image Description: {image_description if image_description else '(No images)'}

{separator}
YOUR ANALYSIS:
{separator}
Please provide your analysis in the following JSON format:

{{
  "has_radicalisation_indicators": true/false,
  "indicators_detected": ["[INDICATOR_ID] Category >> indicator_name", "[INDICATOR_ID] Category >> indicator_name"],
  "reasoning": "Explain your analysis based on the evidence and indicators above",
  "confidence_level": "low/medium/high",
  "evidence_references": "Cite which evidence cases or indicators support your conclusion"
}}

CRITICAL REQUIREMENTS:
1. ONLY select indicators from the "AVAILABLE INDICATORS" list above
2. The "indicators_detected" field MUST contain the EXACT format: "[NUMBER] Category >> indicator_name"
3. If no indicators match, set "has_radicalisation_indicators" to false and "indicators_detected" to []
4. Base your analysis on concrete evidence from the cases provided
5. Only output valid JSON format - no other text"""

    try:
        response = llm.complete(prompt)
        result_text = response.text if hasattr(response, 'text') else str(response)
        print("✓ LLM analysis complete")
        return result_text
    except Exception as e:
        print(f"❌ Error querying LLM: {e}")
        return f'{{"error": "LLM query failed: {str(e)}"}}'


# ===== Test =====
if __name__ == "__main__":
    # Test with sample posts
    test_posts = [
        {
            "content": "I lack motivation. I think it's because I'm comfortable. I have a place to live and a job. Stability is dangerous for creativity.",
            "image": ""
        },
        {
            "content": "Join us in the struggle for justice and freedom.",
            "image": ""
        }
    ]
    
    print("\n" + "="*80)
    print("TESTING RADICALISATION DETECTION")
    print("="*80)
    
    for i, post in enumerate(test_posts, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST {i}")
        print(f"{'#'*80}\n")
        
        result = detect_radicalisation(post["content"], post["image"])
        print("\nRESULT:")
        print(result)
        print()

