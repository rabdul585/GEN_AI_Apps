"""
Configuration File
Settings for LLM parameters, RAG system, and application
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# API KEYS
# ============================================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

for directory in [DATA_DIR, CHROMA_DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LLM GENERATION PARAMETERS
# ============================================================================

# Temperature Presets
TEMPERATURE_PRESETS = {
    "Precise (0.1)": {"temp": 0.1, "top_p": 0.1, "desc": "Most accurate, deterministic"},
    "Balanced (0.5)": {"temp": 0.5, "top_p": 0.9, "desc": "Balanced accuracy & fluency"},
    "Creative (0.9)": {"temp": 0.9, "top_p": 0.95, "desc": "More creative & varied"}
}

# LLM Parameters
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_MAX_TOKENS = 2000
MIN_TOKENS = 10
MAX_TOKENS_LIMIT = 4000

# Frequency and Presence Penalties
FREQUENCY_PENALTY = 0.0  # -2.0 to 2.0
PRESENCE_PENALTY = 0.0   # -2.0 to 2.0

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

AVAILABLE_MODELS = {
    "Grok 4.1 Fast": "x-ai/grok-4.1-fast",
    "Gemini 2.0 Flash": "google/gemini-2.0-flash-exp:free",
    "Llama 4 Maverick": "meta-llama/llama-4-maverick:free",
    "DeepSeek R1": "deepseek/deepseek-r1:free"
}

DEFAULT_OCR_MODEL = "x-ai/grok-4.1-fast"
DEFAULT_RAG_MODEL = "x-ai/grok-4.1-fast"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ============================================================================
# RAG CONFIGURATION
# ============================================================================

# Retrieval Settings
DEFAULT_RETRIEVAL_K = 5
MIN_RETRIEVAL_K = 1
MAX_RETRIEVAL_K = 20
SIMILARITY_THRESHOLD = 0.3

# Chunking
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 4000

CHUNKING_STRATEGIES = ["Semantic", "Fixed Size", "Sentence-based"]
DEFAULT_CHUNKING_STRATEGY = "Semantic"

# Embeddings
EMBEDDING_PROVIDERS = {
    "HuggingFace (Free)": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384
    },
    "OpenAI (Paid)": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536
    }
}

DEFAULT_EMBEDDING_PROVIDER = "HuggingFace (Free)"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # FIXED: Added this line
EMBEDDING_BATCH_SIZE = 32

# ChromaDB
CHROMA_COLLECTION_NAME = "ocr_documents"
CHROMA_DISTANCE_METRIC = "cosine"

# ============================================================================
# OCR & IMAGE PROCESSING
# ============================================================================

MAX_IMAGE_SIZE = (2048, 2048)
SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "gif", "webp", "pdf", "docx", "doc"]
SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "gif", "webp"]
SUPPORTED_DOCUMENT_FORMATS = ["pdf", "docx", "doc"]
MIN_TEXT_LENGTH = 50

OCR_PROMPT = """Extract ALL text from this image with maximum accuracy.
- Extract every word, number, and symbol
- Preserve line breaks and structure
- No explanations or commentary
OUTPUT: Only extracted text."""

# ============================================================================
# RAG PROMPTS
# ============================================================================

RAG_SYSTEM_PROMPT = """You are a document assistant. Answer questions using only the provided context.

RULES:
1. Use ONLY information from the context
2. If answer not found, say "I cannot find this information"
3. Be precise and cite sources
4. Maintain original terminology"""

RAG_QUERY_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

# ============================================================================
# API & ERROR HANDLING
# ============================================================================

MAX_RETRIES = 3
RETRY_DELAY = 3
REQUEST_TIMEOUT = 120
EXPONENTIAL_BACKOFF = True

# ============================================================================
# UI CONFIGURATION
# ============================================================================

APP_TITLE = "RAG OCR & Document Q&A"
APP_ICON = "🔍"
LAYOUT = "wide"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_generation_params(temperature_preset=None, custom_temp=None, custom_top_p=None, custom_max_tokens=None):
    """Get complete LLM generation parameters"""
    if temperature_preset and temperature_preset in TEMPERATURE_PRESETS:
        preset = TEMPERATURE_PRESETS[temperature_preset]
        temp = preset["temp"]
        top_p = preset["top_p"]
    else:
        temp = custom_temp if custom_temp is not None else DEFAULT_TEMPERATURE
        top_p = custom_top_p if custom_top_p is not None else DEFAULT_TOP_P
    
    max_tokens = custom_max_tokens if custom_max_tokens is not None else DEFAULT_MAX_TOKENS
    
    return {
        "temperature": temp,
        "top_p": top_p,
        "top_k": DEFAULT_TOP_K,
        "max_tokens": max_tokens,
        "frequency_penalty": FREQUENCY_PENALTY,
        "presence_penalty": PRESENCE_PENALTY
    }

def validate_config():
    """Validate configuration"""
    errors = []
    if not OPENROUTER_API_KEY:
        errors.append("⚠️ OPENROUTER_API_KEY not set")
    if DEFAULT_TEMPERATURE < 0 or DEFAULT_TEMPERATURE > 2:
        errors.append("⚠️ Temperature out of range")
    return errors

# Validate
errors = validate_config()
if errors:
    for e in errors:
        print(e)