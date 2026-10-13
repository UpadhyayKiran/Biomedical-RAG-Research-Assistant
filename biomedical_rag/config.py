"""Shared paths and environment settings; never points into the academic submission."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get(name, default=''):
    return os.environ.get(name, default)


def path(relative):
    """Resolve generated artifacts beneath BIOMEDICAL_RAG_WORKSPACE."""
    return Path(get('BIOMEDICAL_RAG_WORKSPACE', str(ROOT / 'workspace'))).expanduser().resolve() / relative


EMBED_MODEL = get('BIOMEDICAL_RAG_EMBED_MODEL', 'pritamdeka/S-PubMedBert-MS-MARCO')
BASE_MODEL = get('BIOMEDICAL_RAG_BASE_MODEL', 'BioMistral/BioMistral-7B')
LOG_QUERIES = get('BIOMEDICAL_RAG_LOG_QUERIES', '0') == '1'
