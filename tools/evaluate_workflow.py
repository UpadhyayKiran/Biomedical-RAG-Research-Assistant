"""Evaluate the actual UI retrieval path, without loading the generation model."""
import argparse
import json
from pathlib import Path

from biomedical_rag import config
from biomedical_rag.workflow import run_retrieval_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k', type=int, choices=[3, 5], default=3)
    args = parser.parse_args()
    import faiss
    from sentence_transformers import SentenceTransformer
    index = faiss.read_index(str(config.path('vectorstore/faiss_index.index')))
    metadata = json.loads(config.path('vectorstore/chunk_metadata.json').read_text(encoding='utf8'))
    model = SentenceTransformer(config.EMBED_MODEL)
    result = run_retrieval_evaluation(index, metadata, model, args.k)
    destination = config.path(f'results/workflow_retrieval_k{args.k}.csv')
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(result.to_string(index=False))
    print(f'Saved: {destination}')


if __name__ == '__main__':
    main()
