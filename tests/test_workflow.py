import math
import numpy as np
import faiss
import pytest

from biomedical_rag.workflow import (
    build_safe_evidence_answer, ndcg_at_k, query_understanding_agent,
    retrieve_chunks, validate_answer,
)


def test_rules_are_exposed_as_rules():
    analysis = query_understanding_agent('Compare wearable monitoring outcomes in adult patients with usual care')
    assert 'wearable' in analysis['entities']
    assert analysis['pico']['Comparison'] == 'Usual care / comparator group'
    assert query_understanding_agent('an unknown topic')['entities'] == []


def test_validation_is_format_only_and_fallback_is_cited():
    assert not validate_answer('unstructured answer')['is_valid']
    chunk = {'chunk_text': 'A synthetic study reports monitoring outcomes for a research cohort.',
             'title': 'Synthetic test fixture', 'pmcid': 'PMC123', 'section': 'results'}
    answer = build_safe_evidence_answer('monitoring', [chunk])
    assert 'Synthetic test fixture' in answer and 'PMC123' in answer
    assert validate_answer(answer)['is_valid']
    # An unsupported assertion still passes: the validator does not check facts.
    assert validate_answer('Summary: fabricated. Key Findings: fabricated. Limitations: none. PMC123')['is_valid']


def test_historical_ndcg_normalizes_only_returned_labels():
    assert ndcg_at_k([]) == 0
    assert ndcg_at_k([0, 0]) == 0
    assert ndcg_at_k([1, 1, 0]) == 1
    assert ndcg_at_k([0, 1]) == pytest.approx(1 / math.log2(3))


def test_actual_faiss_retrieval_handles_padding_and_article_diversity():
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 0], [.9, .1], [.8, .2]], dtype='float32'))
    text = ' '.join(['wearable monitoring outcomes evidence'] * 20)
    metadata = [dict(chunk_id=str(i), pmcid=pmcid, section='results', title='Synthetic study', chunk_text=text)
                for i, pmcid in enumerate(['PMC1', 'PMC1', 'PMC2'])]
    class Embedder:
        def encode(self, queries, **kwargs):
            assert kwargs['normalize_embeddings']
            return np.array([[1, 0]], dtype='float32')
    results = retrieve_chunks('wearable monitoring', index, metadata, Embedder(), top_k=5)
    assert [r['pmcid'] for r in results] == ['PMC1', 'PMC2']
    assert 'score' not in metadata[0]  # retrieval must not mutate the stored corpus
