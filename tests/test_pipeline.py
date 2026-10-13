import json
import numpy as np
import pytest

from biomedical_rag import config, create_chunks, create_faiss_index


def test_section_chunking_and_index_alignment(tmp_path, monkeypatch):
    source = tmp_path / 'json'
    source.mkdir()
    words = ['evidence'] * 650
    record = {'pmcid': 'PMC123', 'title': 'Synthetic study', 'sections': [
        {'section': 'results', 'text': ' '.join(words)},
        {'section': 'references', 'text': ' '.join(words)},
    ]}
    (source / 'PMC123.json').write_text(json.dumps(record), encoding='utf8')
    monkeypatch.setattr(create_chunks, 'JSON_FOLDER', str(source))
    monkeypatch.setattr(create_chunks, 'CHUNKS_JSONL', str(tmp_path / 'chunks.jsonl'))
    monkeypatch.setattr(create_chunks, 'CHUNKING_REPORT', str(tmp_path / 'report.csv'))
    create_chunks.create_chunks()
    chunks = create_faiss_index.load_chunks(tmp_path / 'chunks.jsonl')
    assert [len(c['chunk_text'].split()) for c in chunks] == [500, 250]
    assert {c['section'] for c in chunks} == {'results'}
    index = create_faiss_index.create_faiss_index(np.eye(2, dtype='float32'))
    assert index.ntotal == len(chunks)
    create_faiss_index.save_metadata(chunks, tmp_path / 'metadata.json')
    metadata = json.loads((tmp_path / 'metadata.json').read_text())
    assert [c['faiss_id'] for c in metadata] == [0, 1]


def test_workspace_override_does_not_depend_on_current_directory(tmp_path, monkeypatch):
    monkeypatch.setenv('BIOMEDICAL_RAG_WORKSPACE', str(tmp_path))
    assert config.path('json') == tmp_path / 'json'
