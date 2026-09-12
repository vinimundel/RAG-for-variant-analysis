"""Workspace-relative paths for arbitrary biomedical collections."""
import os
import re
from pathlib import Path

ROOT = Path(os.getenv('RAG_WORKSPACE', Path.cwd())).resolve()
DATA_DIR = ROOT / 'data'


def validate_collection(name: str) -> str:
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Collection must start with a letter and contain only letters, digits, _ or -')
    return name.upper()


def get_gene_config(gene_symbol: str) -> dict:
    gene = validate_collection(gene_symbol)
    return {'literature_aliases': [gene], 'literature_mesh_terms': [],
            'literature_csv': f'data/input/{gene}/evidence/{gene.lower()}_literature.csv'}


def paths_for(gene_symbol: str) -> dict[str, Path]:
    gene = validate_collection(gene_symbol)
    input_dir = DATA_DIR / 'input' / gene
    output_dir = DATA_DIR / 'output' / gene
    return {'input': input_dir, 'evidence': input_dir / 'evidence',
            'output': output_dir, 'rag': output_dir / 'rag'}
