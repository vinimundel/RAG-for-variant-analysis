"""Gene profiles and local paths for literature analysis."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

GENE_CATALOG = {'BRAF': {'regions': {'n_terminal_regulatory_region': (1, 154),
                      'ras_binding_domain': (155, 227),
                      'cysteine_rich_domain': (234, 280),
                      'serine_rich_hinge': (281, 456),
                      'kinase_domain': (457, 717),
                      'activation_segment': (594, 623),
                      'c_terminal_tail': (718, 766)},
          'functional_motifs': {'p_loop': (464, 471),
                                'alpha_c_helix': (491, 505),
                                'hrd_motif': (574, 576),
                                'dfg_motif': (594, 596),
                                'activation_segment': (594, 623)},
          'literature_aliases': ['BRAF',
                                 'B-RAF',
                                 'B-Raf',
                                 'BRAF1',
                                 'RAFB1',
                                 'P15056',
                                 'v-raf murine sarcoma viral oncogene homolog B',
                                 'serine/threonine-protein kinase B-raf'],
          'literature_mesh_terms': ['Proto-Oncogene Proteins B-raf'],
          'literature_csv': 'data/input/BRAF/evidence/braf_literature.csv',
          'discovery_profile': 'braf_oncology'},
 'GSDMD': {'regions': {'pore_forming_n_terminal_domain': (1, 275),
                       'interdomain_linker': (250, 285),
                       'autoinhibitory_c_terminal_domain': (276, 484)},
           'functional_motifs': {'caspase_3_7_inactivating_site': (84, 90),
                                 'cys191_regulatory_site': (191, 191),
                                 'inflammatory_caspase_cleavage_region': (270, 280),
                                 'interdomain_linker': (250, 285)},
           'literature_aliases': ['GSDMD',
                                  'GSDMDC1',
                                  'DFNA5L',
                                  'P57764',
                                  'gasdermin D',
                                  'gasdermin-D'],
           'literature_mesh_terms': ['Gasdermins'],
           'literature_csv': 'data/input/GSDMD/evidence/gsdmd_literature.csv',
           'discovery_profile': 'neuro_aging'}}


def get_gene_config(gene_symbol: str) -> dict:
    gene = gene_symbol.upper()
    if gene not in GENE_CATALOG:
        raise ValueError(f"Gene {gene!r} has no validated literature profile. "
                         f"Available: {', '.join(GENE_CATALOG)}")
    return GENE_CATALOG[gene]


def paths_for(gene_symbol: str) -> dict[str, Path]:
    gene = gene_symbol.upper()
    get_gene_config(gene)
    input_dir = DATA_DIR / "input" / gene
    output_dir = DATA_DIR / "output" / gene
    return {
        "input": input_dir,
        "evidence": input_dir / "evidence",
        "output": output_dir,
        "scores": output_dir / "scores",
        "figures": output_dir / "figures" / "literature",
        "rag": output_dir / "rag",
    }
