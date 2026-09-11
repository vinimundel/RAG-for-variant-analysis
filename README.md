# RAG for variant analysis

Pipeline de recuperação de literatura e análise mecanística de variantes, extraído
do projeto Epistasis_predictor. Perfis de consulta preservados: BRAF e GSDMD.

PDF → GROBID/TEI → chunks com proveniência → BM25 + MedCPT/Qdrant → RRF →
MedCPT Cross-Encoder → Ollama → contratos Pydantic.

PDFs são processados exclusivamente por GROBID. Erros de extração são propagados;
não existe fallback para PyMuPDF. Abstracts PubMed continuam sendo uma fonte separada.
O fallback determinístico da inferência LLM continua registrado no resultado.

## Instalação

Requer Python 3.11 ou superior. Na raiz deste repositório:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Os encoders trabalham com cache local. Faça o download explícito antes da indexação:

```bash
python - <<'PYTHON'
from huggingface_hub import snapshot_download
for model in ('MedCPT-Query-Encoder', 'MedCPT-Article-Encoder', 'MedCPT-Cross-Encoder'):
    snapshot_download('ncbi/' + model)
PYTHON
ollama pull llama3.1:8b
```

Ollama deve estar em execução. Configure `OLLAMA_BASE_URL` e `OLLAMA_MODEL` se necessário.
O Qdrant é persistido localmente por padrão; `compose.yaml` também oferece servidor.

## Corpus e índice

Forneça seu próprio CSV bibliográfico e PDFs. Nenhum corpus ou resultado do projeto
original é distribuído aqui. Todos os caminhos abaixo são locais a este repositório.

```bash
mkdir -p data/input/BRAF/evidence
cp /caminho/literatura.csv data/input/BRAF/evidence/braf_literature.csv
python -m pipeline.steps.prepare_literature_corpus --gene BRAF
python -m pipeline.steps.download_literature data/input/BRAF/evidence/braf_literature_canonical.csv --gene BRAF
python -m pipeline.steps.download_pubmed_abstracts --gene BRAF --csv data/input/BRAF/evidence/braf_literature_canonical.csv
python -m pipeline.steps.build_literature_layers --gene BRAF
docker compose up -d grobid
python -m pipeline.steps.build_rag_index --gene BRAF --extract-only
docker compose stop grobid
python -m pipeline.steps.build_rag_index --gene BRAF --index-only
```

Separar extração e indexação reduz o consumo simultâneo de memória do GROBID e dos
encoders. TEIs existentes permitem indexar sem o servidor GROBID. A camada
`primary_evidence` sustenta inferência mecanística; `context_reference` permanece
separada e documentos excluídos não entram no índice ativo.

## Integração com tabelas existentes

`pipeline.steps.run_literature_reranking` conserva o fluxo original para tabelas
previamente anotadas. Ele espera uma entrada externa em
`data/output/GENE/scores/gene_structural_variant_ranking.parquet`; este repositório
não calcula essas features. As etapas `*task11*` conservam auditorias e benchmarks
históricos e exigem seus próprios artefatos e anotações humanas.

```bash
python -m pipeline.steps.run_literature_reranking --gene BRAF --phase retrieval
python -m pipeline.steps.run_literature_reranking --gene BRAF --phase inference
```

## Escopo e interpretação

Inclui aquisição de literatura, curadoria, ingestão, retrieval, inferência, anotação
de evidências e avaliação RAG. Não inclui preditores estruturais, dinâmica molecular,
treinamento RAF, PDFs, bancos Qdrant, modelos, dados de pacientes ou resultados de pesquisa.
As pequenas regras biofísicas usadas para anotar hipóteses permanecem como dependências.

Escores de retrieval medem relevância; não são confiança biológica ou probabilidade
clínica. A inferência pode abster-se ou emitir hipótese que exige validação experimental.
Os testes usam dados sintéticos e mocks; não validam precisão científica do corpus real.

`extraction_manifest.json` registra os arquivos de origem e seus hashes antes das
adaptações, sem incluir o histórico Git ou arquivos de dados do projeto original.
