# Validação da extração

Validação local em 11 de setembro de 2026, com Python 3.11.15:

- 231 testes aprovados, incluindo 5 regressões de ingestão GROBID e 3 testes da
  entrada independente de consulta por variante.
- Pacote wheel gerado com sucesso por setuptools.
- Importação de todos os módulos locais sem dependência de código externo ao projeto.
- Ingestor importado com os imports de `fitz`, `pymupdf` e `langchain_community`
  bloqueados, confirmando a remoção da dependência do parser alternativo.
- Hashes dos 75 arquivos selecionados na origem permanecem iguais aos registrados
  em `extraction_manifest.json`: o código original não foi alterado.

Comando reproduzível após instalar as dependências:

```bash
python -m pytest -q
```

Os testes foram executados na cópia independente, usando as bibliotecas já
instaladas no ambiente Python do projeto de origem. Não foi feita uma instalação
completa das dependências em um ambiente novo durante esta validação local.

Os testes de serviço usam respostas sintéticas. Esta validação não executou
GROBID, download de modelos, inferência Ollama ou avaliação científica no corpus real.
O workflow GitHub Actions instala as dependências e executa a suíte em Python 3.11.

Dois wrappers legados V6 (`finalize_task11_v6` e `preflight_task11_v6`) foram
excluídos porque importavam funções já removidas na origem. Os comandos V7 foram
preservados. A suíte inclui 64 verificações de importação dos módulos distribuídos.
