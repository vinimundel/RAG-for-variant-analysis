"""Download or reuse the PDFs listed in a public, curated demonstration manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
import requests
from healthrag.pipeline.config import paths_for, validate_collection


def prepare(manifest_path: Path, source_dir: Path | None = None):
    manifest = json.loads(manifest_path.read_text())
    collection = validate_collection(manifest['collection'])
    directory = paths_for(collection)['evidence']
    pdf_dir = directory / 'pdfs'
    pdf_dir.mkdir(parents=True, exist_ok=True)
    records, layers, inventory = [], [], []
    for document in manifest['documents']:
        pmid = str(document['pmid'])
        if not pmid.isdigit():
            raise ValueError('PMIDs must be numeric')
        destination = pdf_dir / f'PMID{pmid}.pdf'
        if not destination.exists():
            if source_dir:
                shutil.copy2(source_dir / destination.name, destination)
            else:
                response = requests.get(document['pdf_url'], timeout=120)
                response.raise_for_status()
                if not response.content.startswith(b'%PDF'):
                    raise ValueError(f'Publisher did not return a PDF for {pmid}')
                destination.write_bytes(response.content)
        if not destination.read_bytes().startswith(b'%PDF'):
            raise ValueError(f'Invalid PDF for {pmid}')
        records.append({k:document[k] for k in ['pmid','title','doi','journal','pmcid','year','authors']})
        records[-1]['publication_types'] = ['Journal Article']
        layers.append({'pmid':pmid,'layer':document['corpus_layer']})
        inventory.append({'pubmed_id':pmid,'source_collection':'public_demo',
                          'source_file':destination.name,'source_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
    cache = directory / 'records/pubmed_records.jsonl'
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text('\n'.join(json.dumps(record,ensure_ascii=False) for record in records)+'\n')
    pd.DataFrame(layers).to_csv(directory / f'{collection.lower()}_corpus_layers.csv',index=False)
    pd.DataFrame(inventory).to_csv(directory / f'{collection.lower()}_literature_evidence.csv',index=False)
    report={'collection':collection,'documents':inventory,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    (directory/'demo_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('examples/braf/corpus.json'))
    parser.add_argument('--source-dir',type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare(args.manifest,args.source_dir),indent=2))
