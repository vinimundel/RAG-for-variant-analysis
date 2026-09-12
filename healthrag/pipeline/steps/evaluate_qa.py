"""Generate a reproducible dataset of answers and deterministic retrieval metrics."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
from healthrag.rag.biomedical import BiomedicalRAG,Question


def evaluate(dataset: Path,output: Path):
    rows=[]
    engine=BiomedicalRAG()
    for line in dataset.read_text().splitlines():
        case=json.loads(line)
        result=engine.invoke(Question.model_validate({k:case[k] for k in ('question','collection','profile')}))
        retrieved={source.pmid for source in result.sources}
        expected=set(case['expected_pmids'])
        rows.append({**case,'status_matches':result.status==case['expected_status'],
                     'source_recall':len(expected & retrieved)/len(expected) if expected else None,
                     'result':result.model_dump()})
        # Persist each case so a WSL restart does not discard completed inference.
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps({'rows':rows,'complete':False},indent=2,ensure_ascii=False)+'\n')
    recalls=[r['source_recall'] for r in rows if r['source_recall'] is not None]
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'complete':True,
            'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'cases':len(rows),
            'mean_source_recall':statistics.mean(recalls) if recalls else None,
            'status_match_rate':sum(r['status_matches'] for r in rows)/len(rows) if rows else None,
            'semantic_entailment':'evaluate_with_ragas_and_human_review','rows':rows}
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('examples/braf/evaluation.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('data/output/BRAF/evaluation.json'))
    args=parser.parse_args()
    result=evaluate(args.dataset,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
