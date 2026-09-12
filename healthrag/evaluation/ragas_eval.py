"""RAGAS collections API on saved answers; the judge is explicit and local by default."""
import argparse
import asyncio
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics


def build_metrics(model: str = "qwen2.5:1.5b",
                  base_url: str = "http://localhost:11434/v1",
                  timeout: float = 180) -> tuple[dict, object]:
    # Optional imports keep the serving path independent of evaluator dependencies.
    os.environ.setdefault('RAGAS_DO_NOT_TRACK', 'true')
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness, ContextRecall
    client = AsyncOpenAI(base_url=base_url, api_key=os.getenv('RAGAS_API_KEY','ollama'),
                         timeout=timeout,max_retries=0)
    llm = llm_factory(model,client=client,temperature=0,max_tokens=2048)
    return ({'faithfulness':Faithfulness(llm=llm),'context_recall':ContextRecall(llm=llm)}, client)


async def score_report(report: dict, model: str, base_url: str,
                       timeout: float = 180, metrics: dict | None = None) -> dict:
    client = None
    if metrics is None:
        metrics, client = build_metrics(model, base_url, timeout)
    rows=[]
    for case in report['rows']:
        answer=case['result']
        row={'id':case['id'],'status':'skipped','scores':{},'errors':{}}
        if answer['status']!='answered':
            row['reason']='Abstentions and technical failures are evaluated separately, not scored as faithful answers.'
            rows.append(row)
            continue
        contexts=[s['passage'] for s in answer['sources']]
        response=' '.join(c['text'] for c in answer['claims'])
        calls={'faithfulness':dict(user_input=case['question'],response=response,retrieved_contexts=contexts)}
        if case.get('reference'):
            calls['context_recall']=dict(user_input=case['question'],reference=case['reference'],retrieved_contexts=contexts)
        for name,kwargs in calls.items():
            try:
                result=await asyncio.wait_for(metrics[name].ascore(**kwargs),timeout=timeout)
                value=float(result.value)
                if not math.isfinite(value):
                    raise ValueError('Non-finite metric')
                row['scores'][name]=value
            except Exception as error:
                row['errors'][name]=type(error).__name__
        row['status']='scored' if row['scores'] and not row['errors'] else 'partial' if row['scores'] else 'failed'
        rows.append(row)
    if client is not None:
        await client.close()
    summary={name:{'mean':statistics.mean(values) if values else None,'scored_cases':len(values)}
             for name in metrics for values in [[r['scores'][name] for r in rows if name in r['scores']]]}
    return {'ragas_version':version('ragas'),'judge_model':model,'judge_base_url':base_url,
            'judge_is_not_ground_truth':True,'metrics':summary,'rows':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--judge-model',default=os.getenv('RAGAS_MODEL','qwen2.5:1.5b'))
    parser.add_argument('--base-url',default=os.getenv('RAGAS_BASE_URL','http://localhost:11434/v1'))
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    report=json.loads(args.input.read_text())
    result=asyncio.run(score_report(report,args.judge_model,args.base_url,args.timeout))
    result['input_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['metrics'],indent=2))
    if any(row['status'] in {'failed','partial'} for row in result['rows']):
        raise SystemExit(1)


if __name__=='__main__':
    main()
