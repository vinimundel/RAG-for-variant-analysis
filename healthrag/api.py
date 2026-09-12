"""Local research API with bounded single-request admission for embedded Qdrant."""
import logging
import threading
from fastapi import FastAPI, HTTPException
from healthrag.rag.biomedical import Answer, BiomedicalRAG, Question
from healthrag.rag.profiles import load_profile

app = FastAPI(title='Biomedical Evidence RAG', version='0.2.0',
              description='Research literature QA with traceable citations, not clinical decision support.')
engine = BiomedicalRAG()
request_lock = threading.Lock()


@app.get('/health')
def health():
    """Liveness only; models and indexes are checked when querying."""
    return {'status': 'ok', 'version': '0.2.0', 'mode': 'local-research'}


@app.get('/profiles')
def profiles():
    return [load_profile(name).model_dump() for name in ('general', 'braf_oncology')]


@app.post('/query', response_model=Answer)
def query(request: Question):
    if not request_lock.acquire(blocking=False):
        raise HTTPException(429, 'A query is running; retry shortly.')
    try:
        result = engine.invoke(request)
        if result.status in {'generation_failed', 'invalid_generation'}:
            raise HTTPException(503 if result.status == 'generation_failed' else 502, result.model_dump())
        return result
    except FileNotFoundError:
        raise HTTPException(404, 'Collection index not found. Build it before querying.')
    except HTTPException:
        raise
    except Exception as error:
        logging.getLogger(__name__).warning('query_failed error_type=%s', type(error).__name__)
        raise HTTPException(503, 'Retrieval unavailable; check the index and model cache.')
    finally:
        request_lock.release()
