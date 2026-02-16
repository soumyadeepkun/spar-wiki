from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.core.database import create_session_factory
from src.retrieval.sparse import SparseQueryEncoder, SparseRetrievalEngine
from src.search.routes import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        db_engine, session_factory = create_session_factory(settings.DATABASE_URL)
        app.state.db_engine = db_engine
        app.state.db_session_factory = session_factory

        app.state.sparse_engine = SparseRetrievalEngine.from_files(
            term_offsets_path=str(settings.TERM_OFFSETS_PATH),
            doc_ids_path=str(settings.DOC_IDS_PATH),
            term_weights_path=str(settings.TERM_WEIGHTS_PATH),
            offsets_key=settings.OFFSETS_KEY,
            w_max=settings.W_MAX,
            qmax=settings.QMAX,
        )
        app.state.query_encoder = SparseQueryEncoder(
            hf_token=settings.HF_TOKEN,
            model_name=settings.SPARSE_MODEL_NAME,
            device=settings.SPARSE_MODEL_DEVICE,
        )

        yield

    except Exception as e:
        raise e.__class__(str(e)) from e
    finally:
        if hasattr(app.state, "db_engine"):
            app.state.db_engine.dispose()


app = FastAPI(lifespan=lifespan)

app.include_router(
    search_router,
    prefix="/search",
    tags=["search"],
)


@app.get("/")
async def default():
    # Default gateway
    unique_terms = app.state.sparse_engine.indexed_terms

    return JSONResponse(f"Server running successfully! No. of indexed terms: {unique_terms}")
