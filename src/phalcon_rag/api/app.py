from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from phalcon_rag.api.schemas import AskRequest, AskResponse, SourceResponse
from phalcon_rag.pipeline_factory import create_pipeline

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_pipeline = create_pipeline()

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/ask", response_model=AskResponse)
def ask(request: AskRequest):
    rag_result = app.state.rag_pipeline.run(
        query=request.question,
        context=request.context,
    )

    sources = [
        SourceResponse(
            id=chunk.id,
            type=chunk.source,
            file=chunk.metadata.get("file_path") or chunk.metadata.get("file"),
            section=chunk.metadata.get("section"),
            method=chunk.metadata.get("method"),
            start_line=chunk.metadata.get("start_line"),
            end_line=chunk.metadata.get("end_line"),
        )
        for chunk in rag_result.chunks
    ]

    return AskResponse(
        answer=rag_result.answer,
        sources=sources,
    )
