import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    """Application lifespan events!"""

    logger.info("Starting RAG API Backend!")

    try:
        logger.info("Application startup completed successfully!")
    except:
        logger.error("Failed to start RAG API")
        raise
    yield

    logger.info("Shutting down RAG API")


app = FastAPI(title="RAG AI Agent API",
              description="FastAPI backend RAG Implementation with Vector search",
              version="0.0.1",
              lifespan=lifespan)

# Allowed origins (can be specific URLs or wildcards)
origins = [
    "http://localhost",
    "http://localhost:3000",  # your frontend dev server
    "https://rag-api.doubledigit-solutions.com"
]
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,             # allowed frontend origins
                    allow_credentials=True,            # allow cookies, headers, sessions
                    allow_methods=["*"],               # allow all HTTP methods
                    allow_headers=["*"])               # allow all headers

@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Serve the favicon requests to prevent 404 errors."""
    return FileResponse("favicon.ico")


@app.get("/", tags=["General"])
async def root() -> dict:
    """Root endpoint."""
    return {"message": "Welcome to RAG API!",
            "version": "0.0.1",
            "endpoints": {
                "health": "/healthz",
                "answer": "/ans",
                "greet": "/greet/{name}"
            }
    }

