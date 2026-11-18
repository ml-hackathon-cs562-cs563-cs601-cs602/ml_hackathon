"""FastAPI application entry point for PathSnap.

Loads model artifacts once at startup.
Start command documented in api/README.md.
Logs every request for the evaluator.
Owner: Member A & C
"""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.pipeline import init_artifacts
from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pathsnap.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up PathSnap API: loading model and HMM artifacts...")
    init_artifacts()
    log.info("PathSnap API ready.")
    yield
    log.info("PathSnap API shutting down.")


app = FastAPI(
    title="PathSnap API",
    description="Backend inference and evaluation service for noisy GPS map-matching.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
