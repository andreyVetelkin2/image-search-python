from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from app import search
from app.utils import setup_logger, timer
import logging

setup_logger()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load():
    logging.info("Loading model and index...")
    search.load_index()

@app.post("/search")
@timer
async def search_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Only image files allowed"})

    content = await file.read()
    try:
        results = search.search(content)
        return {"results": results}
    except Exception as e:
        logging.exception("Search failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
