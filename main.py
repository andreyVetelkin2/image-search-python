import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import faiss
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
from io import BytesIO
from model import ImageSearchModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ImageSearchModel(device)
preprocess = model.get_preprocess()
model.eval()

pca = faiss.read_VectorTransform("pca_transform.bin")
index = faiss.read_index("plumbing_ivfpq.faiss")
index.nprobe = 10
filenames = torch.load("filenames.pt")
data_folder = "data/plumbing"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data/plumbing", StaticFiles(directory="data/plumbing"), name="plumbing")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.get_embeddings(tensor).cpu().numpy()
        emb = pca.apply_py(emb)
        faiss.normalize_L2(emb)
        scores, indices = index.search(emb, 5)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "filename": filenames[idx],
            "score": float(score)
        })
    return JSONResponse(content={"results": results})
