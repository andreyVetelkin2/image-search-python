import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import faiss
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
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

app = FastAPI()

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

    results = [filenames[idx] for idx in indices[0]]
    return JSONResponse(content={"results": results})
