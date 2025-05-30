import faiss
import torch
from PIL import Image
import numpy as np
from io import BytesIO

model = None
preprocess = None
index = None
pca = None
filenames = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_index():
    global model, preprocess, pca, index, filenames

    from app.model import ImageSearchModel
    model = ImageSearchModel(device)
    preprocess = model.get_preprocess()
    model.eval()

    pca = faiss.read_VectorTransform("app/index/pca_transform.bin")
    index = faiss.read_index("app/index/plumbing.index")
    index.nprobe = 10
    filenames = torch.load("app/index/filenames.pt")

def search(file_bytes: bytes, top_k: int = 5):
    img = Image.open(BytesIO(file_bytes)).convert("RGB")

    if img.mode != "RGB":
        raise ValueError("Image must be RGB")

    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.get_embeddings(tensor).cpu().numpy()
        emb = pca.apply_py(emb)
        faiss.normalize_L2(emb)
        scores, indices = index.search(emb, top_k)

    return [filenames[idx] for idx in indices[0]]
