#build_embeddings.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import faiss
import numpy as np
from PIL import Image
from model import ImageSearchModel

# Модель и предобработка
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ImageSearchModel(device)
preprocess = model.get_preprocess()

folder = "data/plumbing"
images = []
filenames = []

def augment_image(img):
    # Простые аугментации для TTA
    from torchvision import transforms
    tta_transforms = [
        transforms.Compose([]),  # оригинал
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
        transforms.Compose([transforms.RandomRotation(15)]),
    ]
    return [t(img) for t in tta_transforms]

for file in os.listdir(folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            path = os.path.join(folder, file)
            pil_img = Image.open(path).convert("RGB")
            augmented_tensors = []
            for aug_img in augment_image(pil_img):
                tensor = preprocess(aug_img)
                augmented_tensors.append(tensor)
            # Усредняем эмбеддинги TTA по картинке позже
            images.append(torch.stack(augmented_tensors))  # shape (tta, C, H, W)
            filenames.append(file)
        except Exception as e:
            print(f"❌ {file}: {e}")
    if len(images) >= 1000:
        break

if not images:
    raise RuntimeError("Нет валидных изображений")

# Получаем эмбеддинги с TTA и усредняем
all_embeddings = []
model.eval()
with torch.no_grad():
    for batch in images:
        batch = batch.to(device)  # (tta, C, H, W)
        emb = model.get_embeddings(batch)  # (tta, emb_dim)
        emb = emb.mean(dim=0)  # усредняем по TTA
        all_embeddings.append(emb.cpu())


embeddings = torch.stack(all_embeddings).numpy()

# PCA + нормализация + IVF-PQ
d = embeddings.shape[1]
pca_dim = 128
nlist = 20

mat = faiss.PCAMatrix(d, pca_dim)
mat.train(embeddings)
embeddings_pca = mat.apply_py(embeddings)
faiss.normalize_L2(embeddings_pca)

quantizer = faiss.IndexFlatIP(pca_dim)
index = faiss.IndexIVFPQ(quantizer, pca_dim, nlist, 16, 8)

index.train(embeddings_pca)
index.add(embeddings_pca)
index.nprobe = 10

# Сохраняем
faiss.write_VectorTransform(mat, "pca_transform.bin")
faiss.write_index(index, "plumbing_ivfpq.faiss")
torch.save(filenames, "filenames.pt")

print(f"✅ Сохранили {len(filenames)} эмбеддингов с PCA и IVF-PQ")