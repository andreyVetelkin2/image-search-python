#search.py
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import faiss
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
from model import ImageSearchModel

# Конфигурация
device = "cuda" if torch.cuda.is_available() else "cpu"
data_folder = "data/plumbing"

# Загрузка модели
model = ImageSearchModel(device)
preprocess = model.get_preprocess()
model.eval()

# Загрузка PCA и индекса
pca = faiss.read_VectorTransform("pca_transform.bin")
index = faiss.read_index("plumbing_ivfpq.faiss")
index.nprobe = 10

# Имена файлов
filenames = torch.load("filenames.pt")


def search_and_show(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.get_embeddings(tensor).cpu().numpy()
        emb = pca.apply_py(emb)
        faiss.normalize_L2(emb)
        scores, indices = index.search(emb, 5)

    show_results(indices[0], scores[0], image_path)


def show_results(indices, scores, query_image):
    result_window = tk.Toplevel(root)
    result_window.title("Результаты поиска")
    result_window.configure(bg="#f0f0f0")

    # Запрос
    query_img = Image.open(query_image).convert("RGB")
    query_img.thumbnail((200, 200))
    tk_query = ImageTk.PhotoImage(query_img)
    tk.Label(result_window, text="Вы выбрали:", bg="#f0f0f0", font=("Arial", 12)).pack(pady=5)
    panel = tk.Label(result_window, image=tk_query, bg="#f0f0f0")
    panel.image = tk_query
    panel.pack(pady=5)

    sep = ttk.Separator(result_window, orient='horizontal')
    sep.pack(fill='x', padx=10, pady=10)

    result_frame = tk.Frame(result_window, bg="#f0f0f0")
    result_frame.pack(padx=10, pady=10)

    for i, (idx, score) in enumerate(zip(indices, scores)):
        img_path = os.path.join(data_folder, filenames[idx])
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((150, 150))
            tk_img = ImageTk.PhotoImage(img)

            frame = tk.Frame(result_frame, bg="#ffffff", bd=1, relief="solid")
            frame.grid(row=0, column=i, padx=10)

            panel = tk.Label(frame, image=tk_img, bg="#ffffff")
            panel.image = tk_img  # держим ссылку
            panel.pack()

            label = tk.Label(frame, text=f"{filenames[idx]}\nScore: {score:.3f}", bg="#ffffff", wraplength=150,
                             justify="center")
            label.pack(pady=5)

        except Exception as e:
            print(f"Ошибка при показе {img_path}: {e}")


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if file_path:
        search_and_show(file_path)


# GUI
root = tk.Tk()
root.title("Визуальный поиск сантехники")
root.geometry("500x250")
root.configure(bg="#e8e8e8")

label = tk.Label(root, text="Выберите изображение для поиска", font=("Arial", 16), bg="#e8e8e8")
label.pack(pady=20)

btn = tk.Button(root, text="Загрузить картинку", command=select_file, font=("Arial", 14), bg="#4CAF50", fg="white",
                padx=20, pady=10)
btn.pack()

root.mainloop()
