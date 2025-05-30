# Image-search-python 🔍🚰

**Кратко:**  
Система поиска по изображениям сантехники на базе EfficientNet-B3 + Faiss с TTA и PCA + IVF-PQ.  

---

## Что внутри?

- Модель EfficientNet-B3 с предобученными весами .
- TTA (Test-Time Augmentation) —  для стабильно лучших эмбеддингов.
- Индексация Faiss: PCA + IVF-PQ (можно заменить на HNSW, если хочешь меньше мороки).
- FastAPI + Gunicorn для быстрого и стабильного API.
- Докеризация и docker-compose для запуска.
- Логгирование, базовая валидация .
- Пример теста для автопроверки.

---

## Как запустить

### Локально с Docker:

```bash
git clone https://github.com/your-repo/plumbing-visual-search.git
cd image-search-python
docker-compose up --build
```

### Локально без Docker:

```bash
git clone https://github.com/your-repo/plumbing-visual-search.git
cd image-search-python
pip install -r requirements.txt
uvicorn main:app --reload
```

API будет доступен по адресу: `http://localhost:8000/search`

### Пример запроса:

```bash
curl -X POST "http://localhost:8000/search" -F "file=@your_image.jpg"
```

Ответ:

```json
{
  "results": [
    "image_123.jpg",
    "image_456.jpg",
    "image_789.jpg",
    "image_012.jpg",
    "image_345.jpg"
  ]
}
```

---

## Структура проекта

- `app/` — код FastAPI и модели  
- `app/index/` — сохранённые индексы и данные  
- `Dockerfile` и `docker-compose.yml` — для запуска в докере  
- `requirements.txt` — зависимости  
- `test/` — базовые тесты API  

---

## Как улучшить

- Подменить EfficientNet на CLIP для топового результата (уже жду твоих пулл-реквестов)  
- Сделать обновляемый индекс, чтобы не перезапускать сборку с нуля  
- Добавить продвинутую аутентификацию и лимиты на запросы (иначе сдохнешь от трафика)  
- Прокачать логи и метрики — Prometheus, Grafana и т.д.  

---


