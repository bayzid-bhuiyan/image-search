from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io, os
import numpy as np
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- FastAPI app ---
app = FastAPI(title="Fashion Image Search API")

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
EMBED_PATH = os.path.join(MODEL_DIR, "embeddings.npy")
FILES_PATH = os.path.join(MODEL_DIR, "filenames.pkl")
CSV_PATH = os.path.join(MODEL_DIR, "product.csv")
LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, "all-MiniLM-L6-v2")

# --- Load precomputed embeddings and filenames ---
embeddings = np.load(EMBED_PATH).astype("float32")
with open(FILES_PATH, "rb") as f:
    filenames = pickle.load(f)

# --- Load CSV ---
df = pd.read_csv(CSV_PATH)

# --- Load local smaller model ---
model = SentenceTransformer(LOCAL_MODEL_PATH)

# --- Build FAISS index ---
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# --- Feature extraction ---
def extract_feature(image_path):
    img = Image.open(image_path).convert("RGB")
    feat = model.encode([img], convert_to_numpy=True, normalize_embeddings=True)
    return feat.astype("float32")

# --- Search similar images ---
def search_similar(image_path, top_k=5):
    feat = extract_feature(image_path)
    scores, ids = index.search(feat, top_k)
    results = []

    for j, i in enumerate(ids[0]):
        img_path = filenames[i]
        score = float(scores[0][j])
        pid = None
        # Match product_id using main_photo column
        for _, row in df.iterrows():
            if str(row["main_photo"]) in img_path:
                pid = int(row.get("main_image_id", 0))
                break
        url = f"https://ladonna.com.bd/product/product_description.php?id={pid}" if pid else None
        results.append({
            "image": os.path.basename(img_path),
            "score": score,
            "product_id": pid,
            "url": url
        })
    return results

# --- Routes ---
@app.get("/")
def home():
    return {"message": "Welcome to Fashion Image Search API 👗"}

@app.post("/search")
async def search(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    temp_path = os.path.join(BASE_DIR, "temp.jpg")
    img.save(temp_path)

    results = search_similar(temp_path)
    os.remove(temp_path)

    return JSONResponse(content=results)
