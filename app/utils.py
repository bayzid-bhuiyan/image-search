import numpy as np
<<<<<<< HEAD
import pickle
import os
=======
import pickle, os
>>>>>>> c138a92 (Add local model, embeddings, precomputed features)
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

# --- Load data ---
base_dir = os.path.dirname(__file__)
embeddings = np.load(os.path.join(base_dir, "model/embeddings.npy")).astype("float32")
with open(os.path.join(base_dir, "model/filenames.pkl"), "rb") as f:
    filenames = pickle.load(f)
df = pd.read_csv(os.path.join(base_dir, "model/product.csv"))

<<<<<<< HEAD
# --- Load smaller model (~90 MB) ---
model = SentenceTransformer("all-MiniLM-L6-v2")
=======
# --- Load local smaller model ---
model = SentenceTransformer(os.path.join(base_dir, "model/all-MiniLM-L6-v2"))
>>>>>>> c138a92 (Add local model, embeddings, precomputed features)

# --- Build FAISS index ---
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

<<<<<<< HEAD
# --- Feature extraction ---
=======
# --- Extract features from image ---
>>>>>>> c138a92 (Add local model, embeddings, precomputed features)
def extract_feature(image_path):
    img = Image.open(image_path).convert("RGB")
    feat = model.encode([img], convert_to_numpy=True, normalize_embeddings=True)
    return feat.astype("float32")

<<<<<<< HEAD
# --- Search for similar images ---
=======
# --- Search similar images ---
>>>>>>> c138a92 (Add local model, embeddings, precomputed features)
def search_similar(image_path, top_k=5):
    feat = extract_feature(image_path)
    scores, ids = index.search(feat, top_k)
    results = []

    for j, i in enumerate(ids[0]):
        img_path = filenames[i]
        score = float(scores[0][j])
        pid = None
<<<<<<< HEAD
        # Match product id from CSV using main_photo column
        for _, row in df.iterrows():
            if str(row["main_photo"]) in img_path:
                # Ensure your CSV has 'main_image_id' column
=======
        # Match product_id using main_photo column
        for _, row in df.iterrows():
            if str(row["main_photo"]) in img_path:
>>>>>>> c138a92 (Add local model, embeddings, precomputed features)
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

# --- Example usage ---
if __name__ == "__main__":
    test_image = os.path.join(base_dir, "model/image/uploads/your_test_image.jpg")
    top5 = search_similar(test_image)
    for res in top5:
        print(res)
