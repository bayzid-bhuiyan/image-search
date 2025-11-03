from sentence_transformers import SentenceTransformer

# Load smaller model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Save locally in your repo
model.save('./app/model/all-MiniLM-L6-v2')
print("Model downloaded and saved locally!")
