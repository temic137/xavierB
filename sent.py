from sentence_transformers import SentenceTransformer

# This will download the model if it's not already in the cache
model = SentenceTransformer('all-MiniLM-L6-v2')