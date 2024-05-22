from sentence_transformers import SentenceTransformer
import os

cache_folder = os.path.join(os.getcwd(), "cache")

model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_folder)
faiss_title_embedding = model.encode(pdf_subset.title.values.tolist())

len(faiss_title_embedding), len(faiss_title_embedding[0])
