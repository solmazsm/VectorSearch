model = SentenceTransformer(
    "all-MiniLM-L6-v2", 
    cache_folder=cache_dir
)

faiss_title_embedding = model.encode(pdf_subset.title.values.tolist())
len(faiss_title_embedding), len(faiss_title_embedding[0])
