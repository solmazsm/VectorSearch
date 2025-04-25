#### This folder contains all the necessary implementations required to replicate the studies submitted to VLDB 2026.
#### VectorSearch: A Self-Optimizing Multi-Vector Indexing System for Scalable and Memory-Efficient Vector Retrieval

The experiments used a labeled dataset of 1000 news articles. We implemented the algorithm in Python, using libraries for data manipulation, computations, and NLP. SentenceTransformer encoded document titles into embeddings. Indexes facilitated retrieval. Hyperparameter optimization evaluated combinations of dimensions, thresholds, and models using grid search. All our experiments were performed using the same hardware consisting of RTX NVIDIA 3050 GPUs and i5-11400H @ 2.70GHz with 16GB of memory. The details of each experiment are the following.
We implemented a caching mechanism to store and reuse precomputed embeddings from the Chroma model, enhancing efficiency by eliminating redundant computations. This mechanism efficiently saved embeddings to disk, minimizing the need for recomputation and optimizing resource management. Additionally, the implementation included visualization of the similarity network of news articles, achieved by vectorizing content, calculating pairwise cosine similarity, and constructing a graph representation using NetworkX, visualized with Matplotlib. Further functionality analyzed query time distribution through empirical cumulative distribution function computation and visualization using NumPy and Matplotlib.
 We conducted experiments using three models: all-MiniLM-L6-, 
 roberta-base and bert-base-uncased. The hyperparameters varied included the index dimension $(256, 512, 1024)$ and the similarity threshold $(0.7, 0.8, 0.9)$. 

## 📚 Dataset

The VectorSearch dataset is a collection of news articles and image feature vectors indexed using semantic vector embeddings for efficient search and retrieval. It includes articles from diverse news sources and visual feature datasets, supporting large-scale text and image retrieval experiments.


---

| Dataset | Description | Link |
|:--------|:------------|:-----|
| **All the News** | 2,688,878 articles from 27 U.S. publications (2016–2020); expanded from original 2017 dataset (~100k articles) | [**All the News**](https://components.one/datasets/all-the-news-2-news-articles-dataset) |
| **NewsCatcher** | 108k+ articles spanning eight topics: business, entertainment, health, nation, science, sports, technology, world | [**NewsCatcher**](https://www.newscatcherapi.com/) |
| **News Dataset (Newscatcher API)** | Subset collected from NewsCatcher, focused on topic-wise categorization | [**NewsCatcher**](https://www.newscatcherapi.com/) |
| **Glove1.2M** | Pretrained word embeddings (1.2 million vectors) for large-scale search tasks | — |
| **Deep1M** | 1 million dense vectors for benchmarking similarity search | — |
| **SIFT10M / SIFT1M** | Visual feature vectors from image datasets (benchmarking) | — |

---

### 📥 Download Instructions

Use the `prepare_dataset.py` script to download and preprocess datasets:

```bash
python scripts/prepare_dataset.py --dataset all_the_news
python scripts/prepare_dataset.py --dataset newscatcher

## 📈 Evaluation Metrics

- **Query Throughput (QPS)**  
- **Memory Usage**  
- **Recall@10**  
- **Mean Precision**  
- **Query Time**

---

## 🛠️ Key Features

- **Multi-vector document embedding** using transformer-based models (e.g., MiniLM, BERT, RoBERTa)
- **Hybrid indexing** with (coarse retrieval) and (fine reranking)
- **Dynamic hyperparameter tuning** using grid search
- **Scalable evaluation** on benchmark datasets: News, Glove1.2M, Deep1M, and SIFT10M
- **Caching support** for efficient inference and reduced computation

