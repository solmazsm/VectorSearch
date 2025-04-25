# This folder contains all the necessary implementations required to replicate the studies submitted to VLDB 2026.
# VectorSearch: A Self-Optimizing Multi-Vector Indexing System for Scalable and Memory-Efficient Vector Retrieval

The experiments used a labeled dataset of 1000 news articles. We implemented the algorithm in Python, using libraries for data manipulation, computations, and NLP. SentenceTransformer encoded document titles into embeddings. Indexes facilitated retrieval. Hyperparameter optimization evaluated combinations of dimensions, thresholds, and models using grid search. All our experiments were performed using the same hardware consisting of RTX NVIDIA 3050 GPUs and i5-11400H @ 2.70GHz with 16GB of memory. The details of each experiment are the following.
We implemented a caching mechanism to store and reuse precomputed embeddings from the Chroma model, enhancing efficiency by eliminating redundant computations. This mechanism efficiently saved embeddings to disk, minimizing the need for recomputation and optimizing resource management. Additionally, the implementation included visualization of the similarity network of news articles, achieved by vectorizing content, calculating pairwise cosine similarity, and constructing a graph representation using NetworkX, visualized with Matplotlib. Further functionality analyzed query time distribution through empirical cumulative distribution function computation and visualization using NumPy and Matplotlib.
 We conducted experiments using three models: all-MiniLM-L6-, 
 roberta-base and bert-base-uncased. The hyperparameters varied included the index dimension $(256, 512, 1024)$ and the similarity threshold $(0.7, 0.8, 0.9)$. 
## 📚 Dataset

**VectorSearch Dataset**  
The VectorSearch dataset is a collection of news articles that have been indexed using vector embeddings for efficient semantic search and retrieval. It contains a wide range of articles covering various topics and sources, providing a rich corpus for research and analysis.

The dataset incorporates articles from multiple sources, including but not limited to:

- [**All the News**](https://components.one/datasets/all-the-news-2-news-articles-dataset):  
  This dataset contains **2,688,878 news articles** and essays from **27 American publications**, spanning from **January 1, 2016** to **April 2, 2020**. It is an expanded edition of the original 2017 dataset (~100,000 articles), offering broader media coverage.

- [**NewsCatcher**](https://www.newscatcherapi.com/):  
  News topics were collected and indexed by the NewsCatcher team, covering **108k news articles** across **eight topics**: business, entertainment, health, nation, science, sports, technology, and world.

---

### 🔻 Additional Datasets Used

- **News Dataset** (via NewsCatcher API)  
- **Glove1.2M** (`glove1.2m.tar.gz`)  
- **Deep1M**  
- **SIFT10M / SIFT1M**

---

### 📥 Download Instructions

To download and preprocess the datasets, run:

```bash
python scripts/prepare_dataset.py --dataset all_the_news
python scripts/prepare_dataset.py --dataset newscatcher



