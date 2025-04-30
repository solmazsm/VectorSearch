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
| **SIFT10M / SIFT1M** | Visual feature vectors from image datasets (benchmarking) | [SIFT10M / SIFT1M](http://corpus-texmex.irisa.fr/) |

In our system, the input file (e.g., audio.data) is a binary file containing vectors in 32-bit floating-point (float32) format. Compared to text-based formats, binary files enable significantly faster read times and lower storage overhead, making them well-suited for large-scale, high-dimensional vector retrieval tasks. We extended this support to handle multi-vector inputs per document, enabling retrieval over sets of semantic embeddings instead of single vectors.([**audio.data**]([https://www.newscatcherapi.com/](https://github.com/RSIA-LIESMARS-WHU/LSHBOX-sample-data)), which is in [**float data type**]([https://www.newscatcherapi.com/](https://github.com/RSIA-LIESMARS-WHU/LSHBOX?tab=readme-ov-file)] ) 
