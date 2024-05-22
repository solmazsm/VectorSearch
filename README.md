# VectorSearch
The experiments used a labeled dataset of 1000 news articles. We implemented the algorithm in Python, using libraries for data manipulation, computations, and NLP. SentenceTransformer encoded document titles into embeddings. Indexes facilitated retrieval. Hyperparameter optimization evaluated combinations of dimensions, thresholds, and models using grid search. All our experiments were performed using the same hardware consisting of RTX NVIDIA 3050 GPUs and i5-11400H @ 2.70GHz with 16GB of memory. The details of each experiment are the following.
We implemented a caching mechanism to store and reuse precomputed embeddings from the Chroma model, enhancing efficiency by eliminating redundant computations. This mechanism efficiently saved embeddings to disk, minimizing the need for recomputation and optimizing resource management. Additionally, the implementation included visualization of the similarity network of news articles, achieved by vectorizing content, calculating pairwise cosine similarity, and constructing a graph representation using NetworkX, visualized with Matplotlib. Further functionality analyzed query time distribution through empirical cumulative distribution function computation and visualization using NumPy and Matplotlib.
# Dataset
 NewsCatcher
 Data on news topics was collected by the NewsCatcherteam, which collects and indexes 108k news articles spanning eight topics.
 All the News
 This dataset contains 2,688,878 news articles and essays from 27 American publications, spanning January 1, 2016 to April 2, 2020. We conducted experiments using three models: all-MiniLM-L6-, 
 roberta-base and bert-base-uncased. The hyperparameters varied included the index dimension $(256, 512, 1024)$ and the similarity threshold $(0.7, 0.8, 0.9)$. 
