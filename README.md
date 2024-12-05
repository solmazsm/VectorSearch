This folder contains all the necessary implementations required to replicate the studies presented at The Web Conference 2025.
# VectorSearch
The experiments used a labeled dataset of 1000 news articles. We implemented the algorithm in Python, using libraries for data manipulation, computations, and NLP. SentenceTransformer encoded document titles into embeddings. Indexes facilitated retrieval. Hyperparameter optimization evaluated combinations of dimensions, thresholds, and models using grid search. All our experiments were performed using the same hardware consisting of RTX NVIDIA 3050 GPUs and i5-11400H @ 2.70GHz with 16GB of memory. The details of each experiment are the following.
We implemented a caching mechanism to store and reuse precomputed embeddings from the Chroma model, enhancing efficiency by eliminating redundant computations. This mechanism efficiently saved embeddings to disk, minimizing the need for recomputation and optimizing resource management. Additionally, the implementation included visualization of the similarity network of news articles, achieved by vectorizing content, calculating pairwise cosine similarity, and constructing a graph representation using NetworkX, visualized with Matplotlib. Further functionality analyzed query time distribution through empirical cumulative distribution function computation and visualization using NumPy and Matplotlib.
 We conducted experiments using three models: all-MiniLM-L6-, 
 roberta-base and bert-base-uncased. The hyperparameters varied included the index dimension $(256, 512, 1024)$ and the similarity threshold $(0.7, 0.8, 0.9)$. 
# Dataset
VectorSearch Dataset: The VectorSearch dataset is a collection of news articles that have been indexed using vector embeddings for efficient search and retrieval. It contains a vast array of articles spanning various topics and sources, providing researchers with a rich corpus for exploration and analysis.

The dataset encompasses news articles from multiple sources, including but not limited to those compiled by the NewsCatcher team and the All the News dataset. The VectorSearch dataset leverages vector embeddings to represent the semantic content of articles, enabling advanced search capabilities based on semantic similarity.

By harnessing the power of vector embeddings, the VectorSearch dataset offers researchers the opportunity to delve into large-scale text analysis tasks, such as clustering, classification, and recommendation systems. The incorporation of vector embeddings allows for more nuanced and context-aware search and retrieval compared to traditional keyword-based methods.

With its extensive collection of news articles across various topics and sources, the VectorSearch dataset serves as a valuable resource for researchers interested in exploring the dynamics of media content, analyzing trends, and understanding the dissemination of information in the digital age.

# Download Dataset
Instructions on how to download the dataset can be found here.

<a href="https://components.one/datasets/all-the-news-2-news-articles-dataset">All the News:</a> This dataset contains 2,688,878 news articles and essays from 27 American publications, spanning January 1,2016 to April 2, 2020. It is an expanded edition of the original All the News dataset, which was compiled in early 2017. While the original dataset contains more than 100,000 articles, the new dataset’s greater size and breadth should allow researchers to study a wider selection of media.

<a href="https://www.newscatcherapi.com/">NewsCatcher:</a> Data on news topics was collected by the NewsCatcherteam, which collects and indexes 108k news articles spanning eight topics: business, entertainment, health, nation, science,
sports, technology, and the world.

# Evaluation Implementation


For any questions, concerns, or comments for improvements, etc, please create an issue on the issues page for this project, or email the authors directly.


