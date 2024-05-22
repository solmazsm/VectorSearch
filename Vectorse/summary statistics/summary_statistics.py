#summary statistics for precision, recall, and query time

summary_stats = results_df.groupby(['pretrained_model', 'index_dimension', 'similarity_threshold']).agg({
    'precision': ['mean', 'median', 'min', 'max'],
    'recall': ['mean', 'median', 'min', 'max'],
    'query_time': ['mean', 'median', 'min', 'max']
}).reset_index()


print(summary_stats)
