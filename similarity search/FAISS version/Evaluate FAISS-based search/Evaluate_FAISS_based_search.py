;==========================================
; Title:  search_similar_documents_hnswlib
; Author: Solmaz Seyed Monir
; Date:   3 March 2024
;==========================================

for query, relevant_ids in ground_truth.items():
    similar_docs_faiss = search_similar_documents_faiss(query, index_faiss, pdf_subset)
    precision, recall = evaluate_performance(relevant_ids, similar_docs_faiss)
    faiss_results.append({'query': query, 'precision': precision, 'recall': recall})
