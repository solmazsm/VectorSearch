#include "coarse_fine_multivector.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>


void runMultiVectorRetrieval(const Preprocess& prep, int k, const std::vector<int>& query_ids, const std::string& result_file_path) {
    std::cout << "\n[Coarse-Fine Multi-Vector Retrieval] Starting...\n";

    std::vector<std::vector<float>> multi_query;
    for (int qid : query_ids) {
        multi_query.emplace_back(prep.data.query[qid], prep.data.query[qid] + prep.data.dim);
    }

    
    std::vector<std::pair<float, int>> score_all(prep.data.N);

    for (int i = 0; i < prep.data.N; ++i) {
        float total_dist = 0.0f;
        for (const auto& qvec : multi_query) {
            total_dist += cal_distSqrt(qvec.data(), prep.data.val[i], prep.data.dim);
        }
        score_all[i] = { total_dist / multi_query.size(), i };
    }

    std::sort(score_all.begin(), score_all.end());

 
    std::vector<int> top_candidates;
    int coarse_k = 100;
    for (int i = 0; i < std::min(coarse_k, (int)score_all.size()); ++i) {
        top_candidates.push_back(score_all[i].second);
    }

   
    CoarseIndexer coarse_index(prep.data.dim);
    FineReranker fine_index(prep.data.dim, 32); // M = 32
    fine_index.hnsw.efSearch = 64;

    for (int id : top_candidates) {
        fine_index.add(prep.data.val[id]); // add top coarse results
    }

  
    std::vector<std::pair<float, int>> reranked;
    for (int i = 0; i < prep.data.N; ++i) {
        float total_dist = 0.0f;
        for (const auto& qvec : multi_query) {
            total_dist += cal_distSqrt(qvec.data(), prep.data.val[i], prep.data.dim);
        }
        reranked.push_back({ total_dist / multi_query.size(), i });
    }

    std::sort(reranked.begin(), reranked.end());

    std::ofstream fout(result_file_path);
    for (int i = 0; i < k; ++i) {
        fout << query_ids[0] << " " << reranked[i].second << "\n";
    }
    fout.close();

    std::cout << "Top-" << k << " neighbors written to: " << result_file_path << std::endl;
}
