# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from sklearn.metrics import roc_auc_score
def compute_metrics(
    queries: torch.cuda.FloatTensor, 
    targets: torch.cuda.FloatTensor, 
    split: str
) -> dict:
    """Computes all the metrics specified through the cmd-line. 

    Args:
        params (argparse.Namespace): Command-line parameters.
        queries (torch.cuda.FloatTensor): Query embeddings.
        targets (torch.cuda.FloatTensor): Target embeddings.
        split (str): "validation" or "test"
    """
    # get query and target authors
    query_authors = torch.stack(
            [a for x in queries for a in x['ground_truth']]).cpu().numpy()
    target_authors = torch.stack(
            [a for x in targets for a in x['ground_truth']]).cpu().numpy()

    # get all query and target author embeddings
    q_list = torch.stack([e for x in queries
                          for e in x['{}_embedding'.format(split)]]).cpu().numpy()
    t_list = torch.stack([e for x in targets
                          for e in x['{}_embedding'.format(split)]]).cpu().numpy()
    # Get all act indices
    q_indices = torch.stack([e for x in queries
                          for e in x["act_index"]]).cpu().numpy()
    t_indices = torch.stack([e for x in targets
                          for e in x["act_index"]]).cpu().numpy()
    metric_scores = {}
    metric_scores.update(ranking(q_list, t_list, query_authors, target_authors, q_indices, t_indices))
    
    return metric_scores

def ranking(queries, 
            targets,
            query_authors, 
            target_authors, 
            q_indices,
            t_indices,
            metric='cosine', 
):
    num_queries = len(query_authors)
    ranks = np.zeros((num_queries), dtype=np.float32)
    reciprocal_ranks = np.zeros((num_queries), dtype=np.float32)
    
    distances = pairwise_distances(queries, Y=targets, metric=metric, n_jobs=-1)

    auc_scores = []
    for i in range(num_queries):
        # Get the right targets for current query
        target_index = q_indices[i]
        val_targets = np.where(t_indices == target_index)[0]
        dist = distances[i,val_targets]
        labels = [0] * len(val_targets)
        idx = np.where(target_authors[val_targets] == query_authors[i])[0].item()
        labels[idx] = 1
        auc_scores.append(roc_auc_score(labels, 1-dist))
        # sorted_indices = np.argsort(dist)
        # sorted_target_authors = target_authors[sorted_indices]
        # ranks[i] = np.where(sorted_target_authors ==
        #                     query_authors[i])[0].item()
        # reciprocal_ranks[i] = 1.0 / float(ranks[i] + 1)
        
    return_dict = {
        "AUC" : np.mean(auc_scores),
        # 'R@2': np.sum(np.less_equal(ranks, 2)) / np.float32(num_queries),
        # 'R@8': np.sum(np.less_equal(ranks, 8)) / np.float32(num_queries),
        # 'R@16': np.sum(np.less_equal(ranks, 16)) / np.float32(num_queries),
        # 'R@32': np.sum(np.less_equal(ranks, 32)) / np.float32(num_queries),
        # 'R@64': np.sum(np.less_equal(ranks, 64)) / np.float32(num_queries),
        # 'MRR': np.mean(reciprocal_ranks)
    }

    return return_dict
