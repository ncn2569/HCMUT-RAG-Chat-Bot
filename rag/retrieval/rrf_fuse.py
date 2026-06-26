def rrf_fuse_old(dense_list, hyde_list, bm25_list, k=60, weights=None):

    if weights is None:
        weights = [1, 0.5, 2]
    scores = {}
    all_lists = [
        (dense_list, weights[0]),
        (hyde_list, weights[1]),
        (bm25_list, weights[2]),
    ]
    for list, weights in all_lists:
        for idx, rank in list:
            scores[idx] = scores.get(idx, 0) + weights * (1 / (k + rank))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rrf_fuse(*lists, k=60, weights=None):
    if weights is None:
        weights = [1.0] * len(lists)
    scores = {}
    for cur, weight in zip(lists, weights):
        for idx, rank in cur:
            scores[idx] = scores.get(idx, 0) + weight * (1 / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
