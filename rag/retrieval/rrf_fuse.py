def rrf_fuse(*lists, k=60):

    scores = {}

    for list in lists:
        for idx,rank in list:
            scores[idx] = scores.get(idx,0) + 1/(k+rank)

    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)

    return ranked


def rrf_fuse_test(dense_list, hyde_list, bm25_list, k=60, weights=[1, 1, 2]):

    scores = {}
    all_lists=[
        (dense_list,weights[0]),
        (hyde_list,weights[1]),
        (bm25_list,weights[2])
    ]
    for list,weights in all_lists:
        for idx,rank in list:
            scores[idx] = scores.get(idx,0) + weights*(1/(k+rank))

    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)

    return ranked
