
import numpy  as np
### function to evaluate CMC(Cumulative Matching Characteristics) for Re-ID methods
def evaluate_with_index(sorted_similarity_index, gt_results_index):
    # compute AccK
    AccK = np.zeros(len(sorted_similarity_index))
    cmc = np.zeros(len(sorted_similarity_index))
    mAP = 0.

    for q in sorted_similarity_index:
        mask = np.isin(q, gt_results_index)
        right_index_location = np.argwhere(mask==True).flatten()
        AccK[right_index_location[0]:] = 1
        cmc += AccK

        ap = 0.0
        for i in range(len(gt_results_index)):
            d_recall = 1.0 / len(gt_results_index)
            precision = float(i + 1) / (gt_results_index[i] + 1)
            if gt_results_index[i] != 0:
            # last rank precision, not last match precision
                old_precision = float(i) / (gt_results_index[i])
            else:
                old_precision = 1.0
            ap = ap + d_recall * (old_precision + precision) / 2
        mAP += ap

    cmc /= len(sorted_similarity_index)
    mAP /= len(sorted_similarity_index)
    return cmc, mAP
