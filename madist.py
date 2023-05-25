import numpy as np
import sklearn.metrics as sk
from scipy.spatial.distance import pdist
from sklearn import metrics
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])




class0feature = np.load("task1class0feature.npy")
class1feature = np.load("task1class1feature.npy")
class2feature = np.load("task1class2feature.npy")
class3feature = np.load("task1class3feature.npy")


# print(class0feature.shape)
# exit()

listlen = len(class0feature)
avgfeature0 = np.zeros((1,128))
for feature in class0feature:
    avgfeature0 += feature
avgfeature0 = avgfeature0/listlen

listlen = len(class1feature)
avgfeature1 = np.zeros((1,128))
for feature in class1feature:
    avgfeature1 += feature
avgfeature1 = avgfeature1/listlen

listlen = len(class2feature)
avgfeature2 = np.zeros((1,128))
for feature in class2feature:
    avgfeature2 += feature
avgfeature2 = avgfeature2/listlen

listlen = len(class3feature)
avgfeature3 = np.zeros((1,128))
for feature in class3feature:
    avgfeature3 += feature
avgfeature3 = avgfeature3/listlen

temp = np.zeros((128,128))
count = 0
for feature in class0feature:
    temp += np.dot((feature.T - avgfeature0.T),(feature - avgfeature0))
    count += 1
# print(temp/count)

for feature in class1feature:
    temp += np.dot((feature.T - avgfeature1.T),(feature - avgfeature1))
    count += 1

for feature in class2feature:
    temp += np.dot((feature.T - avgfeature2.T),(feature - avgfeature2))
    count += 1

for feature in class3feature:
    temp += np.dot((feature.T - avgfeature3.T),(feature - avgfeature3))
    count += 1

# temp = temp/count
inversesigma = np.linalg.inv(temp)
# print(inversesigma)
# exit()
maxormin = True
method = 'cosine'

diffdist = 0
right_score = []
wrong_score = []
for feature in class0feature:
    temp0 = np.dot((feature - avgfeature0), inversesigma)
    # print(temp0)
    dist0 = np.dot(temp0,(feature.T - avgfeature0.T))
    # print(dist0)
    temp1 = np.dot((feature - avgfeature1), inversesigma)
    # print(temp0)
    dist1 = np.dot(temp1,(feature.T - avgfeature1.T))
    # print(dist1)
    # exit()
    if maxormin == True:
        right_score.append(max(dist0[0],dist1[0]))
        # print(right_score)
    else:
        right_score.append(min(dist0[0],dist1[0]))


for feature in class1feature:
    temp0 = np.dot((feature - avgfeature0), inversesigma)
    # print(temp0)
    dist0 = np.dot(temp0,(feature.T - avgfeature0.T))
    # print(dist0)
    temp1 = np.dot((feature - avgfeature1), inversesigma)
    # print(temp0)
    dist1 = np.dot(temp1,(feature.T - avgfeature1.T))


    if maxormin == True:
        right_score.append(max(dist0[0],dist1[0]))
        # print(right_score)
    else:
        right_score.append(min(dist0[0],dist1[0]))

for feature in class2feature:
    temp0 = np.dot((feature - avgfeature0), inversesigma)
    # print(temp0)
    dist0 = np.dot(temp0,(feature.T - avgfeature0.T))
    # print(dist0)
    temp1 = np.dot((feature - avgfeature1), inversesigma)
    # print(temp0)
    dist1 = np.dot(temp1,(feature.T - avgfeature1.T))
    if maxormin == True:
        wrong_score.append(max(dist0[0],dist1[0]))
        # print(right_score)
    else:
        wrong_score.append(min(dist0[0],dist1[0]))

        
for feature in class3feature:
    temp0 = np.dot((feature - avgfeature0), inversesigma)
    # print(temp0)
    dist0 = np.dot(temp0,(feature.T - avgfeature0.T))
    # print(dist0)
    temp1 = np.dot((feature - avgfeature1), inversesigma)
    # print(temp0)
    dist1 = np.dot(temp1,(feature.T - avgfeature1.T))

    if maxormin == True:
        wrong_score.append(max(dist0[0],dist1[0]))
        # print(right_score)
    else:
        wrong_score.append(min(dist0[0],dist1[0]))
        
pos = np.array(right_score[:]).reshape((-1, 1))
neg = np.array(wrong_score[:]).reshape((-1, 1))

examples = np.squeeze(np.vstack((pos, neg)))
labels = np.zeros(len(examples), dtype=np.int32)
labels[:len(pos)] += 1
# print(len())
auroc = sk.roc_auc_score(labels, examples)
aupr = sk.average_precision_score(labels, examples)

fpr = fpr_and_fdr_at_recall(labels, examples, 0.95)
print(examples)
print(fpr)
print(auroc)
print(aupr)
