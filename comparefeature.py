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


# class0feature = np.load("class0feature.npy")
# class1feature = np.load("class1feature.npy")
# class2feature = np.load("class2feature.npy")
# class3feature = np.load("class3feature.npy")

# class5feature = np.load("class5feature.npy")
# class6feature = np.load("class6feature.npy")
# class7feature = np.load("class7feature.npy")
# class8feature = np.load("class8feature.npy")
# class9feature = np.load("class9feature.npy")


class0feature = np.load("class0feature.npy")
class6feature = np.load("class6feature.npy")
# print(class6feature)
# exit()
# for feature in class1feature:
#     print(feature.shape)
# print(class1feature[1].shape)
# vec1 = class1feature[1]
# vec2 = class1feature[2]


# dist1 = np.linalg.norm(vec1 - vec2)

# print(dist1)

# vec3 = class6feature[2]

# dist1 = np.linalg.norm(vec1 - vec3)

# print(dist1)
i = 0
listlen = len(class0feature)
# print(listlen)
avgfeature0 = np.zeros((1,128))

for j in range(listlen):
    if(j != i):
        avgfeature0 += class0feature[j]
avgfeature0 = avgfeature0/(listlen-1)

listlen = len(class0feature)
dist0 = 0
for feature in class0feature:
    vec1 = feature
    avgfeature = np.zeros((1,128))
    for j in range(listlen):
        if(j!=i):
            avgfeature += class0feature[j]
    avgfeature = avgfeature / (listlen - 1)

    # dist = np.linalg.norm(vec1 - avgfeature)
    dist = metrics.normalized_mutual_info_score(vec1[0],avgfeature[0])
    dist0 += dist
    # print(dist)
    i += 1

print(dist0/ (listlen - 1))



dist0 = 0
# print(len(class6feature))
# exit()
for feature in class6feature:
    vec1 = feature
    dist = metrics.normalized_mutual_info_score(vec1[0],avgfeature0[0])
    dist0 += dist
    # print(dist)

print(dist0/ len(class6feature))

exit()


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



# avgfeature0 = abs (avgfeature0-np.mean(avgfeature0))
# avgfeature1 = abs(avgfeature1-np.mean(avgfeature1))
# avgfeature2 = abs (avgfeature2-np.mean(avgfeature2))


maxormin = False
method = 'euclidean'

diffdist = 0
listlen = len(class6feature)
# print(listlen)
rightcount = 0
wrongcount = 0
right_score = []
wrong_score = []
for feature in class0feature:
    # dist0 = np.linalg.norm(feature - avgfeature0)
    # dist1 = np.linalg.norm(feature - avgfeature1)
    # dist2 = np.linalg.norm(feature - avgfeature2)
    # dist3 = np.linalg.norm(feature - avgfeature3)
    # print(min(dist0,dist1,dist2,dist3))

    # feature=abs(feature-np.mean(feature))
    # y_=y-np.mean(y)

    # right_score.append(min(dist0,dist1,dist2,dist3))
    # print(dist0)
    # dist0=  np.dot(feature,avgfeature0.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature0))
    # dist1=  np.dot(feature,avgfeature1.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature1))
    # dist2=  np.dot(feature,avgfeature2.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature2))

    # dist0 = np.dot(feature,avgfeature0.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature0))
    # dist1 = np.dot(feature,avgfeature1.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature1))
    # dist2 = np.dot(feature,avgfeature2.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature2))
    dist0 = pdist(np.vstack([feature,avgfeature0]), method)
    dist1 = pdist(np.vstack([feature,avgfeature1]), method)
    dist2 = pdist(np.vstack([feature,avgfeature2]), method)


    if maxormin == True:
        right_score.append(max(dist0[0],dist1[0],dist2[0]))
        # print(right_score)
    else:
        right_score.append(min(dist0[0],dist1[0],dist2[0]))
    # print(min(dist0,dist1,dist2))


for feature in class1feature:
    # dist0 = np.linalg.norm(feature - avgfeature0)
    # dist1 = np.linalg.norm(feature - avgfeature1)
    # dist2 = np.linalg.norm(feature - avgfeature2)
    # dist3 = np.linalg.norm(feature - avgfeature3)
    # # print(min(dist0,dist1,dist2,dist3))
    # right_score.append(min(dist0,dist1,dist2,dist3))
    # if(dist0 < 1.4 or dist1< 1.4 or dist2< 1.4 or dist3< 1.4):
    #     wrongcount += 1
    #     rightcount += 1
    # feature=abs(feature-np.mean(feature))
    # dist0 = pdist(np.vstack([feature,avgfeature0]), 'cosine')
    # dist1 = pdist(np.vstack([feature,avgfeature1]), 'cosine')
    # dist2 = pdist(np.vstack([feature,avgfeature2]), 'cosine')
    dist0 = np.dot(feature,avgfeature0.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature0))
    dist1 = np.dot(feature,avgfeature1.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature1))
    dist2 = np.dot(feature,avgfeature2.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature2))
    dist0 = pdist(np.vstack([feature,avgfeature0]), method)
    dist1 = pdist(np.vstack([feature,avgfeature1]), method)
    dist2 = pdist(np.vstack([feature,avgfeature2]), method)

    if maxormin == True:
        right_score.append(max(dist0[0],dist1[0],dist2[0]))
    else:
        right_score.append(min(dist0[0],dist1[0],dist2[0]))


for feature in class2feature:
    # dist0 = np.linalg.norm(feature - avgfeature0)
    # dist1 = np.linalg.norm(feature - avgfeature1)
    # dist2 = np.linalg.norm(feature - avgfeature2)
    # dist3 = np.linalg.norm(feature - avgfeature3)
    # # print(min(dist0,dist1,dist2,dist3))
    # right_score.append(min(dist0,dist1,dist2,dist3))
    # if(dist0 < 1.4 or dist1< 1.4 or dist2< 1.4 or dist3< 1.4):
    #     wrongcount += 1
    #     rightcount += 1
    # feature=abs(feature-np.mean(feature))

    dist0 = np.dot(feature,avgfeature0.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature0))
    dist1 = np.dot(feature,avgfeature1.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature1))
    dist2 = np.dot(feature,avgfeature2.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature2))
    dist0 = pdist(np.vstack([feature,avgfeature0]), method)
    dist1 = pdist(np.vstack([feature,avgfeature1]), method)
    dist2 = pdist(np.vstack([feature,avgfeature2]), method)

    if maxormin == True:
        right_score.append(max(dist0[0],dist1[0],dist2[0]))
    else:
        right_score.append(min(dist0[0],dist1[0],dist2[0]))




for feature in class6feature:
    # dist0 = np.linalg.norm(feature - avgfeature0)

    # dist1 = np.linalg.norm(feature - avgfeature1)
    # dist2 = np.linalg.norm(feature - avgfeature2)
    # dist3 = np.linalg.norm(feature - avgfeature3)

    dist0 = np.dot(feature,avgfeature0.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature0))
    dist1 = np.dot(feature,avgfeature1.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature1))
    dist2 = np.dot(feature,avgfeature2.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature2))
    dist0 = pdist(np.vstack([feature,avgfeature0]), method)
    dist1 = pdist(np.vstack([feature,avgfeature1]), method)
    dist2 = pdist(np.vstack([feature,avgfeature2]), method)
    # print(dist0)
    # print(dist1)
    # print(dist2)
    # print("-------------------")
    if maxormin == True:
        wrong_score.append(max(dist0[0],dist1[0],dist2[0]))
    else:
        wrong_score.append(min(dist0[0],dist1[0],dist2[0]))
    # print(min(dist0,dist1,dist2))

    # print(wrong_score)

    # print(min(dist0,dist1,dist2,dist3))
    # wrong_score.append(min(dist0,dist1,dist2,dist3))
    # print(dist0)

for feature in class7feature:
    # dist0 = np.linalg.norm(feature - avgfeature0)
    # dist1 = np.linalg.norm(feature - avgfeature1)
    # dist2 = np.linalg.norm(feature - avgfeature2)
    # dist3 = np.linalg.norm(feature - avgfeature3)
    # # print(min(dist0,dist1,dist2,dist3))
    # wrong_score.append(min(dist0,dist1,dist2,dist3))
    dist0 = np.dot(feature,avgfeature0.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature0))
    dist1 = np.dot(feature,avgfeature1.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature1))
    dist2 = np.dot(feature,avgfeature2.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature2))
    dist0 = pdist(np.vstack([feature,avgfeature0]), method)
    dist1 = pdist(np.vstack([feature,avgfeature1]), method)
    dist2 = pdist(np.vstack([feature,avgfeature2]), method)

    print(dist0)
    print(dist1)
    print(dist2)
    print("-------------------")

    if maxormin == True:
        wrong_score.append(max(dist0[0],dist1[0],dist2[0]))
    else:
        wrong_score.append(min(dist0[0],dist1[0],dist2[0]))




    # if(dist0 < 1.4 or dist1< 1.4 or dist2< 1.4 or dist3< 1.4):
    #     wrongcount += 1
    #     rightcount += 1

for feature in class8feature:
    # dist0 = np.linalg.norm(feature - avgfeature0)
    # dist1 = np.linalg.norm(feature - avgfeature1)
    # dist2 = np.linalg.norm(feature - avgfeature2)
    # dist3 = np.linalg.norm(feature - avgfeature3)
    # print(min(dist0,dist1,dist2,dist3))
    # wrong_score.append(min(dist0,dist1,dist2,dist3))
    dist0 = np.dot(feature,avgfeature0.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature0))
    dist1 = np.dot(feature,avgfeature1.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature1))
    dist2 = np.dot(feature,avgfeature2.T)/(np.linalg.norm(feature)*np.linalg.norm(avgfeature2))
    dist0 = pdist(np.vstack([feature,avgfeature0]), method)
    dist1 = pdist(np.vstack([feature,avgfeature1]), method)
    dist2 = pdist(np.vstack([feature,avgfeature2]), method)

    if maxormin == True:
        wrong_score.append(max(dist0[0],dist1[0],dist2[0]))
    else:
        wrong_score.append(min(dist0[0],dist1[0],dist2[0]))





# print(wrongcount)
# print(rightcount)
# print(diffdist/ listlen)

# print(len(right_score))
pos = np.array(right_score[:]).reshape((-1, 1))
neg = np.array(wrong_score[:]).reshape((-1, 1))

examples = np.squeeze(np.vstack((pos, neg)))
labels = np.zeros(len(examples), dtype=np.int32)
labels[:len(pos)] += 1
# print(len())
auroc = sk.roc_auc_score(labels, examples)
aupr = sk.average_precision_score(labels, examples)

fpr = fpr_and_fdr_at_recall(labels, examples, 0.85)
# print(examples)
print(fpr)
print(auroc)
print(aupr)
