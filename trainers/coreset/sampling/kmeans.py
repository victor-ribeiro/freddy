import numpy as np
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs, make_classification
from functools import wraps


@lambda _: _()
def dataset():
    return make_classification(
        n_samples=10000,
        n_classes=3,
        n_informative=3,
        n_clusters_per_class=2,
        weights=[0.5, 0.3, 0.2],
    )


def _n_cluster(dataset, alpha=1, max_iter=100, tol=10e-2):
    val = np.zeros(max_iter)
    base = np.log(1 + alpha)
    for idx, n in enumerate(range(max_iter)):
        # print(val)
        sampler = BisectingKMeans(n_clusters=n + 2)
        sampler.fit(dataset)
        if val[:idx].sum() == 0:

            val[idx] = np.log(1 + sampler.inertia_ * alpha / base)
            continue

        val[idx] = np.log(1 + sampler.inertia_ * alpha / val[val > 0].max() / base)

        if abs(val[:idx].min() - val[idx]) < tol:
            return sampler.cluster_centers_
    # return sampler.cluster_centers_
    return ValueError("Does not converge")


# def kmeans_sampler(dataset, K, alpha=1, tol=10e-3, max_iter=300):
def kmeans_sampler(dataset, K, alpha=1, tol=10e-3, max_iter=300):
    clusters = _n_cluster(dataset, alpha, max_iter, tol)
    dist = pairwise_distances(clusters, dataset).mean(axis=0)
    dist -= np.max(dist)
    dist = np.abs(dist)[::-1]
    sset = np.argsort(dist, kind="heapsort")

    return sset[:K]


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.metrics import classification_report
#     from sklearn.model_selection import train_test_split

#     ds, lbl = dataset

#     X_train, X_test, y_train, y_test = train_test_split(ds, lbl, test_size=0.2)

#     model = XGBClassifier()
#     model.fit(X_train, y_train)

#     sset = kmeans_sampler(100, tol=10e-3)(X_train)

#     k_model = XGBClassifier()
#     k_model.fit(X_train[sset], y_train[sset])

#     print(classification_report(y_test, model.predict(X_test)))
#     print(classification_report(y_test, k_model.predict(X_test)))
