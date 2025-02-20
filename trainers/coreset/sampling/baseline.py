import numpy as np
import multiprocessing as mp


from itertools import batched

from sklearn.metrics import pairwise_distances


# resolver esse import aqui
from cra.craig.lazy_greedy import FacilityLocation, lazy_greedy_heap


N_JOBS = mp.cpu_count()


def random_sampler(data, K):
    size = len(data)
    rng = np.random.default_rng()
    sset = rng.integers(0, size, size=K, dtype=int)
    return sset


def craig_baseline(data, K, b_size=32):
    features = data.astype(np.single)
    V = np.arange(len(features), dtype=int).reshape(-1, 1)
    start = 0
    end = start + b_size
    sset = []
    n_jobs = int(N_JOBS // 2)
    for ds in batched(features, b_size):
        ds = np.array(ds)
        # D = pairwise_distances(features, ds, metric="euclidean", n_jobs=n_jobs)
        D = pairwise_distances(ds, features, metric="euclidean", n_jobs=n_jobs)
        v = V[start:end]
        D = D.max() - D
        B = int(len(D) * (K / len(features)))
        locator = FacilityLocation(D=D, V=v)
        sset_idx, *_ = lazy_greedy_heap(F=locator, V=v, B=B)
        sset_idx = np.array(sset_idx, dtype=int).reshape(1, -1)[0]
        sset.append(sset_idx)
        start += b_size
        end += b_size
    sset = np.hstack(sset)
    return sset
