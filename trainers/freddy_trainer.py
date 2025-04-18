import numpy as np
import heapq
import math
from itertools import batched
from sklearn.metrics import pairwise_distances
from sklearn.cluster import BisectingKMeans

import time
import torch

from .subset_trainer import *

REDUCE = {"mean": np.mean, "sum": np.sum}
__all__ = ["METRICS"]


METRICS = {}


def one_hot_coding(target, classes):
    n = len(target)
    coded = np.zeros((n, classes))
    coded[np.arange(n), target] = 1
    return torch.Tensor(coded)


def _register(fn):
    name = fn.__name__
    METRICS[name] = fn
    __all__.append(name)
    return fn


@_register
def pdist(dataset, metric="euclidean", batch_size=1):
    return pairwise_distances(dataset, metric=metric)


@_register
def codist(dataset, batch_size=1):
    d = pairwise_distances(dataset, dataset, metric="cosine")
    return 1 - d


@_register
def kdist(dataset, centroids):
    d = pairwise_distances(dataset, centroids)
    return d.max() - d


@_register
def similarity(dataset, metric="euclidean", batch_size=1):
    d = pdist(dataset, metric)
    return d.max(axis=1) - d


def base_inc(alpha=1):
    alpha = abs(alpha)
    return math.log(1 + alpha)


def utility_score(e, sset, /, acc=0, alpha=0.1, beta=1.1):
    gamma = (alpha + beta) / 2
    norm = 1 / base_inc(alpha)
    argmax = np.maximum(e, sset)
    f_norm = alpha / (sset.sum() + acc + 1)
    # f_norm = min(0, f_norm)
    util = norm * math.log(1 + (argmax.sum() + acc) * f_norm)
    return util + (math.log(1 + (sset.sum() + acc)) * beta)


class Queue(list):
    def __init__(self, *iterable):
        super().__init__(*iterable)
        heapq._heapify_max(self)

    def append(self, item: "Any"):
        super().append(item)
        heapq._siftdown_max(self, 0, len(self) - 1)

    def pop(self, index=-1):
        el = super().pop(index)
        if not self:
            return el
        val, self[0] = self[0], el
        heapq._siftup_max(self, 0)
        return val

    @property
    def head(self):
        return self.pop()

    def push(self, idx, score):
        item = (idx, score)
        self.append(item)


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


def entropy(x):
    x = np.abs(x)
    total = x.sum()
    p = x / total
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


def kmeans_sampler(
    dataset, K, clusters, alpha=1, tol=10e-3, max_iter=500, relevance=None
):
    print(f"Found {len(clusters)} clusters, tol: {tol}")
    dist = pairwise_distances(dataset, clusters)

    dist -= np.amax(dist, axis=0)
    dist = np.abs(dist).sum(axis=1)
    sset = np.argsort(dist, kind="heapsort")[::-1]
    return sset[:K]


def pmi_kmeans_sampler(dataset, K, alpha=1, tol=10e-3, max_iter=500, importance=None):
    clusters = _n_cluster(dataset, alpha, max_iter, tol)
    print(f"Found {len(clusters)} clusters, tol: {tol}")
    dist = pairwise_distances(clusters, dataset).sum(axis=0)
    h_pc = entropy(np.dot(dataset, clusters.T))
    h_c = entropy(clusters)
    h_p = entropy(dataset)
    pmi = (h_p - h_c) / h_pc

    pmi = dist * pmi * importance
    sset = np.argsort(pmi, kind="heapsort")[::-1]

    return sset[:K]


def freddy(
    dataset,
    base_inc=base_inc,
    alpha=0.15,
    metric="similarity",
    K=1,
    batch_size=128,
    beta=0.75,
    return_vals=False,
    importance=None,
):
    # basic config
    base_inc = base_inc(alpha)
    idx = np.arange(len(dataset))
    idx = np.random.permutation(idx)
    dataset = dataset[idx]
    q = Queue()
    sset = []
    vals = []
    argmax = 0
    inc = 0
    for ds, V in zip(
        batched(dataset, batch_size),
        batched(idx, batch_size),
    ):
        D = METRICS[metric](ds, batch_size=batch_size)
        size = len(D)
        localmax = np.amax(D, axis=1)
        argmax += localmax.sum()
        _ = [q.push(base_inc, i) for i in zip(V, range(size))]
        while q and len(sset) < K:
            score, idx_s = q.head
            s = D[:, idx_s[1]]
            score_s = utility_score(s, localmax, acc=argmax, alpha=alpha, beta=beta)
            inc = score_s - score
            if (inc < 0) or (not q):
                break
            score_t, idx_t = q.head
            if inc > score_t:
                # score = utility_score(s, localmax, acc=argmax, alpha=alpha, beta=beta)
                localmax = np.maximum(localmax, s)
                sset.append(idx_s[0])
                vals.append(score_s)
            else:
                q.push(inc, idx_s)
            q.push(score_t, idx_t)
    np.random.shuffle(sset)
    if return_vals:
        return np.array(vals), sset
    return np.array(sset)


class FreddyTrainer(SubsetTrainer):
    def __init__(
        self,
        args,
        model,
        train_dataset,
        val_loader,
        train_weights=None,
        grad_freddy=False,
    ):
        super().__init__(args, model, train_dataset, val_loader, train_weights)
        self.grad_freddy = grad_freddy
        self.selected = np.zeros(len(train_dataset))
        #
        n = len(train_dataset)
        self.train_frac = 1
        self.min_train_frac = self.args.train_frac
        self.sample_size = int(len(self.train_dataset) * self.min_train_frac)
        self.epoch_selection = []
        self.delta = np.random.normal(0, 1, (n, self.args.num_classes))
        self._relevance_score = np.ones(n)
        self.select_flag = True
        self.cur_error = 10e-7
        self.lambda_ = 0.5
        self.lr = 0.1
        self.targets = np.zeros((self.args.epochs, self.args.num_classes))
        self.clusters = None

    def _select_subset(self, epoch, training_step):

        print(f"selecting subset on epoch {epoch}")
        self.epoch_selection.append(epoch)

        self.train_val_loader = DataLoader(
            # Subset(self.train_dataset, indices=self.subset),
            self.val_loader.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        tgt = [x[1] for x in self.train_dataset.dataset]
        self._get_train_output()
        tgt = one_hot_coding(tgt, self.args.num_classes).cpu().detach().numpy()
        sset = freddy(
            tgt - self.train_output,
            # lambda_=self.lambda_,
            batch_size=512,
            K=self.sample_size,
            # K=int(self.train_frac * len(self.train_dataset)),
            metric=self.args.freddy_similarity,
            alpha=self.args.alpha,
            importance=self._relevance_score,
        )

        ##########################################
        # sset = pmi_kmeans_sampler(
        #     tgt - self.train_softmax,
        #     K=int(self.args.train_frac * len(self.train_dataset)),
        #     importance=self._relevance_score,
        # )
        ##########################################
        self.targets[epoch] += tgt[sset].sum(axis=0)

        print(f"selected ({len(sset)}) [{epoch}]: {self.targets[epoch].astype(int)}")
        # print(np.isin(sset, self.subset).sum())
        # print(np.isin(self.subset, sset).sum())
        self.subset = sset
        self.selected[sset] += 1
        self.train_checkpoint["selected"] = self.selected
        self.train_checkpoint["importance"] = self._relevance_score
        self.train_checkpoint["epoch_selection"] = self.epoch_selection
        self.train_checkpoint["class_histogram"] = self.targets
        self.subset_weights = np.ones(self.sample_size)
        self.train_loader = DataLoader(
            Subset(self.train_dataset, self.subset),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.model.train()

    def _train_epoch(self, epoch):
        self.model.train()
        self._reset_metrics()

        # if (epoch + 1) % 19 == 0:
        # if (epoch + 1) % 20 == 0:
        if epoch % 20 == 0:
            self.train_frac = max(self.min_train_frac, self.train_frac - 0.2)
            self.sample_size = int(len(self.train_dataset) * self.train_frac)
            # print(self.sample_size)
            ##############################
            selection_init = time.perf_counter()
            self._select_subset(epoch, len(self.train_loader) * epoch)
            selection_end = time.perf_counter()
            self.select_time[epoch] = selection_end - selection_init
            ##############################
            self._update_train_loader_and_weights()
            self.train_checkpoint["selection_time"] = self.select_time

        data_start = time.time()
        pbar = tqdm(
            enumerate(self.train_loader), total=len(self.train_loader), file=sys.stdout
        )
        train_loss = 0
        for batch_idx, (data, target, data_idx) in pbar:
            data, target = data.to(self.args.device), target.to(self.args.device)
            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)

            self.optimizer.zero_grad()
            loss, train_acc = self._forward_and_backward(data, target, data_idx)
            train_loss += loss.item()
            data_start = time.time()
            # update progress bar
            pbar.set_description(
                "{}: {}/{} [{}/{} ({:.0f}%)] Loss: {:.6f} Acc: {:.6f}".format(
                    self.__class__.__name__,
                    epoch,
                    self.args.epochs,
                    batch_idx * self.args.batch_size + len(data),
                    len(self.train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(self.train_loader),
                    loss.item(),
                    train_acc,
                )
            )
        self._val_epoch(epoch)

        train_loss /= len(self.train_loader)

        if self.hist:
            print(self.subset)
            self.hist[-1]["avg_importance"] = self._relevance_score[self.subset].mean()

        print(f"relative error: {self.cur_error}")
        if self.hist:
            self.hist[-1]["reaL_error"] = self.cur_error

        self.cur_error = abs(self.cur_error - train_loss)
        self.lr = self.lr_scheduler.get_last_lr()[0]
