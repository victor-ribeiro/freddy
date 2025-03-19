import numpy as np
import heapq
import math
from itertools import batched
from functools import reduce
from sklearn.metrics import pairwise_distances
from sklearn.cluster import (
    BisectingKMeans,
    AgglomerativeClustering,
    FeatureAgglomeration,
)


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


def _n_cluster(dataset, k=1, alpha=1, max_iter=100, tol=10e-2, relevance=None):
    val = np.zeros(max_iter)
    cls = np.zeros(max_iter)
    for idx, n in enumerate(range(max_iter)):
        base = np.log(1 + alpha)
        sampler = BisectingKMeans(n_clusters=n + 2, init="k-means++")
        sampler.fit(dataset, sample_weight=np.abs(relevance))
        inertia = sampler.inertia_ + 10e-8
        if val[:idx].sum() == 0:
            val[idx] = np.log(1 + inertia * relevance.sum()) + base
            # val[idx] += np.exp(inertia - relevance.mean())
            cls[idx] = n + 2
            continue

        val[idx] = np.log(inertia * val[val > 0].sum() * relevance.sum()) + base
        # val[idx] += np.exp(inertia - relevance.mean())
        cls[idx] = n + 2

        # if abs(val[:idx].min() - val[idx]) < tol:
        if abs(val[:idx].max() - val[idx]) < tol:
            # import matplotlib.pyplot as plt

            # plt.plot(cls[cls > 0], val[val > 0])
            # plt.show()
            # exit()
            return sampler.cluster_centers_
    raise ValueError("Does not converge")


def kmeans_sampler(
    dataset, K, clusters, alpha=1, tol=10e-3, max_iter=500, relevance=None
):
    # clusters = _n_cluster(dataset, K, alpha, max_iter, tol, relevance)
    print(f"Found {len(clusters)} clusters, tol: {tol}")
    # dist = pairwise_distances(clusters, dataset, metric="sqeuclidean").sum(axis=0)
    dist = pairwise_distances(clusters, dataset).sum(axis=0)

    dist -= np.amax(dist)
    dist = np.abs(dist)
    sset = np.argsort(dist, kind="heapsort")[::-1]
    print(sset)
    return sset[:K]


def shannon_entropy(vector, epsilon=1e-10):
    abs_vector = np.abs(vector)  # Ensure non-negative
    total = abs_vector.sum(axis=1) + epsilon  # Avoid division by zero
    p = abs_vector / total.reshape(-1, 1)
    return (p * np.log2(1 + p)).sum(axis=1)


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
        self.sample_size = int(len(self.train_dataset) * self.args.train_frac)
        self.grad_freddy = grad_freddy
        self.selected = np.zeros(len(train_dataset))
        #
        n = len(train_dataset)
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
        self.model.eval()
        print(f"selecting subset on epoch {epoch}")
        self.epoch_selection.append(epoch)

        dataset = self.train_dataset.dataset
        dataset = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

        self.model.eval()
        feat = []
        lbl = []
        alpha = 2
        for data, target in dataset:
            pred = self.model.cpu()(data).detach().numpy()
            label = one_hot_coding(target, self.args.num_classes).cpu().detach().numpy()
            feat.append(pred - label)
            lbl.append(label)
        # feat = map(np.abs, feat)
        feat = np.vstack([*feat])
        tgt = np.vstack([*lbl])
        if epoch % 10 == 0:
            self.clusters = _n_cluster(
                feat, self.sample_size, alpha, 500, 10e-3, self._relevance_score
            )
        # sset, score = freddy(
        #     feat,
        #     # lambda_=self.lambda_,
        #     batch_size=256,
        #     K=self.sample_size,
        #     metric=self.args.freddy_similarity,
        #     alpha=self.args.alpha,
        #     relevance=self._relevance_score,
        # )
        # p = self.targets.sum(axis=0) / len(sset)

        sset = kmeans_sampler(
            feat,
            clusters=self.clusters,
            K=self.sample_size,
            relevance=self._relevance_score,
            alpha=alpha,
            tol=10e-2,
        )
        self.targets[epoch] += tgt[sset].sum(axis=0)
        p = self.targets[epoch].sum(axis=0) / len(sset)
        score = np.linalg.norm(feat, axis=1) * (-(p * np.log2(1 + p))).sum()
        score = (score.mean() - score) / score.std()
        self._relevance_score = score
        print(f"score {score}")

        print(f"selected ({len(sset)}) [{epoch}]: {self.targets[epoch]}")
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

        # if epoch % 5 == 0:
        if not epoch or (epoch + 1) % 7 == 0:
            # if not epoch or (epoch + 1) % 5 == 0:
            self._select_subset(epoch, len(self.train_loader) * epoch)
            self._update_train_loader_and_weights()
            # self.lambda_ = max(
            #     0.5, self.lambda_ + (self._relevance_score[self.subset].mean()) * lr
            # )

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
            # if self._relevance_score[data_idx].mean() < 0:
            #     self._relevance_score[data_idx] -= loss.item() * self.lr
            #     # self._relevance_score[data_idx] -= self.grad_norm * self.lr
            # else:
            #     self._relevance_score[data_idx] += loss.item() * self.lr
            # self._relevance_score[data_idx] += self.grad_norm * self.lr
            # self.model.eval()
            # with torch.no_grad():
            #     #     #### teste a rodar
            #     pred = self.model(data).cpu().detach().numpy()
            #     self._relevance_score[data_idx] = shannon_entropy(pred)
            #     # self._relevance_score[data_idx] = (
            #     #     self.train_criterion(pred, target).cpu().detach().numpy()
            #     # )

            # self.model.train()
            #### fim
        self.clusters -= self.clusters * self.grad_norm * self.lr
        self._val_epoch(epoch)

        train_loss /= len(self.train_loader)

        if self.hist:
            print(self.subset)
            self.hist[-1]["avg_importance"] = self._relevance_score[self.subset].mean()

        print(f"relative error: {self.cur_error}")
        if self.hist:
            self.hist[-1]["reaL_error"] = self.cur_error

        # self._relevance_score += self._relevance_score * lr
        self.cur_error = abs(self.cur_error - train_loss)
        # print(shannon_entropy(self.delta[self.subset].mean()).shape)
        # if not epoch or not (1.5 > self.cur_error > 10e-4):
        # self._relevance_score +=  (shannon_entropy(self.delta) + 10e-8)
        # print(self._relevance_score[self.subset])
        self.lr = self.lr_scheduler.get_last_lr()[0]
        # self.cur_error = self._relevance_score[self.subset].mean()
