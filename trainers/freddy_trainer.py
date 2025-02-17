import numpy as np
from functools import partial, reduce
from itertools import batched
from sklearn.metrics import pairwise_distances

import torch
import math
import heapq

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


@_register
def freddy(
    dataset,
    base_inc=base_inc,
    alpha=0.15,
    metric="similarity",
    K=1,
    batch_size=128,
    beta=0.75,
    return_vals=False,
):
    # basic config
    base_inc = base_inc(alpha)
    idx = np.arange(len(dataset))
    # idx = np.random.permutation(idx)
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
                score = utility_score(s, localmax, acc=argmax, alpha=alpha, beta=beta)
                localmax = np.maximum(localmax, s)
                sset.append(idx_s[0])
                vals.append(score)
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
        self.sample_size = int(len(self.train_dataset) * self.args.train_frac)
        self.grad_freddy = grad_freddy
        self.selected = np.zeros(len(train_dataset))
        #
        self.epoch_selection = []
        self.importance_score = np.ones(len(train_dataset))
        self.select_flag = True
        self.cur_error = 0

    def _select_subset(self, epoch, training_step):
        print(f"selecting subset on epoch {epoch}")
        if self.epoch_selection:
            print(f"RESELECTING: {self.epoch_selection[-1]}")
        dataset = self.train_dataset.dataset
        dataset = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

        with torch.no_grad():

            feat = map(
                lambda x: (
                    self.model.cpu()(x[0]).detach().numpy(),
                    one_hot_coding(x[1].cpu().detach().numpy(), self.args.num_classes),
                ),
                dataset,
            )

            # feat = map(lambda x: 0.1 * (x[1] - x[0]) ** 2, feat)
            feat = map(lambda x: x[1] - (x[0] * self.cur_error * self.args.alpha), feat)
            feat = np.vstack([*feat])
            self.args.alpha -= self.cur_error * self.args.alpha
            # feat = feat - self.importance_score.reshape(-1, 1)
            # feat = feat * (self.importance_score.reshape(-1, 1) + self.cur_error)

        sset = freddy(
            feat,
            K=self.sample_size,
            metric=self.args.freddy_similarity,
            alpha=self.args.alpha,
            beta=self.args.beta,
        )
        self.subset = sset
        self.selected[sset] += 1
        self.train_checkpoint["selected"] = self.selected
        self.train_checkpoint["importance"] = self.importance_score
        self.train_checkpoint["epoch_selection"] = self.epoch_selection
        self.subset_weights = np.ones(self.sample_size)

        self.select_flag = False

    def _train_epoch(self, epoch):
        if not epoch:
            self._select_subset(epoch, len(self.train_loader) * epoch)
        self.model.train()
        self._reset_metrics()

        data_start = time.time()
        # use tqdm to display a smart progress bar

        pbar = tqdm(
            enumerate(self.train_loader), total=len(self.train_loader), file=sys.stdout
        )

        with torch.no_grad():
            pred = map(
                lambda x: self.model.cpu()(x[0]).detach().numpy(),
                self.train_loader,
            )
            pred = map(partial(np.argmax, axis=1), pred)
            pred = map(lambda x: one_hot_coding(x, classes=self.args.num_classes), pred)
            tgt = map(lambda x: one_hot_coding(x[1], classes=10), self.train_loader)
            importance = map(self.train_criterion, pred, tgt)
            # importance = reduce(lambda a, b: a + b, importance)

        for batch_idx, (data, target, data_idx) in pbar:

            # load data to device and record data loading time
            data, target = data.to(self.args.device), target.to(self.args.device)
            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)

            self.optimizer.zero_grad()

            # train model with the current batch and record forward and backward time
            loss, train_acc = self._forward_and_backward(data, target, data_idx)

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
        with torch.no_grad():
            pred = map(
                lambda x: self.model.cpu()(x[0]).detach().numpy(),
                self.train_loader,
            )
            pred = map(partial(np.argmax, axis=1), pred)
            pred = map(lambda x: one_hot_coding(x, classes=self.args.num_classes), pred)
            tgt = map(lambda x: one_hot_coding(x[1], classes=10), self.train_loader)
            importance2 = map(self.train_criterion, pred, tgt)
            importance = map(lambda a, b: b - a, importance, importance2)
            print(next(importance))
            exit()

        if self.args.cache_dataset and self.args.clean_cache_iteration:
            self.train_dataset.clean()
            self._update_train_loader_and_weights()

        if self.hist:
            self.hist[-1]["avg_importance"] = self.importance_score[self.subset].mean()
            # self.hist[-1]["avg_importance"] = self.importance_score.mean()
        # error = (self.importance_score - importance).sum()
        # error = np.log(error)
        # error = (self.importance_score[self.subset].mean()) - local_importance / (
        #     self.importance_score.mean() - importance
        # )

        # print(f"relative error [{abs(error)}]")
        # print(f"relative error [{error}]")
        print(f"relative error [{self.cur_error}]")

        self.cur_error = (self.importance_score[self.subset] - importance).mean()
        self.cur_error = abs(self.cur_error)
        # if abs(self.cur_error - error) < 10e-3:
        if self.cur_error < 10e-3:
            self._select_subset(epoch, len(self.train_loader) * epoch)
        if self.hist:
            self.hist[-1]["reaL_error"] = self.cur_error

    # def _forward_and_backward(self, data, target, data_idx):
    #     with torch.no_grad():
    #         pred = self.model.to(self.args.device)(data)
    #         loss_t1 = self.train_criterion(pred, target).cpu().detach().numpy()

    #     loss, train_acc = super()._forward_and_backward(data, target, data_idx)
    #     with torch.no_grad():
    #         pred = self.model.to(self.args.device)(data)
    #         loss_t2 = self.train_criterion(pred, target).cpu().detach().numpy()

    #     importance = (loss_t2 - loss_t1) * self.train_loss.avg
    #     # self.importance_score[data_idx] = importance
    #     self.importance_score[data_idx] += importance
    #     # self.importance_score[data_idx] -= importance

    #     return loss, train_acc
