import numpy as np
import heapq
import math
from itertools import batched
from sklearn.metrics import pairwise_distances

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
    idx = np.random.permutation(idx)
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


@_register
def grad_freddy(
    dataset,
    base_inc=base_inc,
    alpha=0.15,
    metric="similarity",
    K=1,
    batch_size=32,
    beta=0.75,
    return_vals=False,
):
    # basic config
    base_inc = base_inc(alpha)
    idx = np.arange(len(dataset))
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
        # localmax = np.median(D, axis=0)
        localmax = np.amax(D, axis=1)
        argmax += localmax.sum()
        _ = [q.push(base_inc, i) for i in zip(V, range(size))]
        while q and len(sset) < K:
            score, idx_s = q.head
            s = D[:, idx_s[1]]
            s = score + (s * inc) + (np.gradient(s) * (inc**2) * 0.5)
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
    # np.random.shuffle(sset)
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
        # self.importance_score = np.zeros(len(train_dataset))
        self.select_flag = True
        self.cur_error = 1

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

            feat = map(lambda x: x[1] - x[0], feat)
            feat = np.vstack([*feat])

        if self.grad_freddy:
            sset = grad_freddy(
                feat,
                K=self.sample_size,
                metric=self.args.freddy_similarity,
                alpha=self.args.alpha,
                beta=self.args.beta,
            )
        else:
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
        # self.subset_weights = self.importance_score[self.subset]

        self.select_flag = False

    def _train_epoch(self, epoch):
        if not epoch:
            self._select_subset(epoch, len(self.train_loader) * epoch)
        self.model.train()
        self._reset_metrics()

        data_start = time.time()
        # use tqdm to display a smart progress bar
        try:
            grad1 = [*self.model.to(self.args.device).modules()]
            grad1 = grad1.pop()
            grad1 = grad1.weight.grad.data.norm(2).item()
        except:
            grad1 = 0
        importance = self.importance_score.mean()

        pbar = tqdm(
            enumerate(self.train_loader), total=len(self.train_loader), file=sys.stdout
        )
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

        if self.args.cache_dataset and self.args.clean_cache_iteration:
            self.train_dataset.clean()
            self._update_train_loader_and_weights()

        if self.hist:
            self.hist[-1]["avg_importance"] = self.importance_score[self.subset].mean()

        grad2 = [*self.model.to(self.args.device).modules()]
        grad2 = grad2.pop()
        grad2 = grad2.weight.grad.data.norm(2).item()
        # # error = abs(grad2 - grad1) / self.importance_score[self.subset].mean()
        # error = grad2 - grad1 / self.importance_score[self.subset].mean()
        # error = self.importance_score[self.subset].mean() / (
        #     self.importance_score.mean() - importance
        # )
        error = (self.importance_score[self.subset].mean() - importance) / (
            grad2 - grad1
        )
        error = abs(error)
        # error = np.log(error)
        print(f"relative error [{abs(self.cur_error-error)}]")
        # if not epoch or abs(self.cur_error - error) < 10e-2:
        # if not epoch or abs(self.cur_error / error) > 1:
        # if not epoch or abs(self.cur_error - error) < 10e-2:
        if abs(self.cur_error - error) < 10e-2:
            self._select_subset(epoch, len(self.train_loader) * epoch)
        self.cur_error = error

    def _forward_and_backward(self, data, target, data_idx):
        with torch.no_grad():
            pred = self.model.to(self.args.device)(data)
            loss_t1 = self.train_criterion(pred, target).cpu().detach().numpy()

        loss, train_acc = super()._forward_and_backward(data, target, data_idx)
        with torch.no_grad():
            pred = self.model.to(self.args.device)(data)
            loss_t2 = self.train_criterion(pred, target).cpu().detach().numpy()

        # importance = np.abs(loss_t2 - loss_t1)
        importance = np.abs(loss_t2 + loss_t1)
        self.importance_score[data_idx] += importance

        return loss, train_acc

    # def train(self):
    #     self._select_subset(0, 0)
    #     return super().train()
