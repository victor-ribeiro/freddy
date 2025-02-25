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
    importance=None,
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
        _ = [q.push(base_inc * importance[i[0]], i) for i in zip(V, range(size))]
        while q and len(sset) < K:
            score, idx_s = q.head
            s = D[:, idx_s[1]]
            score_s = (
                utility_score(s, localmax, acc=argmax, alpha=alpha, beta=beta)
            ) * importance[idx_s[0]]
            inc = score_s - score
            if (inc < 0) or (not q):
                break
            score_t, idx_t = q.head
            if inc > score_t:
                score = (
                    utility_score(s, localmax, acc=argmax, alpha=alpha, beta=beta)
                ) * importance[idx_s[0]]

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
        n = len(train_dataset)
        self.epoch_selection = []
        self.delta = np.ones((n, self.args.num_classes))
        self.select_flag = True
        self.cur_error = 1

    @property
    def _relevance_score(self):
        score = np.linalg.norm(self.delta, axis=1)
        return score**-1

    def _select_subset(self, epoch, training_step):
        self.model.eval()
        print(f"selecting subset on epoch {epoch}")
        self.epoch_selection.append(epoch)
        if epoch % 5:
            dataset = self.train_dataset.dataset
            dataset = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
            )
            with torch.no_grad():
                delta = map(self._update_delta, dataset)
                delta = map(lambda x: x[1] - x[0], delta)
                self.delta = np.vstack([*delta])

        sset = freddy(
            delta,
            K=self.sample_size,
            metric=self.args.freddy_similarity,
            alpha=self.args.alpha,
            beta=self.args.beta,
            importance=self._relevance_score,
        )
        self.subset = sset
        self.selected[sset] += 1
        self.train_checkpoint["selected"] = self.selected
        self.train_checkpoint["importance"] = self._relevance_score
        self.train_checkpoint["epoch_selection"] = self.epoch_selection
        self.subset_weights = np.ones(self.sample_size)
        self.model.train()

    def _train_epoch(self, epoch):
        self.model.train()
        self._reset_metrics()

        data_start = time.time()
        pbar = tqdm(
            enumerate(self.train_loader), total=len(self.train_loader), file=sys.stdout
        )
        grad = []
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
        self._error_func()
        self._val_epoch(epoch)

        # if self.args.cache_dataset and self.args.clean_cache_iteration:
        #     self.train_dataset.clean()
        #     self._update_train_loader_and_weights()

        # if self.hist:
        #     self.hist[-1]["avg_importance"] = self.importance_score[self.subset].mean()

        # lr = self.lr_scheduler.get_last_lr()[0]
        # # modules = [*self.model.to(self.args.device).modules()]
        # # grad2 = modules[-1]
        # # grad2 = grad2.weight.grad.data
        # # # error = (grad2 - grad1).norm(2).item() * lr
        # # error = ((grad2 - grad1) * lr).norm(2).item()
        # error = abs(self.importance_score[self.subset].mean() - rel)
        # print(f"relative error: {abs(self.cur_error - error)}")
        # print(f"learning rate: {lr}")

        # # if not epoch or abs(self.cur_error - error) < lr:
        # if not epoch or abs(self.cur_error - error) > lr:
        #     # if not epoch or np.isclose(self.cur_error - error, lr):
        #     # if not epoch or np.isclose(self.cur_error - error, lr, atol=10e-2):
        #     self._select_subset(epoch, len(self.train_loader) * epoch)
        # if self.hist:
        #     self.hist[-1]["reaL_error"] = error
        # self.cur_error = error

    # def _forward_and_backward(self, data, target, data_idx):
    #     out = super()._forward_and_backward(data, target, data_idx)
    #     e = self._error_func(data, target)

    def _error_func(self):
        grad = [g.grad.clone() for g in self.model.parameters() if g.grad is not None]
        hess = []
        for g in grad:
            if g.grad is None:
                continue
            self.model.zero_grad()
            h = torch.autograd.grad(
                g,
                self.model.parameters(),
                grad_outputs=torch.ones_like(g),
                # create_graph=True,
                # retain_graph=True,
                # allow_unused=False,
                # materialize_grads=True,
            )
            hess.append(h)

        print(hess)
        # exit()

    def _update_delta(self, train_data):
        data, _ = train_data
        self.model.eval()
        e = torch.normal(0, 1, size=data.shape).to(self.args.device)
        with torch.no_grad():
            data = data.to(self.args.device)
            loss = self.model(data).softmax(dim=1)
            delta_loss = self.model(data + e).softmax(dim=1)
        return loss - delta_loss

    # def train(self):
    #     self._select_subset(0, 0)
    #     return super().train()
