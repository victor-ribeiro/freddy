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


def linear_selector(r, v1, k, lambda_=0.5):
    from scipy.optimize import linprog

    """
    Selects k items to maximize:
    Total relevance - lambda * max(0, -alignment),
    where alignment = sum(r_i * v1_i for selected items).

    Args:
        r (np.ndarray): Relevance scores (shape: [n])
        v1 (np.ndarray): Principal eigenvector components (shape: [n])
        lambda_ (float): Penalty strength for negative alignment
        k (int): Number of items to select

    Returns:
        selected_indices (np.ndarray): Indices of selected items
        final_alignment (float): Final alignment value of selected set
    """
    n = len(r)

    # Linear programming setup
    # Objective: Maximize sum(r_i * x_i) - lambda * z
    # Variables: x_i (binary selection), z (slack for penalty)
    c = np.hstack([-r, lambda_])  # Minimize -sum(r_i x_i) + lambda z

    # Constraints:
    # 1. sum(x_i) = k (select exactly k items)
    A_eq = np.hstack([np.ones(n), 0]).reshape(1, n + 1)
    b_eq = np.array([k])

    # 2. z >= -sum(r_i v1_i x_i) (penalty slack)
    A_ub = np.hstack([r * v1, 1]).reshape(1, n + 1)
    b_ub = np.array([0])

    # 3. z >= 0
    A_ub = np.vstack([A_ub, [0] * n + [-1]])
    b_ub = np.hstack([b_ub, 0])

    # Bounds: x_i in [0, 1], z >= 0
    bounds = [(0, 1)] * n + [(0, None)]

    # Solve the linear program
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    # Extract selected items
    x = result.x[:n]
    selected_indices = np.where(x > 0.5)[0]  # Threshold to binary

    # Compute final alignment
    final_alignment = np.sum(r[selected_indices] * v1[selected_indices])
    return selected_indices, final_alignment


def freddy(
    dataset,
    base_inc=base_inc,
    alpha=0.15,
    metric="similarity",
    K=1,
    batch_size=128,
    beta=0.75,
    return_vals=False,
    relevance=None,
):
    sample_size = K / len(dataset)
    print(sample_size)
    idx = np.arange(len(dataset))
    selected, alignment = [], []
    for ds, V in zip(
        batched(dataset, batch_size),
        batched(idx, batch_size),
    ):
        D = METRICS[metric](ds, batch_size=batch_size)
        V = np.array(V)
        r = D @ relevance[V]
        eigenvals, eigenvectors = np.linalg.eigh(D)
        max_eigenval = np.argsort(eigenvals)[-1]
        v1 = eigenvectors[max_eigenval]
        sset, score = linear_selector(r, v1, k=sample_size * batch_size, lambda_=0.5)
        # sset, score = linear_selector(r, v1, k=K, lambda_=0.5)
        selected.append(V[sset])
        alignment.append(score)

    selected = np.hstack(selected)
    alignment = np.hstack(alignment)
    print(len(selected), K, len(dataset))
    print(len(np.unique(selected)), K)
    print(len(alignment), K)
    exit()


@_register
def _freddy(
    dataset,
    base_inc=base_inc,
    alpha=0.15,
    metric="similarity",
    K=1,
    batch_size=128,
    beta=0.75,
    return_vals=False,
    relevance=None,
):
    # basic config
    base_inc = base_inc(alpha)
    # base_inc = 0
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
        size = len(ds)
        v = list(V)
        D = METRICS[metric](ds, batch_size=batch_size)
        localmax = np.amax(D, axis=1)
        argmax += localmax.sum()
        _ = [q.push(base_inc, i) for i in zip(V, range(size))]
        ##################
        eigenvals, eigenvectors = np.linalg.eigh(D)
        max_eigenval = np.argsort(eigenvals)[-1]
        v_i = eigenvectors[max_eigenval]
        # normalize relevance
        r = relevance[v]
        r /= np.linalg.norm(r)
        # sign alignment
        if r @ -v_i > 0:
            v_i = -v_i
        # linear penalty
        # r = r @ v_i - max(0.0, -r @ v_i)
        # exponential penalty
        r = v_i * localmax

        localmax = r * v_i - np.exp(-r * v_i)
        # r = np.maximum(relevance[v], r)
        ##################
        while q and len(sset) < K:
            score, idx_s = q.head
            s = D[idx_s[1], :] + r

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
            # base_inc = min(1, base_inc + 10e-2)
    if return_vals:
        return np.array(vals), sset
    # import matplotlib.pyplot as plt

    # plt.plot(vals)
    # plt.show()
    return np.array(sset)


def shannon_entropy(vector, epsilon=1e-10):
    abs_vector = np.abs(vector)  # Ensure non-negative
    total = abs_vector.sum(axis=1) + epsilon  # Avoid division by zero
    total = total.reshape(-1, 1)
    p = abs_vector / total
    p = p[p > 0]  # Remove zeros to avoid log(0)
    # p += 1  # Remove zeros to avoid log(0)
    return -(p * np.log2(p))


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
        # self._relevance_score = np.random.normal(0, 1, n)
        self._relevance_score = np.random.normal(0, 1, n)
        self.select_flag = True
        self.cur_error = 0

    def _select_subset(self, epoch, training_step):
        self.model.eval()
        print(f"selecting subset on epoch {epoch}")
        self.epoch_selection.append(epoch)
        self.f_embedding()
        sset = freddy(
            self.delta,
            batch_size=32,
            K=self.sample_size,
            metric=self.args.freddy_similarity,
            alpha=self.args.alpha,
            beta=self.args.beta,
            # relevance=1 / self._relevance_score,
            relevance=self._relevance_score,
        )
        self.subset = sset

        self.selected[sset] += 1
        self.train_checkpoint["selected"] = self.selected
        self.train_checkpoint["importance"] = self._relevance_score
        self.train_checkpoint["epoch_selection"] = self.epoch_selection
        self.subset_weights = np.ones(self.sample_size)
        self.model.train()
        print(f"selected {len(sset)}")

    def _train_epoch(self, epoch):
        self.model.train()
        self._reset_metrics()

        lr = self.lr_scheduler.get_last_lr()[0]

        data_start = time.time()
        pbar = tqdm(
            enumerate(self.train_loader), total=len(self.train_loader), file=sys.stdout
        )
        train_loss = 0
        for batch_idx, (data, target, data_idx) in pbar:
            # load data to device and record data loading time
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
            self.hist[-1]["avg_importance"] = self._relevance_score[self.subset].mean()

        print(f"relative error: {self.cur_error}")
        if self.hist:
            self.hist[-1]["reaL_error"] = self.cur_error

        # if self.cur_error > 1 or not epoch:
        # if self._relevance_score[self.subset].mean() < 10e-4 or not epoch:
        if epoch % 5 == 0:
            self._select_subset(epoch, len(self.train_loader) * epoch)
            # self._relevance_score = shannon_entropy(self.delta)
            self._relevance_score = np.linalg.norm(self.delta, axis=1)
            # print(self.train_dataset.dataset[3])
            # print(self.delta)
            # print(self.subset)
            # print(self.subset.shape)
            # print(self.delta.shape)
            self._update_train_loader_and_weights()
        # self.delta[self.subset] += self.cur_error * lr
        # self._relevance_score -= self._relevance_score * lr
        # self.cur_error = abs(self.cur_error - self._relevance_score[self.subset].mean())
        self.cur_error = abs(self.cur_error - train_loss)

    def f_embedding(self):
        dataset = self.train_dataset.dataset
        print(len(dataset))
        dataset = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )
        delta = map(self.calc_embbeding, dataset)
        self.delta = np.vstack([*delta])

    def calc_embbeding(self, train_data, ord=1):
        data, target = train_data
        # data, target = data.cpu(), target.cpu()
        data, target = data.to(self.args.device), target.to(self.args.device)
        target = torch.nn.functional.one_hot(target, self.args.num_classes).float()
        pred = self.model(data).softmax(dim=1)
        loss = self.val_criterion(pred, target)
        w = [*self.model.modules()]
        w = (w[-1].weight,)
        return self._update_delta((data, target))
        f = self._update_delta((data, target))
        grad = torch.autograd.grad(loss, w, retain_graph=True, create_graph=True)[0]
        g = torch.inner(f, grad.T)

        hess = torch.autograd.grad(grad, w, retain_graph=True, grad_outputs=grad)[0]
        gg = torch.inner(g, hess)

        return torch.inner(f, torch.inner(gg.T, g.T).T).cpu().detach().numpy()

    def _update_delta(self, train_data):
        data, target = train_data
        data = data.to(self.args.device)
        self.model.eval()
        e = torch.normal(0, 1, size=data.shape).to(self.args.device)
        with torch.no_grad():
            data = data.to(self.args.device)
            loss = self.model(data)
            delta_loss = self.model(data + e).softmax(dim=1)
        # return loss - delta_loss
        return (loss - target).cpu().detach().numpy()
        return loss - target
        # return loss - delta_loss
