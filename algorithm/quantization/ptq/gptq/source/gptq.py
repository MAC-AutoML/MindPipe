import math
import time

import torch
import torch.nn as nn
import transformers

from algorithm.common.device import empty_cache
from algorithm.common.device import synchronize
from quant import *


DEBUG = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        return_real_quant=False,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        if dead.any():
            W[:, dead.to(W.device)] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        real_group_size = self.columns if groupsize == -1 else int(groupsize)
        if return_real_quant:
            num_groups = math.ceil(self.columns / real_group_size)
            QInt = torch.zeros((self.rows, self.columns), dtype=torch.int16, device=self.dev)
            QScale = torch.empty((self.rows, num_groups), dtype=torch.float32, device=self.dev)
            QScale.fill_(float("nan"))

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=H.device)
        Hinv = None
        current_damp = damp
        last_error = None
        for _ in range(6):
            H_work = H.clone()
            H_work[diag, diag] += current_damp
            try:
                H_work = torch.linalg.cholesky(H_work)
                H_work = torch.cholesky_inverse(H_work)
                H_work = torch.linalg.cholesky(H_work, upper=True)
                Hinv = H_work
                break
            except RuntimeError as error:
                last_error = error
                if "cholesky" not in str(error).lower():
                    raise
                current_damp = current_damp * 10 if current_damp > 0 else torch.tensor(1e-4, device=H.device)
        if Hinv is None:
            raise last_error

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2].to(self.dev)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                if return_real_quant:
                    q, q_int = quantize_with_qint(
                        w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    )
                    q_signed = q_int.to(torch.float32) - self.quantizer.zero
                    QInt[:, i1 + i] = q_signed.flatten().to(torch.int16)
                    group_index = 0 if groupsize == -1 else (i1 + i) // groupsize
                    QScale[:, group_index] = self.quantizer.scale.reshape(-1).float()
                    q = q.flatten()
                else:
                    q = quantize(
                        w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            if i2 < self.columns:
                Hinv_tail = Hinv[i1:i2, i2:].to(self.dev)
                W[:, i2:] -= Err1.matmul(Hinv_tail)

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        if self.dev.type in {"cuda", "npu"}:
            synchronize(self.dev)
        # print('time %.2f' % (time.time() - tick))
        # print('error', torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]
            if return_real_quant:
                QInt = QInt[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
            if return_real_quant:
                QInt = QInt.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
        if return_real_quant:
            return {
                "int_weight": QInt.reshape(self.layer.weight.shape).detach(),
                "scale": torch.nan_to_num(QScale, nan=0.0).detach(),
                "group_size": real_group_size,
            }
        return None

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        empty_cache(self.dev)
# Fix GPU/NPU adaptation bugs and use a unified device abstraction.
