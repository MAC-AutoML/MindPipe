import time

import torch
import torch.nn as nn
import transformers
import numpy as np

from algorithm.common.device import empty_cache
from algorithm.common.device import synchronize


class ALPS_prune:

    def __init__(self, layer, nsamples, seqlen, dev=None):
        self.layer = layer
        self.dev = dev if dev is not None else self.layer.weight.device

        self.nsamples = nsamples
        self.seqlen = seqlen
        self.equi_nsamples = self.nsamples*self.seqlen
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.XtX = torch.zeros((self.columns, self.columns), device=self.dev).float()

        self.count = 0

        self.nsamples = 0


    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))
        out = out.t()
        if isinstance(self.layer, nn.Conv2d):
            print(inp.shape)
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        inp = inp.float()
        out = out.float()


        self.XtX += inp.matmul(inp.t())
        self.nsamples += tmp


    def ALPS_admm(self, sp, nm_n = 0, nm_m = 0, rho=0.1, max_iter = 300, update_iter = 3, switch_iter = 30):

        # get dense weight
        W = self.layer.weight.data.clone().float().to(self.dev)

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        # ridge term — XtX stays on GPU
        damp1 = 0.01 * torch.mean(torch.diag(self.XtX)).item()
        diag = torch.arange(self.XtX.shape[0], device=self.dev)
        self.XtX[diag,diag] += damp1

        # normalization
        X_norm = torch.diag(self.XtX).sqrt() + 1e-8
        self.XtX = self.XtX / X_norm
        self.XtX = (self.XtX.T / X_norm).T

        self.YtX = torch.matmul(W * X_norm, self.XtX)

        admm_st = time.time()

        # initialization — all on GPU
        B = (W * X_norm).t().clone()
        W = None
        B_orig = B.clone()
        V = torch.zeros_like(B)
        D = torch.zeros_like(B)
        D_suppp = torch.zeros_like(B)
        D_supp = torch.zeros_like(B)

        totp, num_cout = B.shape
        L, Q = torch.linalg.eigh(self.XtX.double())
        XTX_inv = (Q @ ((1/(L+(rho))) * Q).T).float()

        init_rho = False
        fix_supp = False
        D_fix = torch.zeros_like(D)

        Res0 = torch.sum(B_orig * self.YtX.T)

        params = B.shape[0]*B.shape[1]
        k_spar = int(np.round((1-sp)*params))


        if nm_n == 0:
            D = B.clone().reshape(-1)
            _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
            D[loss_idx] = 0
            D_suppp = (D == 0).to(torch.float)
            D = D.reshape(totp, num_cout)
        else:
            new_dim = totp * num_cout / nm_m
            new_dim = int(new_dim)
            k_spar = totp * num_cout * nm_n/nm_m


            D = B.clone().t().reshape((new_dim, nm_m))
            _, loss_idx = torch.topk(-D**2,nm_m - nm_n, dim = 1)
            D = D.scatter(src=torch.zeros((new_dim,nm_m-nm_n), device=self.dev),dim=1,index=loss_idx)
            D_suppp = (D == 0).to(torch.float)
            D = D.reshape(num_cout, totp).t()

        D_init = D.clone()
        errorp = 1
        for i_admm in range(max_iter):


            B = XTX_inv @ (self.YtX.T-V+rho*D)

            if fix_supp:
                D = ((V + rho * B) / rho) * D_fix
            elif nm_n == 0:
                D = ((V + rho * B) / rho).reshape(-1)
                _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
                D[loss_idx] = 0
                D = D.reshape(totp, num_cout)
            else:
                D = ((V + rho * B) / rho).t().reshape((new_dim, nm_m))
                _, loss_idx = torch.topk(-D**2,nm_m - nm_n, dim = 1)
                D = D.scatter(src=torch.zeros((new_dim,nm_m-nm_n), device=self.dev),dim=1,index=loss_idx)
                D_supp = (D == 0).to(torch.float)
                D = D.reshape(num_cout, totp).t()

            V = V + rho * (B - D)

            if (i_admm+1) % update_iter == 0:


                if nm_n == 0:
                    D_supp = (D.reshape(-1) == 0).to(torch.float)
                supp_change = torch.sum((D_supp-D_suppp)**2)

                if not fix_supp:
                    if supp_change / k_spar > 0.1:
                        init_rho = True
                        rho *= 1.3
                    elif supp_change / k_spar > 0.005:
                        init_rho = True
                        rho *= 1.2
                    elif supp_change > 0.5:
                        if init_rho:
                            rho *= 1.1
                        else:
                            rho /= 5
                            B = B_orig.clone()
                            D = D_init.clone()
                            V = torch.zeros_like(B)
                    else:
                        if init_rho:
                            break
                        else:
                            rho /= 5

                D_suppp = (D_supp).clone()
                if rho > 1e6:
                    rho = 1e6

                XTX_inv = (Q @ ((1/(L+(rho))) * Q).T).float()

                if nm_n == 0:
                    Btest = B.reshape(-1)
                    _, loss_idx = torch.topk(-Btest**2,totp * num_cout - k_spar)
                    Btest[loss_idx] = 0
                    Btest = Btest.reshape(totp, num_cout)
                else:
                    Btest = B.t().reshape((new_dim, nm_m))
                    _, loss_idx = torch.topk(-Btest**2,nm_m - nm_n, dim = 1)
                    Btest = Btest.scatter(src=torch.zeros((new_dim,nm_m-nm_n), device=self.dev),dim=1,index=loss_idx)
                    Btest = Btest.reshape(num_cout, totp).t()

                Resc = torch.matmul(self.XtX, Btest) - self.YtX.T
                Resc = torch.diag(torch.matmul((Btest-B_orig).t(), Resc))

                errorc = (torch.sum(Resc) / Res0).item()

                if i_admm >= switch_iter and supp_change / k_spar < 0.0003:
                    break

        if nm_n == 0:
            B = B.reshape(-1)
            _, loss_idx = torch.topk(-B**2,totp * num_cout - k_spar)
            B[loss_idx] = 0
            B = B.reshape(totp, num_cout)
        else:
            B = B.t().reshape((new_dim, nm_m))
            _, loss_idx = torch.topk(-B**2,nm_m - nm_n, dim = 1)
            B = B.scatter(src=torch.zeros((new_dim,nm_m-nm_n), device=self.dev),dim=1,index=loss_idx)
            B = B.reshape(num_cout, totp).t()

        V = None
        D = None

        Res = torch.matmul(self.XtX, B) - self.YtX.T
        Res = torch.diag(torch.matmul((B - B_orig).t(), Res))
        error = (torch.sum(Res) / Res0).item()

        admm_time = time.time() - admm_st

        back_st = time.time()
        B = self.cg_batch(self.XtX, self.YtX.T,
                       (B != 0).to(torch.float), M_bmm=None, X0=B, rtol=1e-4, atol=0., maxiter=10, verbose=False)
        back_time = time.time() - back_st

        Res = torch.matmul(self.XtX, B) - self.YtX.T
        Res = torch.diag(torch.matmul((B - B_orig).t(), Res))
        error = (torch.sum(Res) / Res0).item()

        synchronize(self.dev)

        if isinstance(self.layer, transformers.Conv1D):
            self.layer.weight.data = (B.t() / X_norm).t().reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        else:
            self.layer.weight.data = (B.t() / X_norm).reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        return error



    # A modified version of https://github.com/sbarratt/torch_cg

    def cg_batch(self, A, B, A_supp, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
        """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm."""
        n, m = B.shape

        if M_bmm is None:
            M_bmm = lambda x: x
        if X0 is None:
            X0 = M_bmm(B)
        if maxiter is None:
            maxiter = 5 * n

        assert B.shape == (n, m)
        assert X0.shape == (n, m)
        assert rtol > 0 or atol > 0
        assert isinstance(maxiter, int)

        X_k = X0

        R_k = B - A @ X_k
        R_k = R_k * A_supp

        Z_k = M_bmm(R_k)

        P_k = torch.zeros_like(Z_k)

        P_k1 = P_k
        R_k1 = R_k
        R_k2 = R_k
        X_k1 = X0
        Z_k1 = Z_k
        Z_k2 = Z_k

        B_norm = torch.norm(B, dim=1)
        stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

        if verbose:
            print("%03s | %010s %06s" % ("it", "dist", "it/s"))

        optimal = False
        start = time.perf_counter()
        for k in range(1, maxiter + 1):
            start_iter = time.perf_counter()
            Z_k = M_bmm(R_k)

            if k == 1:
                P_k = Z_k
                R_k1 = R_k
                X_k1 = X_k
                Z_k1 = Z_k
            else:
                R_k2 = R_k1
                Z_k2 = Z_k1
                P_k1 = P_k
                R_k1 = R_k
                Z_k1 = Z_k
                X_k1 = X_k
                denominator = (R_k2 * Z_k2).sum(0)
                denominator[denominator == 0] = 1e-8
                beta = (R_k1 * Z_k1).sum(0) / denominator
                P_k = Z_k1 + beta.unsqueeze(0) * P_k1

            denominator = (P_k * (A@P_k)).sum(0)
            denominator[denominator == 0] = 1e-8
            alpha = (R_k1 * Z_k1).sum(0) / denominator
            X_k = X_k1 + alpha.unsqueeze(0) * P_k
            R_k = R_k1 - alpha.unsqueeze(0) * (A@P_k)
            R_k = R_k * A_supp
            end_iter = time.perf_counter()

            residual_norm = torch.norm(A@X_k - B, dim=1)

            if verbose:
                print("%03d | %8.4e" %
                      (k, torch.max(residual_norm/B_norm)))

            if (residual_norm <= stopping_matrix).all():
                optimal = True
                break


        end = time.perf_counter()

        if verbose:
            if optimal:
                print("Terminated in %d steps (optimal). Took %.3f ms." %
                      (k, (end - start) * 1000))
            else:
                print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                      (k, (end - start) * 1000))

        return X_k


    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        self.X = None
        self.Y = None
        self.XXt = None
        self.YXt = None
        self.YtX = None
        self.XtX = None
        empty_cache(self.dev)
