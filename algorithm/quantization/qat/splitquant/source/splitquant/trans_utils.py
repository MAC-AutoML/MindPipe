import torch
import torch.nn as nn

from splitquant.function_utils import get_init_weight


# ---------- transformation version of singular value decomposition ----------
class SVDSingleTransMatrix(nn.Module):
    def __init__(self, size):
        super(SVDSingleTransMatrix, self).__init__()
        self.linear_u = nn.Linear(size, size, bias=False, dtype=torch.float32)
        self.linear_u.weight.data = get_init_weight(size).to(self.linear_u.weight)
        self.linear_u = nn.utils.parametrizations.orthogonal(self.linear_u, orthogonal_map="cayley", use_trivialization=False)
        self.linear_v = nn.Linear(size, size, bias=False, dtype=torch.float32)
        self.linear_v.weight.data = get_init_weight(size).to(self.linear_v.weight)
        self.linear_v = nn.utils.parametrizations.orthogonal(self.linear_v, orthogonal_map="cayley", use_trivialization=False)
        self.linear_diag = torch.nn.Parameter(torch.ones(size, dtype=torch.float32), requires_grad=True)

        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        init_shape = inp.shape
        matirx = self.get_matrix(inv_t=inv_t).to(inp)
        inp = inp.reshape(-1, matirx.shape[0])
        return inp.matmul(matirx).reshape(init_shape)

    def get_matrix(self, inv_t=False):
        if not self._eval_mode:
            orthog_u, orthog_v = self.linear_u.weight, self.linear_v.weight
            linear_diag = self.linear_diag
            if inv_t:
                linear_diag = 1 / linear_diag
            return orthog_u @ torch.diag(linear_diag) @ orthog_v.t()
        else:
            if inv_t:
                return self.matrix_inv_t
            return self.matrix

    def to_eval_mode(self):
        if not self._eval_mode:
            matrix = self.linear_u.weight @ torch.diag(self.linear_diag) @ self.linear_v.weight.t()
            matrix_inv_t = self.linear_u.weight @ torch.diag(1 / self.linear_diag) @ self.linear_v.weight.t()
            self.matrix = nn.Parameter(matrix, requires_grad=False)
            self.matrix_inv_t = nn.Parameter(matrix_inv_t, requires_grad=False)
            self._eval_mode = True
            del self.linear_u, self.linear_diag, self.linear_v

    def __repr__(self):
        res = f"SVDSingleTransMatrix(eval_mode={self._eval_mode}"
        if hasattr(self, 'matrix'):
            res += f", matrix.shape={self.matrix.shape})"
        else:
            res += f", matrix.shape={self.linear_u.weight.shape})"
        return res


class SVDSingleGroupTransMatrix(nn.Module):
    def __init__(self, in_features, group_size, add_diag=False):
        super().__init__()
        assert in_features % group_size == 0, "in_features must be divisible by group_size"
        self.in_features = in_features
        self.group_size = group_size
        self.num_groups = in_features // group_size

        init_weight = get_init_weight(group_size).unsqueeze(0).expand(self.num_groups, -1, -1).clone()
        self.linear_u_raw = nn.Parameter(init_weight.clone(), requires_grad=True)
        self.linear_v_raw = nn.Parameter(init_weight.clone(), requires_grad=True)
        self.linear_diag = nn.Parameter(torch.ones(self.num_groups, group_size, dtype=torch.float32), requires_grad=True)

        self.add_diag = add_diag
        self.use_diag = True
        if self.add_diag:
            self.diag_scale = nn.Parameter(torch.ones(in_features, dtype=torch.float32), requires_grad=True)
        self._eval_mode = False

    def _cayley(self, raw):
        x = raw.tril()
        a = x - x.transpose(-1, -2)
        eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device).expand_as(a)
        return torch.linalg.solve(torch.add(eye, a, alpha=-0.5), torch.add(eye, a, alpha=0.5))

    def _matrices(self, inv_t=False):
        matrix_u = self._cayley(self.linear_u_raw)
        matrix_v = self._cayley(self.linear_v_raw)
        linear_diag = self.linear_diag
        if inv_t:
            linear_diag = 1 / linear_diag
        return matrix_u, matrix_v, linear_diag

    def forward(self, inp, inv_t=False):
        assert inp.shape[-1] == self.in_features
        if self.add_diag and self.use_diag:
            if inv_t:
                inp = inp / self.diag_scale.to(inp)
            else:
                inp = inp * self.diag_scale.to(inp)

        init_shape = inp.shape
        inp = inp.reshape(-1, self.num_groups, self.group_size)
        if not self._eval_mode:
            matrix_u, matrix_v, linear_diag = self._matrices(inv_t=inv_t)
            out = torch.einsum("bng,ngh->bnh", inp, matrix_u.to(inp))
            out = out * linear_diag.to(out)
            out = torch.einsum("bng,nhg->bnh", out, matrix_v.to(out))
        else:
            matrices = self.matrix_inv_t if inv_t else self.matrix
            out = torch.einsum("bng,ngh->bnh", inp, matrices.to(inp))
        return out.reshape(init_shape)

    def to_eval_mode(self):
        if self._eval_mode:
            return
        matrix_u, matrix_v, linear_diag = self._matrices(inv_t=False)
        matrix_u_inv, matrix_v_inv, linear_diag_inv = self._matrices(inv_t=True)
        self.matrix = nn.Parameter(
            torch.matmul(matrix_u * linear_diag.unsqueeze(1), matrix_v.transpose(-1, -2)),
            requires_grad=False,
        )
        self.matrix_inv_t = nn.Parameter(
            torch.matmul(matrix_u_inv * linear_diag_inv.unsqueeze(1), matrix_v_inv.transpose(-1, -2)),
            requires_grad=False,
        )
        del self.linear_u_raw, self.linear_v_raw, self.linear_diag
        self._eval_mode = True

    def __repr__(self):
        return (
            "SVDSingleGroupTransMatrix("
            f"eval_mode={self._eval_mode}, in_features={self.in_features}, "
            f"group_size={self.group_size}, num_groups={self.num_groups})"
        )
# Adapt SplitQuant to Qwen2.5, LLaMA-2, LLaMA-3, Qwen2.5-VL, and MiniCPM models.
