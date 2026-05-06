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
        super(SVDSingleGroupTransMatrix, self).__init__()
        assert in_features % group_size == 0, "in_features must be divisible by group_size"
        self.in_features = in_features
        self.group_size = group_size
        self.num_groups = in_features // group_size

        self.linear_u_list = nn.ModuleList()
        self.linear_v_list = nn.ModuleList()
        self.linear_diag_list = nn.ParameterList()
        for _ in range(self.num_groups):
            linear_u = nn.Linear(group_size, group_size, bias=False, dtype=torch.float32)
            linear_u.weight.data = get_init_weight(group_size).to(linear_u.weight)
            linear_u = nn.utils.parametrizations.orthogonal(linear_u, orthogonal_map="cayley", use_trivialization=False)
            self.linear_u_list.append(linear_u)

            linear_v = nn.Linear(group_size, group_size, bias=False, dtype=torch.float32)
            linear_v.weight.data = get_init_weight(group_size).to(linear_v.weight)
            linear_v = nn.utils.parametrizations.orthogonal(linear_v, orthogonal_map="cayley", use_trivialization=False)
            self.linear_v_list.append(linear_v)

            self.linear_diag_list.append(torch.nn.Parameter(torch.ones(group_size, dtype=torch.float32), requires_grad=True))

        self.add_diag = add_diag
        self.use_diag = True
        if self.add_diag:
            self.diag_scale = torch.nn.Parameter(torch.ones(in_features, dtype=torch.float32), requires_grad=True)
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        assert inp.shape[-1] == self.in_features
        if self.add_diag and self.use_diag:
            if inv_t:
                inp = inp / self.diag_scale.to(inp)
            else:
                inp = inp * self.diag_scale.to(inp)

        inp_chunks = torch.chunk(inp, self.num_groups, dim=-1)
        out_chunks = []
        if not self._eval_mode:
            for idx, x in enumerate(inp_chunks):
                matrix_u = self.linear_u_list[idx].weight
                matrix_v = self.linear_v_list[idx].weight
                linear_diag = self.linear_diag_list[idx]
                if inv_t:
                    linear_diag = 1 / linear_diag
                out_chunks.append(((x @ matrix_u) * linear_diag) @ matrix_v.t())
        else:
            matrices = self.matrix_inv_t_list if inv_t else self.matrix_list
            for idx, x in enumerate(inp_chunks):
                out_chunks.append(x @ matrices[idx].to(x))
        return torch.cat(out_chunks, dim=-1)

    def to_eval_mode(self):
        if self._eval_mode:
            return
        matrix_list = []
        matrix_inv_t_list = []
        for idx in range(self.num_groups):
            matrix_u = self.linear_u_list[idx].weight
            matrix_v = self.linear_v_list[idx].weight
            linear_diag = self.linear_diag_list[idx]
            matrix_list.append(nn.Parameter(matrix_u @ torch.diag(linear_diag) @ matrix_v.t(), requires_grad=False))
            matrix_inv_t_list.append(nn.Parameter(matrix_u @ torch.diag(1 / linear_diag) @ matrix_v.t(), requires_grad=False))
        self.matrix_list = nn.ParameterList(matrix_list)
        self.matrix_inv_t_list = nn.ParameterList(matrix_inv_t_list)
        del self.linear_u_list, self.linear_v_list, self.linear_diag_list
        self._eval_mode = True

    def __repr__(self):
        return (
            "SVDSingleGroupTransMatrix("
            f"eval_mode={self._eval_mode}, in_features={self.in_features}, "
            f"group_size={self.group_size}, num_groups={self.num_groups})"
        )
# Adapt SplitQuant to Qwen2.5, LLaMA-2, LLaMA-3, Qwen2.5-VL, and MiniCPM models.
