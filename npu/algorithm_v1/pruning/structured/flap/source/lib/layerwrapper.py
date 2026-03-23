import torch
import torch.nn as nn

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        
    def free(self):
        self.scaler_row = None
        torch.cuda.empty_cache()


# Define BiasGPT class
class BiasGPT:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, metric):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.out_dim = layer.weight.data.shape[0]
        self.in_dim = layer.weight.data.shape[1]
        self.type = metric
        self.nsamples = 0

        self.baseline_inp = torch.zeros((self.in_dim), device=self.dev)
        if self.type == "WIFN":
            self.scaler_inp = torch.zeros((self.in_dim), device=self.dev)
        else:   
            self.fluc_inp = torch.zeros((self.in_dim), device=self.dev)

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()   # (dim, seqlen)

        old_baseline_inp = self.baseline_inp
        self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        if self.type == "WIFN":
            inp = inp.type(torch.float32)
            self.scaler_inp *= self.nsamples / (self.nsamples + batch_size)
            self.scaler_inp += torch.norm(inp, p=2, dim=1) ** 2  / (self.nsamples + batch_size)
        else:
            if self.nsamples == 0:
                self.fluc_inp = 0
            else:
                self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
                self.fluc_inp += torch.sum((inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1) / (self.nsamples + batch_size)   # a²+b²+c²...没开根号

        self.nsamples += batch_size

        
    def free(self):
        self.baseline_inp = None
        if hasattr(self, 'fluc_inp'):
            self.fluc_inp = None
        if hasattr(self, 'scaler_inp'):
            self.scaler_inp = None
        torch.cuda.empty_cache()  
