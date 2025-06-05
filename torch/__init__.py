class Tensor:
    def __init__(self, data=None, **kwargs):
        self.data = data
    def to(self, *args, **kwargs):
        return self
    def unsqueeze(self, dim):
        return self
    def size(self, dim=None):
        if dim is None:
            return len(self.data)
        return self.data.shape[dim]
    def cpu(self):
        return self
    def numpy(self):
        return self.data
    def __getitem__(self, idx):
        return Tensor(self.data[idx])
    def view(self, *shape):
        """Return a reshaped tensor if possible, otherwise ``self``.

        The implementation is intentionally minimal and only supports
        reshaping via ``numpy.reshape``.  If ``self.data`` cannot be
        converted to a ``numpy`` array or the new shape is incompatible,
        the original tensor is returned unchanged.
        """
        if not shape:
            return self
        try:
            import numpy as np
            new_data = np.array(self.data).reshape(*shape)
        except Exception:
            return self
        return Tensor(new_data)
    def item(self):
        import numpy as np
        if isinstance(self.data, (list, tuple, np.ndarray)):
            return float(np.array(self.data).flatten()[0])
        return float(self.data)

class _NoGrad:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def no_grad():
    return _NoGrad()

class Module:
    def __init__(self, *args, **kwargs):
        pass
    def to(self, *args, **kwargs):
        return self
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Conv2d(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class BatchNorm2d(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class Linear(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class ModuleList(list):
    def __init__(self, iterable=()):
        super().__init__(iterable)

def tensor(data, **kwargs):
    return Tensor(data)

def tanh(x):
    return x

# minimal dtype constants used in tests
float32 = 'float32'
float64 = 'float64'

def log_softmax(x, dim=None):
    return x

def ones(*shape):
    import numpy as np
    return Tensor(np.ones(shape))

def zeros(*shape):
    import numpy as np
    return Tensor(np.zeros(shape))

def exp(x):
    return x

import types

nn = types.ModuleType('torch.nn')
nn.Module = Module
nn.Conv2d = Conv2d
nn.BatchNorm2d = BatchNorm2d
nn.Linear = Linear
nn.ModuleList = ModuleList

class functional(types.ModuleType):
    @staticmethod
    def relu(x):
        return x

    @staticmethod
    def log_softmax(x, dim=None):
        return x

F = functional('torch.nn.functional')
F.relu = functional.relu
F.log_softmax = functional.log_softmax

import sys
sys.modules[__name__ + '.nn'] = nn
sys.modules[__name__ + '.nn.functional'] = F
sys.modules[__name__ + '.functional'] = F
cuda = types.SimpleNamespace(is_available=lambda: False)

utils = types.ModuleType('torch.utils')
data = types.ModuleType('torch.utils.data')

class Dataset:
    pass

class DataLoader:
    def __init__(self, dataset=None, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors

data.Dataset = Dataset
data.DataLoader = DataLoader
data.TensorDataset = TensorDataset
utils.data = data
sys.modules[__name__ + '.utils'] = utils
sys.modules[__name__ + '.utils.data'] = data
