class Tensor:
    def __init__(self, data=None, **kwargs):
        self.data = data
    def to(self, *args, **kwargs):
        return self

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
    return data

def tanh(x):
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
