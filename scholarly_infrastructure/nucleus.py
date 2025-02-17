"""core functionalities of this lib"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/00_nucleus (other infras).ipynb.

# %% auto 0
__all__ = [
    "MuteWarnings",
    "ensure_array",
    "default_on_exception",
    "append_dict_list",
    "partial_with_self",
]

# %% ../notebooks/00_nucleus (other infras).ipynb 3
import warnings


class MuteWarnings:
    def __enter__(self):
        # self.warnings_show = warnings.showwarning
        # warnings.showwarning = lambda *args, **kwargs: None
        self.mute()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # warnings.showwarning = self.warnings_show
        self.close()

    def mute(self):
        warnings.filterwarnings("ignore", append=True)

    def resume(self):
        warnings.filters.pop(0)


# %% ../notebooks/00_nucleus (other infras).ipynb 4
import torch
import numpy as np


def ensure_array(x: torch.TensorType | np.ndarray | list):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:  # list
        return np.array(x)


# %% ../notebooks/00_nucleus (other infras).ipynb 5
from .logging.nucleus import logger
from decorator import decorator


@decorator
def default_on_exception(func, default_value=None, verbose=False, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        # logger.warning(f"An exception occurred: {e}")
        if verbose:
            logger.exception(e)
        return default_value


# %% ../notebooks/00_nucleus (other infras).ipynb 7
def append_dict_list(dict, name, value):
    dict[name] = dict.get(name, []) + [value]


# %% ../notebooks/00_nucleus (other infras).ipynb 8
# TODO 暂时无法使用 decorator实现这个; 目前尽量不要使用这个API
def partial_with_self(method, *args, **kwargs):
    def wrapped(self, *additional_args, **additional_kwargs):
        # Combine provided args and kwargs with additional ones
        all_args = args + additional_args
        all_kwargs = kwargs | additional_kwargs
        return method(self, *all_args, **all_kwargs)

    return wrapped
