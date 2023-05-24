from typing import List, Any, Union, Tuple
from torch import Tensor
import torch
import pandas as pd
from tqdm.notebook import tqdm
import torchmetrics


def make_sequence_from_time_series(ts: Any, train_seq_size: int, test_seq_size: int) \
        -> Union[Tuple[List, List], Tuple[Tensor, Tensor]]:
    """
    :param ts: target values from dataset
    :param train_seq_size: integer to separate length of train sequence
    :param test_seq_size: integer to separate length of test sequence
    :return: x, y
    """

    x, y = [], []
    if ts.shape[0] - train_seq_size - test_seq_size != 0:
        for i in range(ts.shape[0] - train_seq_size - test_seq_size):
            x_i = ts[i: i + train_seq_size]
            y_i = ts[i + train_seq_size: i + train_seq_size + test_seq_size]
            x.append(x_i)
            y.append(y_i)
    else:
        x = ts[:train_seq_size]
        y = ts[train_seq_size:]
    return x, y


def data_scaling(ts: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    return (ts - ts.mean())/ts.std()


def train_test_split(x_seq: List, y_seq: List, train_size: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return torch.stack(x_seq[: round(len(x_seq) * train_size)]), torch.stack(y_seq[: round(len(x_seq) * train_size)]), \
           torch.stack(x_seq[round(len(x_seq) * train_size):]), torch.stack(y_seq[round(len(x_seq) * train_size):])






