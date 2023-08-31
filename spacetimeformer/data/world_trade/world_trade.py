import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset


class World_Trade_Data:
    def _read(self, split):
        df = pd.read_csv(self.path)
        df = df.drop(columns='exp-imp')
        df = df[(df != 0).any(axis=1)]
        x = df.columns[1:]
        df_np = df.to_numpy()
        df_np = df_np[:, 1:]
        df_np = df_np.transpose()

        if split == 'train':
            return x[:20], df_np[:20]
        elif split == 'val':
            return x[20:23], df_np[21:24]
        elif split == 'test':
            return x[23:26], df_np[23:26]

    def _split_set(self, data):
        x = []
        y = []
        i = 0
        while i+4 < len(data):
            x.append(data[i:i+3])
            y.append(data[i+3])
            i += 1

        return np.array(x), np.array(y)

    def __init__(self, path):
        self.path = path

        context_train, target_train = self._read("train")
        context_val, target_val = self._read("val")
        context_test, target_test = self._read("test")

        x_c_train, y_c_train = self._split_set(context_train)
        x_t_train, y_t_train = self._split_set(target_train)

        x_c_val, y_c_val = self._split_set(context_val)
        x_t_val, y_t_val = self._split_set(target_val)

        x_c_test, y_c_test = self._split_set(context_test)
        x_t_test, y_t_test = self._split_set(target_test)

        self.scale_max = y_c_train.max((0, 1))

        y_c_train = self.scale(y_c_train)
        y_t_train = self.scale(y_t_train)

        y_c_val = self.scale(y_c_val)
        y_t_val = self.scale(y_t_val)

        y_c_test = self.scale(y_c_test)
        y_t_test = self.scale(y_t_test)

        self.train_data = (x_c_train, y_c_train, x_t_train, y_t_train)
        self.val_data = (x_c_val, y_c_val, x_t_val, y_t_val)
        self.test_data = (x_c_test, y_c_test, x_t_test, y_t_test)

    def scale(self, x):
        return x / self.scale_max

    def inverse_scale(self, x):
        return x * self.scale_max

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--data_path", type=str, default="./data/metr_la/")
        parser.add_argument("--context_points", type=int, default=12)

        parser.add_argument("--target_points", type=int, default=12)


def METR_LA_Torch(data: World_Trade_Data, split: str):
    assert split in ["train", "val", "test"]
    if split == "train":
        tensors = data.train_data
    elif split == "val":
        tensors = data.val_data
    else:
        tensors = data.test_data
    tensors = [torch.from_numpy(x).float() for x in tensors]
    return TensorDataset(*tensors)


if __name__ == "__main__":
    data = World_Trade_Data(path=os.path.abspath('./spacetimeformer/data/world_trade/tomato_ts_df.csv'))
    dset = METR_LA_Torch(data, "test")
    breakpoint()
