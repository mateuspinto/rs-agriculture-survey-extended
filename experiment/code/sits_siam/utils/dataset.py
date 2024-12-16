from typing import Dict, Tuple
from tqdm.std import tqdm

import pathlib
import numpy as np
import torch
import pandas as pd
from typing import Union
import os


class SitsDatasetFromDataframe(torch.utils.data.Dataset):
    """
    Dataset class for satellite image time series data from pd.DataFrame.

    Processes input DataFrame into tensors suitable for training machine learning models, including data normalization
    and computation of sample weights.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame containing satellite images and labels. Expected columns include 'id', 'time', 'doy', and spectral bands.
    max_seq_len : int, optional
        Maximum sequence length for time series data (default is 70).
    num_features : int, optional
        Number of features per sample (default is 10).
    mean : List[float], optional
        Mean values for normalization of each feature (default values provided).
    std : List[float], optional
        Standard deviation values for normalization of each feature (default values provided).

    Attributes
    ----------
    xs : np.ndarray
        Array of input features of shape (num_samples, max_seq_len, num_features).
    doys : np.ndarray
        Array of day-of-year values corresponding to each time step.
    ys : np.ndarray
        Array of labels corresponding to each sample.

    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        max_seq_len: int = 70,
        num_features: int = 10,
        transform=None,
    ):
        BANDS = [
            "blue",
            "green",
            "red",
            "red_edge_1",
            "red_edge_2",
            "red_edge_3",
            "nir",
            "red_edge_4",
            "swir_1",
            "swir_2",
        ]

        dataframe = dataframe.sort_values(["id", "time"])

        ids = dataframe["id"].to_numpy()
        times = dataframe["time"].astype(int).to_numpy()
        doys = dataframe["doy"].to_numpy()
        bands_data = dataframe[BANDS].to_numpy()

        unique_ids, id_indices = np.unique(ids, return_inverse=True)
        num_ids = len(unique_ids)

        self.xs = np.zeros((num_ids, max_seq_len, num_features), dtype=np.half)
        self.doys = np.zeros((num_ids, max_seq_len), dtype=np.int16)

        self.xs[id_indices, times, :] = bands_data
        self.doys[id_indices, times] = doys

        labels_df = dataframe[["id", "label"]].drop_duplicates("id").set_index("id")
        self.ys = labels_df.loc[unique_ids, "label"].to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return self.ys.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get the sample corresponding to the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], torch.Tensor]
            A tuple containing:
            - A dictionary with keys:
                'doy': torch.Tensor of day-of-year values,
                'mask': torch.Tensor indicating missing data,
                'x': torch.Tensor of normalized input features,
                'weight': torch.Tensor of sample weights.
            - The corresponding label tensor.
        """
        sample = {"doy": self.doys[idx], "x": self.xs[idx], "y": self.ys[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class SitsDatasetFromFormerFormat(torch.utils.data.Dataset):
    def __init__(self, folder_path, max_seq_len, transform=None, limit=None):
        filenames = sorted(pathlib.Path(folder_path).glob("*.npz"))

        size = len(filenames) if limit is None else limit
        self.xs = np.zeros((size * 25, max_seq_len, 10), dtype=np.half)
        self.doys = np.zeros((size * 25, max_seq_len), dtype=np.int16)
        self.transform = transform

        for id in tqdm(range(size)):
            filename = filenames[id]
            data = np.load(filename)
            ts = data["ts"]  # Shape: (seq_len, 10, 5, 5)
            doy = data["doy"]  # Shape: (seq_len,)

            seq_len = ts.shape[0]
            if seq_len > max_seq_len:
                seq_len = max_seq_len

            ts_reshaped = ts[:seq_len].transpose(2, 3, 0, 1).reshape(-1, seq_len, 10)
            doy_replicated = np.tile(doy[:seq_len], (25, 1))

            start_idx = id * 25
            end_idx = start_idx + 25

            self.xs[start_idx:end_idx, :seq_len, :] = ts_reshaped
            self.doys[start_idx:end_idx, :seq_len] = doy_replicated

    def __len__(self) -> int:
        return self.ys.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        sample = {"doy": self.doys[idx], "x": self.xs[idx], "y": 0}

        if self.transform:
            sample = self.transform(sample)

        return sample


class SitsPretrainDatasetFromNpz(torch.utils.data.Dataset):
    def __init__(
        self, npz_dir: Union[pathlib.Path, str], world_size: int = 1, transform=None
    ):
        if isinstance(npz_dir, str):
            npz_dir = pathlib.Path(npz_dir)

        self.transform = transform
        # Detect the local rank from the environment variable
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.local_rank = local_rank
        self.world_size = world_size

        # List all .npz files and ensure divisibility by world_size
        self.npz_files = sorted(npz_dir.glob("*.npz"))
        total_files = len(self.npz_files)

        # Adjust the number of files to be divisible by world_size
        if total_files % world_size != 0:
            total_files = (total_files // world_size) * world_size
            self.npz_files = self.npz_files[:total_files]

        files_per_gpu = total_files // world_size

        # Assign a subset of files to each GPU
        start_file_idx = local_rank * files_per_gpu
        end_file_idx = start_file_idx + files_per_gpu
        self.npz_files = self.npz_files[start_file_idx:end_file_idx]

        # Preallocate arrays for 'ts' and 'doys'
        num_samples_per_file = 100000 * 25  # Based on your assumption
        total_samples = num_samples_per_file * len(self.npz_files)
        self.ts = np.zeros((total_samples, 45, 10), dtype=np.float16)
        self.doys = np.zeros((total_samples, 45), dtype=np.int16)

        # Load data into RAM for this GPU
        for n, npz_file_path in tqdm(
            enumerate(self.npz_files),
            total=len(self.npz_files),
            desc=f"Loading dataset into RAM on GPU {local_rank}...",
        ):
            data = np.load(npz_file_path)
            start_idx = n * num_samples_per_file
            end_idx = start_idx + num_samples_per_file
            self.ts[start_idx:end_idx] = data["ts"].astype(np.float16)
            self.doys[start_idx:end_idx] = data["doys"].astype(np.int16)

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, idx: int):
        sample = {"x": self.ts[idx], "doy": self.doys[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __add__(self, other):
        if not isinstance(other, SitsPretrainDatasetFromNpz):
            raise ValueError("Can only add SitsPretrainDatasetFromNpz objects together")

        combined_dataset = SitsPretrainDatasetFromNpz(
            npz_dir=pathlib.Path("."), world_size=self.world_size
        )
        combined_dataset.ts = np.concatenate((self.ts, other.ts), axis=0)
        combined_dataset.doys = np.concatenate((self.doys, other.doys), axis=0)
        return combined_dataset


class SitsFinetuneDatasetFromNpz(torch.utils.data.Dataset):
    def __init__(self, npz_file: Union[pathlib.Path, str, None] = None, transform=None):
        if npz_file is not None:
            if isinstance(npz_file, str):
                npz_file = pathlib.Path(npz_file)

            data = np.load(npz_file)
            self.ts = data["ts"].astype(np.float16)
            self.doys = data["doys"].astype(np.int16)
            self.ys = data["ys"].astype(np.int16)
        else:
            self.ts = None
            self.doys = None
            self.ys = None

        self.transform = transform

    def __len__(self):
        return len(self.ts) if self.ts is not None else 0

    def __getitem__(self, idx: int):
        sample = {"x": self.ts[idx], "doy": self.doys[idx], "y": self.ys[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def num_classes(self):
        return np.max(self.ys) + 1

    @property
    def sequence_length(self):
        return self.ts.shape[1]

    @property
    def average_real_sequence_length(self):
        non_zero_counts = np.count_nonzero(self.doys, axis=1)
        return non_zero_counts.mean()

    def __add__(self, other):
        if not isinstance(other, SitsFinetuneDatasetFromNpz):
            raise TypeError("Can only add SitsFinetuneDatasetFromNpz objects.")

        if self.ts is None or other.ts is None:
            raise ValueError("Cannot add empty datasets.")

        if self.sequence_length != other.sequence_length:
            raise ValueError(
                "Datasets must have the same sequence length to be summed."
            )

        # Concatenate the data from both datasets
        ts = np.concatenate([self.ts, other.ts], axis=0)
        doys = np.concatenate([self.doys, other.doys], axis=0)
        ys = np.concatenate([self.ys, other.ys], axis=0)

        # Create a new dataset with the concatenated data
        new_dataset = SitsFinetuneDatasetFromNpz(npz_file=None)
        new_dataset.ts = ts
        new_dataset.doys = doys
        new_dataset.ys = ys
        new_dataset.transform = self.transform

        return new_dataset


if __name__ == "__main__":
    import pandas as pd

    whole_df = pd.read_parquet("data/california_sits_bert_original.parquet")
    train_df = whole_df[whole_df["use_bert"] == 0].reset_index(drop=True)
    train_dataset = SitsDatasetFromDataframe(train_df, max_seq_len=45)
    print(train_dataset[0])
