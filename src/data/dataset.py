from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset

from .iiotset import load_iiotset
from .mqttset import load_mqttset
from .x_iiotd import load_x_iiotd


def dataset_factory(
    dataset: Literal["mqttset", "iiotset", "x-iiotd"],
    data_dir: Path,
    sequence_length: int,
    train_val_test_split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_state: int | None = 42,
    features: list[str] | None = None,
) -> tuple[dict[str, "NetworkTrafficDataset"], dict[str, Any]]:
    data, extra = load_data(
        dataset,
        data_dir,
        sequence_length=sequence_length,
        train_val_test_split_ratio=train_val_test_split_ratio,
        random_state=random_state,
    )
    pos_weight = extra["pos_weight"]

    return {
        split: NetworkTrafficDataset(data, sequence_length, split, features)
        for split in ["train", "val", "test"]
    }, {"pos_weight": pos_weight}


def find_features(df: pl.DataFrame) -> dict[str, list[int]]:
    feature_to_input_indices: dict[str, list[int]] = {}
    for idx, column in enumerate(
        [column for column, dtype in df.schema.items() if dtype.is_numeric()]
    ):
        name = column.split("__")[0]
        indices = feature_to_input_indices.get(name, [])
        indices.append(idx)
        feature_to_input_indices[name] = indices
    return feature_to_input_indices


def load_data(
    dataset: Literal["mqttset", "iiotset", "x-iiotd"],
    data_dir: Path,
    sequence_length: int,
    train_val_test_split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_state: int | None = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    chunked_data_path = data_dir / f"{dataset}_{sequence_length}.parquet"
    if not chunked_data_path.exists():
        processed_data_path = data_dir / f"{dataset}.parquet"
        if not processed_data_path.exists():
            if dataset == "mqttset":
                data = load_mqttset()
            elif dataset == "iiotset":
                data = load_iiotset()
            elif dataset == "x-iiotd":
                data = load_x_iiotd()
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
            data.write_parquet(processed_data_path)
        else:
            data = pl.read_parquet(processed_data_path)

        data = _split_data(data, sequence_length, train_val_test_split_ratio, random_state)
        data.write_parquet(chunked_data_path)
    else:
        data = pl.read_parquet(chunked_data_path)

    labels_train_val = (
        data.filter(pl.col("split").is_in(["train", "val"]))
        .group_by("sequence_id", maintain_order=True)
        .agg(pl.last("sequence_label"))["sequence_label"]
        .to_numpy()
    )
    num_negatives = np.sum(labels_train_val == 0)
    num_positives = np.sum(labels_train_val == 1)
    pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32)

    return data, {"pos_weight": pos_weight}


def _split_data(
    df: pl.DataFrame,
    s_length: int,
    train_val_test_split_ratio: tuple[float, float, float],
    random_state: int | None,
) -> pl.DataFrame:
    special_column_names = [
        "attack_label",
        "attack_type",
        "session_id",
        "timestamp",
        "sequence_id",
    ]

    diff = set(special_column_names) - set(df.columns + ["sequence_id"])
    assert len(diff) == 0, f"Missing columns: {diff}"

    # fill nulls
    for column in df.columns:
        if not df[column].has_nulls():
            continue
        if df.schema[column] == pl.Utf8:
            df = df.with_columns(pl.col(column).fill_null("NONE").alias(column))
        elif df.schema[column].is_numeric():
            df = df.with_columns(
                pl.col(column).fill_null(-1).alias(column),
                pl.col(column).is_not_null().alias(f"{column}.is_valid"),
            )
        elif df.schema[column] == pl.Boolean:
            df = df.with_columns(
                pl.col(column).fill_null(False).alias(column),
                pl.col(column).is_not_null().alias(f"{column}.is_valid"),
            )
        else:
            raise ValueError(f"Unknown dtype: {df.schema[column]} for column: {column}")

    numerical_column_names = sorted(
        [
            column
            for column, dtype in df.schema.items()
            if dtype.is_numeric() and column not in special_column_names
        ]
    )
    boolean_column_names = sorted(
        [
            column
            for column, dtype in df.schema.items()
            if dtype == pl.Boolean and column not in special_column_names
        ]
    )
    categorical_column_names = sorted(
        [
            column
            for column, dtype in df.schema.items()
            if dtype == pl.Utf8 and column not in special_column_names
        ]
    )
    print(numerical_column_names)
    print(categorical_column_names)
    print(boolean_column_names)

    #
    # create sequences
    # ---------
    df = _create_sequences(df, s_length)
    labels = (
        df.group_by("sequence_id", maintain_order=True)
        .agg(pl.last("sequence_label"))["sequence_label"]
        .to_numpy()
    )

    # split data into train/val/test splits
    num_sequences = len(labels)
    indices = np.arange(num_sequences)
    num_train = int(train_val_test_split_ratio[0] * num_sequences)
    num_val = int(train_val_test_split_ratio[1] * num_sequences)
    num_train_val = num_train + num_val
    num_test = num_sequences - num_train_val
    # type: np.ndarray, np.ndarray
    indices_train_val, indices_test = train_test_split(
        indices,
        test_size=num_test,
        random_state=random_state,
        stratify=labels,
    )
    # type: np.ndarray, np.ndarray
    indices_train, indices_val = train_test_split(
        indices_train_val,
        train_size=num_train,
        random_state=random_state,
        stratify=labels[indices_train_val],
    )

    indices_train, indices_val, indices_test = [
        (np.reshape(indices, (-1, 1)).repeat(s_length, axis=1).flatten() * s_length)
        + np.tile(np.arange(s_length), indices.shape[0])
        for indices in [indices_train, indices_val, indices_test]
    ]
    df = pl.concat(
        [df[indices_train], df[indices_val], df[indices_test]],
        how="vertical",
    )

    splits = (
        ["train"] * len(indices_train) + ["val"] * len(indices_val) + ["test"] * len(indices_test)
    )
    df = df.with_columns(pl.Series("split", splits, dtype=pl.Utf8))
    num_train_val = num_train_val * s_length

    # Encode data
    # --------------

    # encode categorical data
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical_columns = df[categorical_column_names].to_numpy()
    categorical_columns_train_val = encoder.fit_transform(categorical_columns[:num_train_val])
    categorical_columns_test = encoder.transform(categorical_columns[num_train_val:])
    categorical_columns = np.concatenate(
        [categorical_columns_train_val, categorical_columns_test], axis=0
    )

    # scale numeric data
    numerical_columns = df[numerical_column_names].to_numpy()
    scaler = StandardScaler()
    numerical_columns_train_val = scaler.fit_transform(numerical_columns[:num_train_val])
    numerical_columns_test = scaler.transform(numerical_columns[num_train_val:])
    numerical_columns = np.concatenate(
        [numerical_columns_train_val, numerical_columns_test], axis=0
    )

    # combine data into a single dataframe
    data_dict = {}

    for idx, name in enumerate(numerical_column_names):
        data_dict[name] = numerical_columns[:, idx]

    for name in boolean_column_names:
        data_dict[name] = df[name]

    idx = 0
    for name, categories in zip(categorical_column_names, encoder.categories_):
        for i in range(len(categories)):
            data_dict[f"{name}__{i}"] = categorical_columns[:, idx + i]
        idx += len(categories)

    data_dict["sequence_id"] = df["sequence_id"]
    data_dict["sequence_label"] = df["sequence_label"]
    data_dict["attack_type"] = df["attack_type"]
    data_dict["attack_label"] = df["attack_label"]
    data_dict["split"] = splits

    df = pl.DataFrame(data_dict)
    df = df[sorted(df.columns)]
    return df


def _create_sequences(df: pl.DataFrame, s_length: int) -> pl.DataFrame:
    return (
        df.sort(["session_id", "timestamp"])
        .with_columns(
            pl.col("session_id")
            .cum_count()
            .over("session_id")
            .cast(pl.Float64)
            .truediv(s_length)
            .ceil()
            .cast(pl.Int64)
            .alias("session_chunk_index"),
        )
        .with_columns(
            (pl.col("session_id") + "||" + pl.col("session_chunk_index").cast(pl.Utf8)).alias(
                "sequence_id"
            )
        )
        .with_columns(
            pl.len().over("sequence_id").alias("sequence_length"),
            (pl.col("attack_label").cast(pl.Int64).sum().over("sequence_id") > 0)
            .cast(pl.Int64)
            .alias("sequence_label"),
        )
        .filter(pl.col("sequence_length") == s_length)
        .drop("session_id", "timestamp", "session_chunk_index")
    )


class NetworkTrafficDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        df: pl.DataFrame,
        sequence_length: int,
        split: str | None = None,
        features_to_use: list[str] | list[int] | None = None,
    ):

        split = split.strip().lower() if split else split
        df = df.filter(pl.col("split") == split) if split else df

        labels = (
            df.group_by("sequence_id", maintain_order=True)
            .agg(pl.last("sequence_label"))["sequence_label"]
            .to_list()
        )
        df = df.drop("sequence_label", "attack_label", "attack_type", "split", "sequence_id")
        input_dim = len(df.columns)

        self.inputs = (
            torch.from_numpy(df.to_numpy().astype(np.float32))
            .contiguous()
            .view(-1, sequence_length, input_dim)
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self._feature_to_input_indices = find_features(df)

        # convert feature indices to names
        feature_names = list(self._feature_to_input_indices.keys())
        features_to_use = features_to_use or []
        for i in range(len(features_to_use)):
            item = features_to_use[i]
            if isinstance(item, int):
                features_to_use[i] = feature_names[item]

        self._features_to_use = features_to_use or feature_names
        self._input_indices = []
        for feature in self._features_to_use:
            self._input_indices.extend(self._feature_to_input_indices[feature])

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_to_input_indices.keys())

    @property
    def num_features(self) -> int:
        return len(self._feature_to_input_indices)

    def compute_input_indices(self, features: list[int] | list[str]) -> list[int]:
        indices = []
        for idx, (name, input_indices) in enumerate(self._feature_to_input_indices.items()):
            if idx in features or name in features:
                indices.extend(input_indices)
        return indices

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx, :, self._input_indices], self.labels[idx]
