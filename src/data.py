from pathlib import Path
from typing import Any, Literal
import kagglehub
import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

__all__ = ["dataset_factory", "AnomalyDetectionDataset"]


def dataset_factory(
    dataset: Literal["mqttset"],
    random_state: int | None = 42,
    feature_indices: list[int] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, "AnomalyDetectionDataset"], dict[str, Any]]:
    if dataset == "mqttset":
        return _fetch_mqttset(feature_indices=feature_indices, random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _fetch_mqttset(
    variants: list[str] | None = None,
    feature_indices: list[int] | None = None,
    random_state: int | None = 42,
) -> tuple[dict[str, "AnomalyDetectionDataset"], dict[str, Any]]:
    variants = variants or ["normal"]
    assert all([variant in ["normal", "reduced", "augmented"] for variant in variants])

    def clean(df: pl.DataFrame):
        # remove superfluous columns
        superfluous_columns = [
            "mqtt.conflag.qos",
            "mqtt.conack.flags.sp",
            "mqtt.conflag.willflag",
            "mqtt.conflag.retain",
            "mqtt.sub.qos",
            "mqtt.suback.qos",
            "mqtt.conflag.reserved",
            "mqtt.willmsg",
            "mqtt.willmsg_len",
            "mqtt.willtopic",
            "mqtt.willtopic_len",
            "mqtt.conack.flags.reserved",
        ]
        df = df.select(pl.exclude(superfluous_columns))

        # convert some columns with numbers represented as hex-strings to int
        string_flag_cols = [
            "mqtt.hdrflags",
            "tcp.flags",
            "mqtt.conflags",
            "mqtt.conack.flags",
            "mqtt.protoname",
        ]
        df = df.with_columns(
            (
                (
                    pl.col(col)
                    .map_elements(lambda x: int(x, 16), return_dtype=pl.Int64)
                    .alias(col)
                    if col != "mqtt.protoname"
                    else (pl.col(col) == "MQTT").alias("is_mqtt_proto")
                )
                for col in string_flag_cols
            )
        )
        df = df.select(pl.exclude(["mqtt.protoname"]))

        # feature engineering for `msg.msg`
        df = df.with_columns(pl.col("mqtt.msg").str.len_chars().alias("mqtt.msg.length"))

        # create anomaly label for training
        coding_bin = {
            "legitimate": 0,
            "dos": 1,
            "malformed": 1,
            "bruteforce": 1,
            "slowite": 1,
            "flood": 1,
        }
        df = df.with_columns(
            pl.col("target").replace_strict(coding_bin, return_dtype=pl.Int64).alias("label")
        )

        x = df.drop(["target", "label", "mqtt.msg"])
        x = x.select(sorted(x.columns))  # sort for consistency
        y = df["label"].to_numpy()
        messages = df["mqtt.msg"].to_list()
        target_names = df["target"].to_list()
        return x.to_numpy(), y, messages, target_names, x.columns

    # 1. download dataset
    # ------------------
    mqtt_path = Path(kagglehub.dataset_download("cnrieiit/mqttset")) / "Data/FINAL_CSV"
    variant_suffixes = {
        "normal": "",
        "reduced": "_reduced",
        "augmented": "_augmented",
    }

    # 2. prepare training data
    # ---------------------
    df_train_val = pl.concat(
        [
            pl.read_csv(
                mqtt_path / f"train70{variant_suffixes[variant]}.csv", infer_schema_length=10_000
            )
            for variant in variants
        ],
        how="vertical",
    )
    x_train_val, y_train_val, messages_train_val, label_names_train_val, feature_names = clean(
        df_train_val
    )
    scaler = StandardScaler()
    x_train_val = scaler.fit_transform(x_train_val)

    num_negatives = np.sum(y_train_val == 0)
    num_positives = np.sum(y_train_val == 1)
    pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32)

    (
        x_train,
        x_val,
        y_train,
        y_val,
        messages_train,
        messages_val,
        label_names_train,
        label_names_val,
    ) = train_test_split(
        x_train_val,
        y_train_val,
        messages_train_val,
        label_names_train_val,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train_val,
    )

    # 3. prepare testing data
    # -----------------------
    df_test = pl.concat(
        [
            pl.read_csv(
                mqtt_path / f"test30{variant_suffixes[variant]}.csv", infer_schema_length=10_000
            )
            for variant in variants
        ],
        how="vertical",
    )
    x_test, y_test, messages_test, label_names_test, feature_names_1 = clean(df_test)
    x_test = scaler.transform(x_test)

    assert feature_names == feature_names_1, "Feature names do not match."

    return (
        {
            "train": AnomalyDetectionDataset(
                x_train, y_train, messages_train, label_names_train, feature_names, feature_indices
            ),
            "val": AnomalyDetectionDataset(
                x_val, y_val, messages_val, label_names_val, feature_names, feature_indices
            ),
            "test": AnomalyDetectionDataset(
                x_test, y_test, messages_test, label_names_test, feature_names, feature_indices
            ),
        },
        {"pos_weight": pos_weight},
    )


class AnomalyDetectionDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        messages: list[str],
        label_names: list[str],
        feature_names: list[str],
        feature_indices: list[int] | None = None,
    ):
        # sanity check
        assert features.shape[0] == labels.shape[0] == len(messages) == len(label_names)
        assert features.shape[1] == len(feature_names)

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.messages = messages
        self.label_names = label_names
        self.feature_names = feature_names
        self.feature_indices = feature_indices or list(range(len(feature_names)))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx, self.feature_indices], self.labels[idx]
