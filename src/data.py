import ipaddress
import json
import socket
import struct
from pathlib import Path
from typing import Any, Literal

import kagglehub
import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

__all__ = [
    "dataset_factory",
    "AnomalyDetectionDataset",
    "load_iiotset",
    "load_mqttset",
    "load_data",
    "load_x_iiotd",
    "find_features",
]


def dataset_factory(
    dataset: Literal["mqttset", "iiotset"],
    data_dir: Path,
    random_state: int | None = 42,
    features: list[str] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, "AnomalyDetectionDataset"], dict[str, Any]]:

    data, extra = load_data(dataset, data_dir, random_state, **kwargs)
    pos_weight = extra["pos_weight"]

    return {
        split: AnomalyDetectionDataset(data, split, features) for split in ["train", "val", "test"]
    }, {"pos_weight": pos_weight}


def load_data(
    dataset: Literal["mqttset", "iiotset", "x-iiotd"],
    data_dir: Path,
    random_state: int | None = 42,
    **kwargs: Any,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    data_path = data_dir / f"{dataset}.parquet"
    if not data_path.exists():
        if dataset == "mqttset":
            data = load_mqttset(
                random_state=random_state,
                **kwargs,
            )
        elif dataset == "iiotset":
            data = load_iiotset(random_state=random_state)
        elif dataset == "x-iiotd":
            data = load_x_iiotd(random_state=random_state)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        data.write_parquet(data_path)
    else:
        data = pl.read_parquet(data_path)

    y_train_val = data.filter(pl.col("split").is_in(["train", "val"]))["attack_label"].to_numpy()
    num_negatives = np.sum(y_train_val == 0)
    num_positives = np.sum(y_train_val == 1)
    pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32)

    return data, {"pos_weight": pos_weight}


def load_mqttset(
    variants: list[str] | None = None,
    random_state: int | None = 42,
) -> pl.DataFrame:
    variants = variants or ["normal"]
    assert all([variant in ["normal", "reduced", "augmented"] for variant in variants])

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
                mqtt_path / f"train70{variant_suffixes[variant]}.csv",
                infer_schema_length=10_000,
            )
            for variant in variants
        ],
        how="vertical",
    )
    df_test = pl.concat(
        [
            pl.read_csv(
                mqtt_path / f"test30{variant_suffixes[variant]}.csv",
                infer_schema_length=10_000,
            )
            for variant in variants
        ],
        how="vertical",
    )
    df = pl.concat([df_train_val, df_test], how="vertical")
    num_train_val = df_train_val.shape[0]

    # data preprocessing
    # -----------------------
    # (i) remove superfluous columns
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

    # (ii) create attack label for training
    attack_label_coding = {
        "legitimate": 0,
        "dos": 1,
        "malformed": 1,
        "bruteforce": 1,
        "slowite": 1,
        "flood": 1,
    }
    df = df.with_columns(
        pl.col("target")
        .replace_strict(attack_label_coding, return_dtype=pl.Int64)
        .alias("attack_label")
    )
    df = df.rename({"target": "attack_type"})
    attack_types = df["attack_type"]
    attack_labels = df["attack_label"].to_numpy()
    df = df.drop("attack_label", "attack_type")

    # (ii) convert categorical columns
    categorical_column_names = sorted(
        [
            "mqtt.hdrflags",
            "tcp.flags",
            "mqtt.conflags",
            "mqtt.conack.flags",
            "mqtt.protoname",
        ]
    )
    categorical_columns = df[categorical_column_names].to_numpy()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical_columns_train_val = encoder.fit_transform(categorical_columns[:num_train_val])
    categorical_columns_test = encoder.transform(categorical_columns[num_train_val:])
    categorical_columns = np.concatenate(
        [categorical_columns_train_val, categorical_columns_test], axis=0
    )
    df = df.drop(categorical_column_names)

    # (iii) feature engineering
    df = df.with_columns(pl.col("mqtt.msg").str.len_chars().alias("mqtt.msg.length"))
    df = df.drop("mqtt.msg")

    # (iv) standardize numeric columns
    numerical_column_names = sorted(df.columns)
    df = df[numerical_column_names]
    numerical_columns = df.to_numpy()
    scaler = StandardScaler()
    numerical_columns_train_val = scaler.fit_transform(numerical_columns[:num_train_val])
    numerical_columns_test = scaler.transform(numerical_columns[num_train_val:])
    numerical_columns = np.concatenate(
        [numerical_columns_train_val, numerical_columns_test], axis=0
    )

    # (v) combine everything back to a single dataframe
    data_dict = {}

    for idx, name in enumerate(numerical_column_names):
        data_dict[name] = numerical_columns[:, idx]

    idx = 0
    for name, categories in zip(categorical_column_names, encoder.categories_):
        for i in range(len(categories)):
            data_dict[f"{name}__{i}"] = categorical_columns[:, idx + i]
        idx += len(categories)

    data_dict["attack_label"] = (
        attack_labels.flatten() if len(attack_labels.shape) > 1 else attack_labels
    )
    data_dict["attack_type"] = attack_types

    indices_train_val = list(range(num_train_val))
    indices_train, indices_val = train_test_split(
        indices_train_val,
        test_size=0.15,
        random_state=random_state,
        stratify=attack_labels[indices_train_val],
    )
    splits = np.full((len(df),), "train")
    splits[indices_train] = "train"
    splits[indices_val] = "val"
    splits[num_train_val:] = "test"
    data_dict["split"] = splits.tolist()

    return pl.DataFrame(data_dict)


def load_iiotset(random_state: int | None = 42) -> pl.DataFrame:
    # 1. download dataset
    # ------------------
    data_path = (
        Path(
            kagglehub.dataset_download(
                "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"
            )
        )
        / "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
    )
    df = pl.read_csv(
        data_path,
        infer_schema_length=1000,
        schema_overrides={
            "dns.qry.name.len": pl.Utf8,
            "tcp.srcport": pl.Utf8,
            "http.request.method": pl.Utf8,
            "http.request.full_uri": pl.Utf8,
            "http.request.version": pl.Utf8,
            "http.request.uri.query": pl.Utf8,
            "http.file_data": pl.Utf8,
            "http.referer": pl.Utf8,
        },
    )

    # 2. Data preprocessing
    # --------------------

    # (i) remove superfluous columns
    superfluous_columns = [
        "mqtt.msg_decoded_as",
        "dns.qry.type",
        "http.tls_port",
        "icmp.unused",
    ]
    df = df.select(pl.exclude(superfluous_columns))

    # (ii) feature engineering on some columns
    df = df.with_columns(
        # some frame.time are not valid dateTime entries. (only 10% though)
        pl.col("frame.time")
        .str.strip_chars()
        .str.strptime(pl.Datetime, "%Y %H:%M:%S.%9f", strict=False)
        .dt.timestamp("ms")
        .is_not_null()
        .alias("frame.time.valid"),
        # use length to summarize data
        pl.col("tcp.options").str.len_chars().alias("tcp.options.length"),
        # use common subnet mask
        pl.struct(["ip.src_host", "ip.dst_host"])
        .map_elements(
            lambda x: find_shared_subnet_mask(x["ip.src_host"], x["ip.dst_host"]),
            return_dtype=pl.Utf8,
        )
        .alias("ip.subnet"),
        # convert to number
        pl.col("tcp.srcport")
        .map_elements(try_convert_to_float, return_dtype=pl.Float64)
        .alias("tcp.srcport.num"),
        # use length
        pl.col("mqtt.msg").str.len_chars().alias("mqtt.msg.length"),
    )

    # (iii) discard some columns
    df = df.select(
        pl.exclude(
            [
                "frame.time",
                "tcp.payload",
                "tcp.options",
                "ip.src_host",
                "ip.dst_host",
                "tcp.srcport",
                "http.request.full_uri",
                "http.request.uri.query",
                "http.file_data",
                "mqtt.msg",
                "arp.src.proto_ipv4",
                "arp.dst.proto_ipv4",
                "arp.subnet.proto_ipv4",
                # numerical
                "arp.opcode",
                "tcp.flags",
                "dns.qry.qu",
                "http.content_length",
                "icmp.transmit_timestamp",
                "mbtcp.trans_id",
                "udp.port",
                "udp.time_delta",
                "tcp.len",
                "dns.qry.name",
                "icmp.checksum",
                "icmp.seq_le",
                "tcp.checksum",
                "tcp.ack",
                "tcp.connection.synack",
                "dns.retransmit_request_in",
                "dns.retransmit_request",
                "mqtt.conflag.cleansess",
                "tcp.flags.ack",
                "tcp.connection.syn",
                "tcp.connection.rst",
                "tcp.connection.fin",
                "arp.hw.size",
            ]
        )
    )

    # (iv) split data in train/val/test splits
    attack_types = df["Attack_type"].to_list()
    attack_labels = df["Attack_label"].to_numpy()
    df = df.drop("Attack_type", "Attack_label")
    indices = list(range(len(df)))
    indices_train_val, indices_test = train_test_split(
        indices,
        test_size=0.15,
        random_state=random_state,
        stratify=attack_labels,
    )
    _, indices_val = train_test_split(
        indices_train_val,
        test_size=0.15,
        random_state=random_state,
        stratify=attack_labels[indices_train_val],
    )

    splits = np.full((len(df),), "train")
    splits[indices_val] = "val"
    splits[indices_test] = "test"

    # (v) encode categorical data
    categorical_column_names = sorted(
        [
            "frame.time.valid",
            "ip.subnet",
            "mqtt.msg.length",  # might be overfitting lol
            "http.request.version",
            "mqtt.conack.flags",
            "http.request.method",
            "dns.qry.name.len",
            "http.referer",
            "mqtt.topic",
            "mqtt.protoname",
        ]
    )

    categorical_columns = df[categorical_column_names].to_numpy()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical_columns_train_val = encoder.fit_transform(categorical_columns[indices_train_val])
    categorical_columns_test = encoder.transform(categorical_columns[indices_test])
    categorical_columns = np.concatenate(
        [categorical_columns_train_val, categorical_columns_test], axis=0
    )
    df = df.drop(categorical_column_names)

    # (vi) scale numeric data
    numerical_column_names = sorted(df.columns)
    df = df[numerical_column_names]
    numerical_columns = df.to_numpy()
    scaler = StandardScaler()
    numerical_columns_train_val = scaler.fit_transform(numerical_columns[indices_train_val])
    numerical_columns_test = scaler.transform(numerical_columns[indices_test])
    numerical_columns = np.concatenate(
        [numerical_columns_train_val, numerical_columns_test], axis=0
    )

    # (vii) combine data into a single dataframe
    data_dict = {}

    for idx, name in enumerate(numerical_column_names):
        data_dict[name] = numerical_columns[:, idx]

    idx = 0
    for name, categories in zip(categorical_column_names, encoder.categories_):
        for i in range(len(categories)):
            data_dict[f"{name}__{i}"] = categorical_columns[:, idx + i]
        idx += len(categories)

    data_dict["attack_label"] = (
        attack_labels.flatten() if len(attack_labels.shape) > 1 else attack_labels
    )
    data_dict["attack_type"] = attack_types
    data_dict["split"] = splits.tolist()

    return pl.DataFrame(data_dict)


def load_x_iiotd(random_state: int | None = 42) -> pl.DataFrame:
    # 1. download dataset
    # ------------------
    dataset_path = (
        Path(kagglehub.dataset_download("munaalhawawreh/xiiotid-iiot-intrusion-dataset"))
        / "X-IIoTID dataset.csv"
    )

    df = pl.read_csv(dataset_path, infer_schema_length=100_000)

    # 2. data cleaning & preprocessing
    # ------------------
    # (i) remove superfluous columns
    superfluous_columns = ["is_SYN_with_RST", "Bad_checksum"]
    df = df.select(pl.exclude(superfluous_columns))

    # (ii) create attack_label column
    attack_label_coding = {"Normal": 0, "Attack": 1}
    df = df.with_columns(
        pl.col("class3")
        .replace_strict(attack_label_coding, return_dtype=pl.Int64)
        .alias("attack_label")
    )
    df = df.drop("class3", "class1")
    df = df.rename({"class2": "attack_type"})

    # (iii) remove rows with "missing" values
    is_missing_predicate = None
    for column, dtype in df.schema.items():
        if dtype == pl.Utf8:
            p = pl.col(column).is_in(["-", "", "?", "#DIV/0!"])
            is_missing_predicate = p if is_missing_predicate is None else is_missing_predicate | p
    assert is_missing_predicate is not None
    df = df.filter((~is_missing_predicate) & (pl.col("Avg_user_time") != "aza"))

    # (iv) convert 'numerical' strings to numbers
    numerical_column_names = [
        "Duration",
        "Timestamp",
        "Scr_bytes",
        "Des_bytes",
        "missed_bytes",
        "Scr_pkts",
        "Scr_ip_bytes",
        "Des_pkts",
        "Des_ip_bytes",
        "total_bytes",
        "total_packet",
        "paket_rate",
        "byte_rate",
        "Scr_packts_ratio",
        "Des_pkts_ratio",
        "Scr_bytes_ratio",
        "Des_bytes_ratio",
        "Avg_user_time",
        "Std_user_time",
        "Avg_nice_time",
        "Std_nice_time",
        "Avg_system_time",
        "Std_system_time",
        "Avg_iowait_time",
        "Std_iowait_time",
        "Avg_ideal_time",
        "Std_ideal_time",
        "Avg_tps",
        "Std_tps",
        "Avg_rtps",
        "Std_rtps",
        "Avg_wtps",
        "Std_wtps",
        "Avg_ldavg_1",
        "Std_ldavg_1",
        "Avg_kbmemused",
        "Std_kbmemused",
        "Avg_num_Proc/s",
        "Std_num_proc/s",
        "Avg_num_cswch/s",
        "std_num_cswch/s",
        "anomaly_alert",
    ]
    numerical_column_names = sorted(numerical_column_names)
    for column in numerical_column_names:
        if column == "anomaly_alert":
            anomaly_alert_label = {"FALSE": 0, "TRUE": 1}
            df = df.with_columns(
                pl.col(column)
                .replace_strict(anomaly_alert_label, return_dtype=pl.Int64)
                .alias(column)
            )
        else:
            df = df.with_columns(
                pl.col(column).str.strip_chars().cast(pl.Float64, strict=True).alias(column)
            )

    # (v) feature engineering
    df = df.with_columns(
        pl.struct(["Des_IP", "Scr_IP"])
        .map_elements(
            lambda x: find_shared_subnet_mask(x["Des_IP"], x["Scr_IP"]), return_dtype=pl.Utf8
        )
        .alias("IP.Subnet")
    )

    # (vi) drop others
    df = df.drop("Des_IP", "Scr_IP", "Date")

    # (vii) split data in train/val/test splits
    attack_labels = df["attack_label"].to_numpy()
    attack_types = df["attack_type"].to_numpy()
    indices = list(range(len(df)))
    indices_train_val, indices_test = train_test_split(
        indices,
        test_size=0.15,
        random_state=random_state,
        stratify=attack_labels,
    )
    _, indices_val = train_test_split(
        indices_train_val,
        test_size=0.15,
        random_state=random_state,
        stratify=attack_labels[indices_train_val],
    )

    splits = np.full((len(df),), "train")
    splits[indices_val] = "val"
    splits[indices_test] = "test"

    # (viii) categorical
    categorical_column_names = [
        "Service",
        "Protocol",
        "OSSEC_alert_level",
        "Conn_state",
        "is_with_payload",
        "is_pure_ack",
        "Is_SYN_ACK",
        "is_syn_only",
        "OSSEC_alert",
        "Login_attempt",
        "Succesful_login",
        "File_activity",
        "Process_activity",
        "read_write_physical.process",
        "is_privileged",
        "FIN or RST",
    ]
    categorical_column_names = sorted(categorical_column_names)
    categorical_columns = df[categorical_column_names].to_numpy()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical_columns_train_val = encoder.fit_transform(categorical_columns[indices_train_val])
    categorical_columns_test = encoder.transform(categorical_columns[indices_test])
    categorical_columns = np.concatenate(
        [categorical_columns_train_val, categorical_columns_test], axis=0
    )
    df = df.drop(categorical_column_names)

    # (ix) scale numeric data
    df = df[numerical_column_names]
    numerical_columns = df.to_numpy()
    scaler = StandardScaler()
    numerical_columns_train_val = scaler.fit_transform(numerical_columns[indices_train_val])
    numerical_columns_test = scaler.transform(numerical_columns[indices_test])
    numerical_columns = np.concatenate(
        [numerical_columns_train_val, numerical_columns_test], axis=0
    )

    # (x) combine data into a single dataframe
    data_dict = {}

    for idx, name in enumerate(numerical_column_names):
        data_dict[name] = numerical_columns[:, idx]

    idx = 0
    for name, categories in zip(categorical_column_names, encoder.categories_):
        for i in range(len(categories)):
            data_dict[f"{name}__{i}"] = categorical_columns[:, idx + i]
        idx += len(categories)

    data_dict["attack_label"] = (
        attack_labels.flatten() if len(attack_labels.shape) > 1 else attack_labels
    )
    data_dict["attack_type"] = attack_types
    data_dict["split"] = splits.tolist()

    return pl.DataFrame(data_dict)


def ip_to_int(ip):
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except (OSError, socket.error):
        return None


def int_to_ip(i):
    return socket.inet_ntoa(struct.pack("!I", i))


def find_shared_subnet_mask(ip1: str, ip2: str, default_value: str = "null") -> str | None:
    try:
        ipaddress.IPv4Address(ip1)
        ipaddress.IPv4Address(ip2)
    except ipaddress.AddressValueError:
        return default_value

    ip1_int = ip_to_int(ip1)
    ip2_int = ip_to_int(ip2)

    if ip1_int is None or ip2_int is None:
        return default_value

    xor = ip1_int ^ ip2_int
    if xor == 0:
        mask_len = 32
    else:
        mask_len = 32 - xor.bit_length()

    mask_int = (0xFFFFFFFF << (32 - mask_len)) & 0xFFFFFFFF
    return int_to_ip(mask_int)


def try_convert_to_float(x, default_value=-1):
    try:
        num = float(x)
        return int(num) if num == int(num) else num
    except ValueError:
        return default_value


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


class AnomalyDetectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        df: pl.DataFrame,
        split: str | None = None,
        features_to_use: list[str] | list[int] | None = None,
    ):

        split = split.strip().lower() if split else split
        df = df.filter(pl.col("split") == split) if split else df

        attack_labels = df["attack_label"].to_list()
        attack_types = df["attack_type"].to_list()
        df = df.drop(["attack_label", "attack_type", "split"])

        self.inputs = torch.from_numpy(df.to_numpy().astype(np.float32))
        self.attack_labels = torch.tensor(attack_labels, dtype=torch.float32)
        self.attack_types = attack_types
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
        return self.inputs[idx, self._input_indices], self.attack_labels[idx]


class Tokenizer:
    def __init__(self, chars: list[str]):
        # 0 for padding, 1 for unk
        self.char_to_int = {char: i + 2 for i, char in enumerate(chars)}
        self.char_to_int["<PAD>"] = 0
        self.char_to_int["<UNK>"] = 1
        self.int_to_char = {i: char for char, i in self.char_to_int.items()}

    def __len__(self):
        return len(self.char_to_int)

    @property
    def pad_token_id(self):
        return self.char_to_int["<PAD>"]

    @property
    def unknown_token_id(self):
        return self.char_to_int["<UNK>"]

    def tokenize_messages(self, messages: list[str], max_len: int) -> np.ndarray:
        tokenized_msgs = []
        for msg_str in tqdm(messages, desc="Tokenizing messages"):
            tokenized_msgs.append(self.tokenize_message(message=msg_str, max_len=max_len))
        return np.array(tokenized_msgs, dtype=np.int64)

    def tokenize_message(self, message: str, max_len: int) -> list[int]:
        char_map = self.char_to_int
        tokens = [char_map.get(char, char_map["<UNK>"]) for char in message]

        if len(tokens) < max_len:
            tokens.extend([char_map["<PAD>"]] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
        return tokens

    @staticmethod
    def from_json(json_path: Path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return Tokenizer(data["chars"])

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump({"chars": list(self.char_to_int.keys())}, f)

    @staticmethod
    def create_from_messages(messages: list[str]) -> "Tokenizer":
        chars = set()
        for message in messages:
            chars.update(message)
        return Tokenizer(sorted(list(chars)))
