from pathlib import Path

import kagglehub
import polars as pl

from .transforms import find_shared_subnet_mask, get_port_feature_expressions


def load_mqttset() -> pl.DataFrame:

    # 1. download dataset
    # ------------------
    data_path = Path(kagglehub.dataset_download("cnrieiit/mqttset")) / "Data/CSV"

    df_list = [
        pl.read_csv(
            path, infer_schema_length=10_000, schema_overrides={"mqtt.msg": pl.Utf8}
        ).with_columns(
            pl.lit(path.stem).alias("attack_type"),
            pl.lit(path.stem.startswith("legitimate")).alias("attack_label"),
        )
        for path in data_path.iterdir()
    ]

    schema = {
        column: _find_common_supertype([frame.schema[column] for frame in df_list])
        for column in df_list[0].schema.keys()
    }
    df: pl.DataFrame = pl.concat([df.cast(schema) for df in df_list], how="vertical")

    # data preprocessing
    # -----------------------
    # (i) remove superfluous & useless columns
    superfluous_columns = [
        "frame.time_invalid",
        "mqtt.willmsg",
        "mqtt.willmsg_len",
        "mqtt.willtopic",
        "mqtt.willtopic_len",
        "ip.proto",
    ]
    df = df.drop(superfluous_columns)

    # columns with too specific information
    # the original authors drop some of these fields
    useless_columns = [
        "tcp.checksum",
        "eth.src",
        "eth.dst",
        "mqtt.passwd_len",
        "mqtt.passwd",
        "mqtt.topic",
        "mqtt.topic_len",
        "mqtt.clientid",
        "mqtt.clientid_len",
        "mqtt.username",
        "mqtt.username_len",
        "frame.time_delta_displayed",
        "tcp.window_size_value",
        "mqtt.clientid",
        # ???
        "tcp.analysis.initial_rtt",
        "frame.len",
        "frame.cap_len",
        "frame.number",
        "frame.time_relative",
        "frame.time_delta",
    ]
    df = df.drop(useless_columns)

    # (ii) cast data to right types
    df = df.with_columns(
        *[pl.col(column).cast(pl.Int64) for column in ["mqtt.msgid", "tcp.len", "mqtt.len"]],
        # *[pl.col(column).cast(pl.Float32) for column in ["tcp.analysis.initial_rtt"]],
    )

    # (iii) feature engineering
    content_columns = [
        "mqtt.msg",
    ]
    df = df.with_columns(
        *[
            pl.col(column).fill_null("").str.strip_chars().str.len_chars().alias(f"{column}.len")
            for column in content_columns
        ],
        pl.struct(["ip.src", "ip.dst"])
        .map_elements(
            lambda x: find_shared_subnet_mask(x["ip.src"], x["ip.dst"]), return_dtype=pl.Utf8
        )
        .alias("ip.subnet"),
        *get_port_feature_expressions("tcp.srcport"),
        *get_port_feature_expressions("tcp.dstport"),
    ).drop(*content_columns, "ip.src", "ip.dst", "tcp.srcport", "tcp.dstport")

    # (iv) rename some columns
    df = df.with_columns(pl.col("tcp.stream").cast(pl.Utf8)).rename(
        {"tcp.stream": "session_id", "frame.time_epoch": "timestamp"}
    )

    return df


def _find_common_supertype(dtypes):
    dtype_precedence = [
        pl.Boolean,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.Float32,
        pl.Float64,
        pl.Date,
        pl.Datetime,
        pl.Utf8,
    ]
    safe_cast_map = {
        pl.Boolean: {pl.Boolean, pl.Utf8},
        pl.Int8: {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.Utf8},
        pl.Int16: {pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.Utf8},
        pl.Int32: {pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.Utf8},
        pl.Int64: {pl.Int64, pl.Float32, pl.Float64, pl.Utf8},
        pl.Float32: {pl.Float32, pl.Float64, pl.Utf8},
        pl.Float64: {pl.Float64, pl.Utf8},
        pl.Utf8: {pl.Utf8},
        pl.Date: {pl.Date, pl.Utf8},
        pl.Datetime: {pl.Datetime, pl.Utf8},
    }
    for candidate in dtype_precedence:
        if all(candidate in safe_cast_map.get(dt, {}) for dt in dtypes):
            return candidate
    return pl.Utf8
