import hashlib
from pathlib import Path

import kagglehub
import polars as pl

from .transforms import (
    find_shared_subnet_mask,
    cast_to_valid_ip,
    ip_properties_exprs,
    get_port_feature_expressions,
)


def load_x_iiotd() -> pl.DataFrame:
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

    # (ii) create attack_label column and map columns to valid types
    df = df.with_columns(
        pl.col("class3")
        .replace_strict({"Normal": 0, "Attack": 1}, return_dtype=pl.Int64, default=None)
        .alias("attack_label"),
    )
    df = df.drop("class3", "class1")
    df = df.rename({"class2": "attack_type", "Timestamp": "timestamp"})
    df = df.with_columns(
        pl.col("Date").str.strip_chars().str.strptime(pl.Date, strict=False),
        pl.col("anomaly_alert").replace_strict(
            {"FALSE": 0, "TRUE": 1}, return_dtype=pl.Int64, default=None
        ),
        *[
            pl.col(column).map_elements(cast_to_valid_ip, return_dtype=pl.Utf8)
            for column in ["Scr_IP", "Des_IP"]
        ],
        *[
            pl.col(column).cast(pl.Int64, strict=False)
            for column in [
                "Scr_port",
                "Des_port",
                "timestamp",
                "missed_bytes",
                "Scr_bytes",
                "Des_bytes",
                "total_packet",
                "total_bytes",
                "Scr_pkts",
                "Scr_ip_bytes",
                "Des_pkts",
                "Des_ip_bytes",
            ]
        ],
        *[
            pl.col(column).cast(pl.Float64, strict=False)
            for column in [
                "Duration",
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
            ]
        ],
    )

    # (iii) session identification ... each row of this dataset is already a summary of a flow
    # so it's very likely to have a lot of sessions with only one entry
    df = df.with_columns(pl.lit("0").alias("session_id"))

    # (iv) feature engineering
    df = df.with_columns(
        pl.struct(["Des_IP", "Scr_IP"])
        .map_elements(
            lambda x: find_shared_subnet_mask(x["Des_IP"], x["Scr_IP"]), return_dtype=pl.Utf8
        )
        .alias("IP.Subnet"),
        *ip_properties_exprs("Des_IP", "Scr_IP"),
        *get_port_feature_expressions("Scr_port"),
        *get_port_feature_expressions("Des_port"),
    )

    # (v) drop others
    df = df.drop("Des_IP", "Scr_IP", "Date", "Scr_port", "Des_port")

    return df


def _assign_flow_session_id(df: pl.DataFrame) -> pl.DataFrame:
    """
    Assigns a unique session_id to each row in a Polars DataFrame,
    assuming each row already represents a distinct network flow/session.

    If you need to group packets into flows first (like in previous examples),
    that logic would need to precede this function or be incorporated.
    This function is suitable for datasets where each row is ALREADY a flow summary.

    The session_id is created by hashing a combination of key fields that
    uniquely identify a flow, plus a timestamp to differentiate potentially
    reused 5-tuples over time if not already disambiguated by other means
    in the source data.

    Args:
        df (pl.DataFrame): Input DataFrame where each row is a flow.
                           Must contain columns like 'Scr_IP', 'Scr_port',
                           'Des_IP', 'Des_port', 'Protocol', and 'Timestamp'.

    Returns:
        pl.DataFrame: DataFrame with an added 'session_id' column.
    """
    # Define the columns that constitute a unique flow identifier
    # Timestamp is included to differentiate flows if the 5-tuple is reused.
    # If your 'Timestamp' is the *start* of the flow and rows are already unique flows,
    # then this combination should be highly unique.
    # If 'Duration' is available and more suitable to make it unique with Timestamp,
    # you could consider it. But for simple per-flow ID, Timestamp (start of flow) is good.

    # Ensure necessary columns exist
    required_cols = ["Scr_IP", "Scr_port", "Des_IP", "Des_port", "Protocol", "timestamp"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for session ID generation: {col}")

    # Convert all identifying columns to string type for consistent hashing
    # Fill nulls with a placeholder string to avoid errors in concatenation/hashing
    # and to ensure nulls contribute to a different hash.
    # Create a composite key string from the identifying columns
    separator = "||"
    composite_key_expr = None
    for col in required_cols:
        expr = pl.col(col).cast(pl.Utf8).fill_null("NONE")
        if composite_key_expr is None:
            composite_key_expr = expr
        else:
            composite_key_expr = composite_key_expr + separator + expr

    # Apply a hash function to the composite key to create a session_id
    # Using map_elements to apply the Python hash function
    assert composite_key_expr is not None
    df = df.with_columns(
        composite_key_expr.map_elements(
            lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest(), return_dtype=pl.Utf8
        ).alias("session_id")
    )

    return df
