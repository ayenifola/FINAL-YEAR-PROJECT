import kagglehub
import polars as pl
from pathlib import Path

from .transforms import (
    find_shared_subnet_mask,
    cast_to_valid_ip,
    ip_properties_exprs,
)


def load_iiotset() -> pl.DataFrame:
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

    # (i) rename columns
    # these fields feel swapped
    df = df.rename({"dns.qry.name.len": "dns.qry.name", "dns.qry.name": "dns.qry.name.len"})
    df = df.rename({"Attack_label": "attack_label", "Attack_type": "attack_type"})

    # (ii) remove superfluous columns
    superfluous_columns = [
        "mqtt.msg_decoded_as",
        "dns.qry.type",
        "http.tls_port",
        "icmp.unused",
    ]
    df = df.select(pl.exclude(superfluous_columns))

    # (iii) cast fields to the right type
    ip_columns = [
        "ip.src_host",
        "ip.dst_host",
        "arp.src.proto_ipv4",
        "arp.dst.proto_ipv4",
    ]
    df = df.with_columns(
        # frame.time: convert to timestamp (seconds)
        (
            pl.col("frame.time")
            .str.strip_chars()
            .str.strptime(pl.Datetime, "%Y %H:%M:%S.%9f", strict=False)
            .dt.timestamp("us")
            .cast(pl.Float64)
            * 1e-6
        ),
        *_cast_numeric_like_cols_exprs(),
        pl.col("dns.qry.name.len").cast(pl.Int64),
        *[
            pl.col(column).map_elements(
                cast_to_valid_ip,
                return_dtype=pl.Utf8,
            )
            for column in ip_columns
        ],
    )

    # (iv) session identification ... this would be used later for grouping and chunking
    df = (
        df.with_columns(_protocol_type_expr(), *_sorted_ips_exprs())
        .with_columns(*_sorted_ports_exprs())
        .with_columns(_session_key_base_expr())
    )
    df = _assign_session_id(df, timeout_seconds=60)
    df = df.drop(
        "session_key_base",
        "protocol_type_inferred",
        "ip1",
        "ip2",
        "port1_tcp",
        "port2_tcp",
    )

    # (v) feature engineering on some columns
    df = df.with_columns(
        pl.col("frame.time").fill_null(0),
        pl.col("frame.time").is_not_null().alias("frame.time.is_valid"),
        # use a common subnet mask
        pl.struct(["ip.src_host", "ip.dst_host"])
        .map_elements(
            lambda x: find_shared_subnet_mask(x["ip.src_host"], x["ip.dst_host"]),
            return_dtype=pl.Utf8,
        )
        .alias("ip.subnet"),
        *ip_properties_exprs(*ip_columns),
        # use length to summarize data
        *[
            pl.col(column).str.strip_chars().str.len_chars().alias(f"{column}.length")
            for column in [
                "tcp.payload",
                "tcp.options",
                "http.request.full_uri",
                "http.request.uri.query",
                "http.file_data",
                "mqtt.msg",
            ]
        ],
    )
    df = df.drop(
        "ip.src_host",
        "ip.dst_host",
        "tcp.payload",
        "tcp.options",
        "http.request.full_uri",
        "http.request.uri.query",
        "http.file_data",
        "mqtt.msg",
    )
    df = df.rename({"frame.time": "timestamp"})

    # (vi) discard some columns
    df = df.select(
        pl.exclude(
            [
                # checksums
                "tcp.checksum",
                "icmp.checksum",
                # processed or used in some capacity
                *ip_columns,
                # almost superfluous
                "dns.retransmit_request_in",
                "dns.retransmit_request",
                # columns that EDA doesn't suggest being useful
                "tcp.srcport",
                "tcp.dstport",
                # "arp.opcode",
                # "tcp.flags",
                # "http.content_length",
                # "icmp.transmit_timestamp",
                # "mbtcp.trans_id",
                # "udp.port",
                # "udp.time_delta",
                # "tcp.len",
                # "icmp.seq_le",
                # "tcp.ack_raw",
                # "tcp.connection.synack",
                # "mqtt.conflag.cleansess",
                # "tcp.flags.ack",
                # "tcp.connection.syn",
                # "tcp.connection.rst",
                # "tcp.connection.fin",
                # "arp.hw.size",
            ]
        )
    )

    return df


def _cast_numeric_like_cols_exprs() -> list[pl.Expr]:
    numeric_like_cols = [
        "tcp.srcport",
        "tcp.dstport",
        "udp.port",
        "udp.stream",
        "icmp.seq_le",
        "icmp.checksum",
        "arp.opcode",
        "mqtt.msgtype",
        "mbtcp.unit_id",
        "tcp.ack",
        "tcp.ack_raw",
        "tcp.checksum",
        "tcp.connection.fin",
        "tcp.connection.rst",
        "tcp.connection.syn",
        "tcp.connection.synack",
        "tcp.flags.ack",
        "tcp.len",
        "tcp.seq",
        "mqtt.len",
    ]
    return [
        pl.col(col_name).cast(pl.Float64, strict=False).fill_null(0.0).cast(pl.Int64)
        for col_name in numeric_like_cols
    ]


def _protocol_type_expr() -> pl.Expr:
    # Create expressions for checking protocol conditions
    is_tcp = (pl.col("tcp.srcport").is_not_null()) & (pl.col("tcp.dstport").is_not_null())
    is_udp_stream = pl.col("udp.stream").is_not_null()
    is_udp_port = pl.col("udp.port").is_not_null()
    is_icmp = pl.col("icmp.checksum").is_not_null()  # Or other ICMP specific fields
    is_arp = pl.col("arp.opcode").is_not_null()
    is_mqtt_other = (
        (pl.col("mqtt.msgtype").is_not_null())
        & (pl.col("tcp.srcport").is_null())
        & (pl.col("udp.port").is_null())
        & (pl.col("udp.stream").is_null())
    )
    is_mbtcp_other = (pl.col("mbtcp.trans_id").is_not_null()) & (pl.col("tcp.srcport").is_null())

    return (
        pl.when(is_tcp)
        .then(pl.lit("TCP"))
        .when(is_udp_stream)
        .then(pl.lit("UDP"))
        .when(is_udp_port)
        .then(pl.lit("UDP"))
        .when(is_icmp)
        .then(pl.lit("ICMP"))
        .when(is_arp)
        .then(pl.lit("ARP"))
        .when(is_mqtt_other)
        .then(pl.lit("MQTT_OTHER_TRANSPORT"))
        .when(is_mbtcp_other)
        .then(pl.lit("MBTCP_OTHER_TRANSPORT"))
        .otherwise(pl.lit("Unknown"))
        .alias("protocol_type_inferred")
    )


def _sorted_ips_exprs() -> list[pl.Expr]:
    ip_src_str = pl.col("ip.src_host")
    ip_dst_str = pl.col("ip.dst_host")
    return [
        pl.when(ip_src_str <= ip_dst_str).then(ip_src_str).otherwise(ip_dst_str).alias("ip1"),
        pl.when(ip_src_str <= ip_dst_str).then(ip_dst_str).otherwise(ip_src_str).alias("ip2"),
    ]


def _sorted_ports_exprs() -> list[pl.Expr]:
    ip_src_str = pl.col("ip.src_host")
    ip_dst_str = pl.col("ip.dst_host")
    return [
        pl.when((pl.col("protocol_type_inferred") == "TCP") & (ip_src_str == pl.col("ip1")))
        .then(pl.col("tcp.srcport"))
        .when((pl.col("protocol_type_inferred") == "TCP") & (ip_src_str != pl.col("ip1")))
        .then(pl.col("tcp.dstport"))
        .otherwise(None)
        .alias("port1_tcp"),  # Null for non-TCP or if ips were null
        pl.when((pl.col("protocol_type_inferred") == "TCP") & (ip_src_str == pl.col("ip1")))
        .then(pl.col("tcp.dstport"))
        .when((pl.col("protocol_type_inferred") == "TCP") & (ip_src_str != pl.col("ip1")))
        .then(pl.col("tcp.srcport"))
        .otherwise(None)
        .alias("port2_tcp"),
    ]


def _session_key_base_expr() -> pl.Expr:
    # This will create a tuple (or struct in Polars) as the key
    # It uses pl.struct to combine columns into a single key column.

    # Define expressions for each part of the key based on protocol
    key_expr = (
        pl.when(pl.col("protocol_type_inferred") == "TCP")
        .then(pl.struct(["protocol_type_inferred", "ip1", "port1_tcp", "ip2", "port2_tcp"]))
        .when((pl.col("protocol_type_inferred") == "UDP") & pl.col("udp.stream").is_not_null())
        .then(
            pl.struct(
                [
                    "protocol_type_inferred",
                    "ip1",
                    "ip2",
                    pl.col("udp.stream").alias("udp_stream_id"),
                ]
            )
        )
        .when((pl.col("protocol_type_inferred") == "UDP") & pl.col("udp.port").is_not_null())
        .then(
            pl.struct(
                [
                    "protocol_type_inferred",
                    "ip1",
                    "ip2",
                    pl.col("udp.port").alias("udp_port_val"),
                ]
            )
        )
        .when(pl.col("protocol_type_inferred") == "ICMP")
        .then(
            pl.struct(
                [
                    "protocol_type_inferred",
                    "ip1",
                    "ip2",
                    pl.col("icmp.seq_le").fill_null(-1).alias("icmp_id"),
                ]
            )
        )
        .when(pl.col("protocol_type_inferred") == "ARP")
        .then(pl.struct([pl.lit("ARP_PKT").alias("protocol_type_inferred"), "ip1", "ip2"]))
        .otherwise(
            pl.struct(
                [
                    pl.col("protocol_type_inferred"),
                    pl.col("ip.src_host").alias("ip1"),
                    pl.col("ip.dst_host").alias("ip2"),
                ]
            )
        )
        .alias("session_key_base")
    )
    return key_expr


def _assign_session_id(df: pl.DataFrame, timeout_seconds: float = 60) -> pl.DataFrame:
    df = df.sort(["session_key_base", "frame.time"])

    session_instance_ids_series_list = []
    for _, df_group in df.group_by("session_key_base", maintain_order=True):
        session_instance_counter = 0
        group_name_str = str(df_group["session_key_base"][0])  # Get the group key value as string
        last_time = None
        instance_ids = []

        is_tcp_group = df_group["protocol_type_inferred"][0] == "TCP"

        # iter_rows is not the most performant but needed for this logic easily
        for row_tuple in df_group.iter_rows(named=True):
            current_time = row_tuple["frame.time"]
            current_time = current_time if current_time is not None else None
            new_session_for_tcp = False

            if is_tcp_group:
                syn = row_tuple.get("tcp.connection.syn", 0.0) == 1.0  # Default to 0.0 if None
                ack = row_tuple.get("tcp.flags.ack", 0.0) == 1.0
                fin = row_tuple.get("tcp.connection.fin", 0.0) == 1.0
                rst = row_tuple.get("tcp.connection.rst", 0.0) == 1.0

                if syn and not ack:  # Pure SYN
                    if last_time is not None:  # Not the first packet of the base session
                        session_instance_counter += 1
                    new_session_for_tcp = True

            if (
                not new_session_for_tcp
                and last_time is not None
                and (current_time - last_time > timeout_seconds)
            ):
                session_instance_counter += 1

            instance_ids.append(f"{group_name_str}_{session_instance_counter}")
            last_time = current_time

            if is_tcp_group and (fin or rst):
                # Reset to force new instance for next packet in this 5-tuple (if any)
                last_time = None

        session_instance_ids_series_list.append(pl.Series("session_id", instance_ids))

    return df.with_columns(pl.concat(session_instance_ids_series_list))
