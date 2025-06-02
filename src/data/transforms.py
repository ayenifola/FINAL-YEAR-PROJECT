import socket
import struct
import ipaddress

import polars as pl


def ip_to_int(ip):
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except (OSError, socket.error):
        return None


def int_to_ip(i):
    return socket.inet_ntoa(struct.pack("!I", i))


def find_shared_subnet_mask(
    ip1: str | None, ip2: str | None, default_value: str | None = None
) -> str | None:
    if ip1 is None or ip2 is None:
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


def ip_properties_exprs(*columns: str) -> list[pl.Expr]:
    expressions = []
    for column in columns:
        expressions.extend(
            [
                pl.col(column)
                .map_elements(is_ip_private, return_dtype=pl.Boolean)
                .alias(f"{column}.is_private")
                .fill_null(False),
                pl.col(column)
                .map_elements(is_ip_loopback, return_dtype=pl.Boolean)
                .alias(f"{column}.is_loopback")
                .fill_null(False),
                pl.col(column)
                .map_elements(is_ip_multicast, return_dtype=pl.Boolean)
                .alias(f"{column}.is_multicast")
                .fill_null(False),
                pl.col(column)
                .map_elements(is_ip_link_local, return_dtype=pl.Boolean)
                .alias(f"{column}.is_link_local")
                .fill_null(False),
                pl.col(column)
                .map_elements(is_ip_unspecified, return_dtype=pl.Boolean)
                .alias(f"{column}.is_unspecified")
                .fill_null(False),
            ]
        )
    return expressions


def get_port_feature_expressions(
    port_column_name: str, cleaned_port_col_alias: str | None = None
) -> list[pl.Expr]:
    """
    Generates a list of Polars expressions for engineering features
    from a network port column.

    Args:
        port_column_name (str): The name of the original port column in the DataFrame
                                (e.g., "Scr_port", "Des_port").
        cleaned_port_col_alias (str | None, optional):
                                The alias for the cleaned (integer cast, null filled)
                                port column. If None, it defaults to the
                                original port_column_name. This is useful if you
                                want to create the cleaned column once and then
                                reference it in subsequent feature expressions.

    Returns:
        list[pl.Expr]: A list of Polars expressions. The first expression in the list
                       is for cleaning the port column. The subsequent expressions
                       generate new features based on this cleaned port column.
    """
    if not isinstance(port_column_name, str) or not port_column_name:
        raise ValueError("port_column_name must be a non-empty string.")

    # If no specific alias is provided for the cleaned column, use the original name.
    # This means the original column will be overwritten by its cleaned version.
    effective_port_col = cleaned_port_col_alias if cleaned_port_col_alias else port_column_name

    expressions = []

    # # Expression 0: Clean the port column (cast to Int64, fill nulls with -1)
    # # This expression creates/overwrites the 'effective_port_col'
    # cleaning_expr = (
    #     pl.col(port_column_name)
    #     .cast(pl.Int64, strict=False)
    #     .fill_null(-1)
    #     .alias(effective_port_col) # Alias to 'effective_port_col'
    # )
    # expressions.append(cleaning_expr)

    # Feature expressions will now refer to 'effective_port_col'

    # Expression 1: Port Category
    category_expr = (
        pl.when((pl.col(effective_port_col) >= 0) & (pl.col(effective_port_col) <= 1023))
        .then(pl.lit("well-known"))
        .when((pl.col(effective_port_col) >= 1024) & (pl.col(effective_port_col) <= 49151))
        .then(pl.lit("registered"))
        .when((pl.col(effective_port_col) >= 49152) & (pl.col(effective_port_col) <= 65535))
        .then(pl.lit("dynamic-private"))
        .otherwise(pl.lit("invalid"))
        .alias(f"{port_column_name}.category")  # Output column name based on original
    )
    expressions.append(category_expr)

    # Expression 2: Is Well-Known Port (Boolean)
    is_well_known_expr = (
        (pl.col(effective_port_col) >= 0) & (pl.col(effective_port_col) <= 1023)
    ).alias(f"{port_column_name}.is_well_known")
    expressions.append(is_well_known_expr)

    # Expression 3: Is Common Service Port (Boolean for a few examples)
    common_service_ports = [
        20,
        21,  # FTP
        22,  # SSH
        23,  # Telnet
        25,  # SMTP
        53,  # DNS
        80,  # HTTP
        110,  # POP3
        123,  # NTP
        143,  # IMAP
        161,
        162,  # SNMP
        443,  # HTTPS
        445,  # SMB
        514,  # Syslog
        1883,  # MQTT
        8883,  # MQTT over TLS
        5683,  # CoAP
        5684,  # CoAP over DTLS
    ]
    is_common_service_expr = (
        pl.col(effective_port_col)
        .is_in(common_service_ports)
        .alias(f"{port_column_name}.is_common_service")
    )
    expressions.append(is_common_service_expr)

    return expressions


def is_ip_private(ip_str: str) -> bool | None:
    """Checks if an IPv4 or IPv6 address is private."""
    if not ip_str or ip_str == "0" or ip_str == "0.0.0.0":  # Handle common placeholders
        return None  # Or False, depending on how you want to treat these
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        return ip_obj.is_private
    except ValueError:
        # Invalid IP string
        return None  # Or False


def is_ip_loopback(ip_str: str) -> bool | None:
    """Checks if an IPv4 or IPv6 address is a loopback address."""
    if not ip_str or ip_str == "0" or ip_str == "0.0.0.0":
        return None
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        return ip_obj.is_loopback
    except ValueError:
        return None


def is_ip_multicast(ip_str: str) -> bool | None:
    """Checks if an IPv4 or IPv6 address is a multicast address."""
    if not ip_str or ip_str == "0" or ip_str == "0.0.0.0":
        return None
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        return ip_obj.is_multicast
    except ValueError:
        return None


def is_ip_link_local(ip_str: str) -> bool | None:
    """Checks if an IPv4 or IPv6 address is link-local."""
    if not ip_str or ip_str == "0" or ip_str == "0.0.0.0":
        return None
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        return ip_obj.is_link_local
    except ValueError:
        return None


def is_ip_unspecified(ip_str: str) -> bool | None:
    """Checks if an IPv4 or IPv6 address is unspecified (e.g., '0.0.0.0' or '::')."""
    if (
        not ip_str
    ):  # Only check empty string here, as '0' or '0.0.0.0' are handled by is_unspecified
        return None
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        return ip_obj.is_unspecified  # This correctly handles "0.0.0.0" and "::"
    except ValueError:
        return None


def cast_to_valid_ip(ip: str | None) -> str | None:
    try:
        ipaddress.IPv4Address(ip)
        return ip
    except (ipaddress.AddressValueError, ipaddress.NetmaskValueError):
        return None
