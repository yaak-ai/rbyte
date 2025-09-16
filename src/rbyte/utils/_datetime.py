from datetime import UTC, datetime, tzinfo


def datetime_from_nanos(timestamp: int, tz: tzinfo = UTC) -> datetime:
    return datetime.fromtimestamp(timestamp=timestamp / 1e9, tz=tz)
