"""Explicit T1 target validation; simulator availability is checked at launch."""

from tvm.target import Target

from tilelang.backend.target import register_target_normalizer


def normalize_target(value):
    if isinstance(value, Target):
        if value.kind.name != "t1":
            return None
    elif isinstance(value, dict):
        if value.get("kind") != "t1":
            return None
    elif isinstance(value, str):
        if not value.strip() or value.split()[0] != "t1":
            return None
    else:
        return None
    target = Target(value)
    vlen = int(target.attrs["vlen"])
    if not 128 <= vlen <= 65536 or vlen & (vlen - 1):
        raise ValueError("T1 vlen must be a power of two between 128 and 65536 bits")
    return target


register_target_normalizer("t1", normalize_target)
