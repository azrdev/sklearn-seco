"""
Miscellaneous things not depending on anything else from sklearn_seco.
"""

import math


def log2(x: float) -> float:
    """`log2(x) if x > 0 else 0`"""
    return math.log2(x) if x > 0 else 0
