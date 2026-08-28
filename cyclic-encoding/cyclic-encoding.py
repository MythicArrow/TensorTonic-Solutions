import math

def cyclic_encoding(values: list, period: float) -> list:
    """
    Returns the sine and cosine encoding of every cyclic value.
    """
    # Write code here
    encoded = []
    for v in values:
        angle = (2 * math.pi * v) / period
        sin = math.sin(angle)
        cos = math.cos(angle)
        encoded.append([sin,cos])
    return encoded