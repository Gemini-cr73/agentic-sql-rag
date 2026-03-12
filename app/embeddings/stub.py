from __future__ import annotations

import hashlib

EMBED_DIM = 384


def embed_text(text: str, dim: int = EMBED_DIM) -> list[float]:
    """
    Deterministic embedding stub (no numpy).

    - Same input text -> same vector
    - Produces `dim` floats
    - Expands entropy by repeatedly hashing
    - Output values in range [-1.0, 1.0]
    """

    if not isinstance(text, str):
        raise TypeError(f"embed_text expects str, got {type(text)}")

    if dim <= 0:
        raise ValueError("Embedding dimension must be positive")

    # Initial hash
    h = hashlib.sha256(text.encode("utf-8")).digest()

    out: list[float] = []

    # Keep hashing the previous hash to expand entropy
    while len(out) < dim:
        for b in h:
            # Map byte (0–255) -> [-1, 1]
            value = (b / 255.0) * 2.0 - 1.0
            out.append(value)

            if len(out) >= dim:
                break

        # Re-hash the previous digest for new entropy
        h = hashlib.sha256(h).digest()

    return out
