"""Demo, drifted week — drift SHOULD fire.

Same agent, same model, same RAG index. But the *query distribution*
shifted: user intents moved from sweaters to summer items, embeddings
are pulled +1.5σ. ragdrift catches it on the embedding dimension.

Run::

    pip install -e ".[demo]"
    python demo/run_drifted_week.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragdrift_arize import DriftConfig, DriftSpanProcessor


def main() -> int:
    rng = np.random.default_rng(0)

    baseline = rng.normal(size=(1000, 64)).astype(np.float32)

    bridge = DriftSpanProcessor(
        DriftConfig(
            baseline_embeddings=baseline,
            embedding_threshold=0.05,
            flush_every=64,
        )
    )

    # May queries shift mean to +1.5σ — summer items the index has poor coverage for.
    n_queries = 256
    for _ in range(n_queries):
        emb = rng.normal(loc=1.5, size=(64,)).astype(np.float32)
        bridge.observe(emb)

    bridge.force_flush()
    print("[drifted week] drift attributes:")
    for k, v in bridge._cache.to_attrs().items():
        print(f"  {k} = {v}")

    if bridge._cache.embedding_exceeded:
        print("\n🚨 drift.embedding > threshold — Arize alert would fire.")
    else:
        print("\n(no drift fired — this would be surprising; check baseline.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
