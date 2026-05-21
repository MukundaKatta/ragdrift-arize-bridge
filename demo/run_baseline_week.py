"""Demo, baseline week — drift should NOT fire.

Generates a fake retail-RAG agent run with embeddings drawn from the
same distribution as the baseline. The bridge observes each prediction.
The drift scalars stay under threshold. Phoenix span attributes are
quiet.

Run::

    pip install -e ".[demo]"
    python demo/run_baseline_week.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragdrift_arize import DriftConfig, DriftSpanProcessor


def main() -> int:
    rng = np.random.default_rng(0)

    # Pretend baseline is 1000 user queries collected in January.
    baseline = rng.normal(size=(1000, 64)).astype(np.float32)

    bridge = DriftSpanProcessor(
        DriftConfig(
            baseline_embeddings=baseline,
            embedding_threshold=0.05,
            flush_every=64,
        )
    )

    # Pretend the May query distribution is the same as January (no drift).
    n_queries = 256
    for _ in range(n_queries):
        # Draw from the same N(0,1) baseline distribution.
        emb = rng.normal(size=(64,)).astype(np.float32)
        bridge.observe(emb)

    # Final force flush + print.
    bridge.force_flush()
    print("[baseline week] drift attributes:")
    for k, v in bridge._cache.to_attrs().items():
        print(f"  {k} = {v}")
    print("\nAll five scalars should be under their thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
