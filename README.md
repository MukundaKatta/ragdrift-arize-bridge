# ragdrift × Arize bridge

[![PyPI](https://img.shields.io/pypi/v/ragdrift-arize-bridge.svg)](https://pypi.org/project/ragdrift-arize-bridge/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Devpost](https://img.shields.io/badge/devpost-Google%20Cloud%20Rapid%20Agent-D4A853.svg)](https://rapid-agent.devpost.com/)

**The Rust-fast scalar Arize Phoenix lacks for production alerting.**

Phoenix is great for embedding visualization. What it doesn't give you is
a single number you can alert on at 3 AM. `ragdrift` is that number ​—​ five
of them, actually:

| Dimension     | Method                          |
|---------------|---------------------------------|
| Embedding     | MMD² (RBF) + sliced Wasserstein |
| Data          | per-feature KS + PSI            |
| Response      | KS on lengths                   |
| Confidence    | KS + optional ECE delta         |
| Query         | k-means + symmetric KL          |

Rust core, PyO3 binding, full report in ~180 ms on 10k samples.

This bridge plugs ragdrift into the OpenTelemetry span pipeline Phoenix
is already collecting. Every span gains five `drift.*` attributes you
can alert on in Arize.

## Install

```bash
pip install ragdrift-arize-bridge
# add the [demo] extras for the runnable retail-drift demo
pip install 'ragdrift-arize-bridge[demo]'
```

Pulls in `ragdrift-py` (the Rust core via PyO3) and the standard
`opentelemetry-api` / `opentelemetry-sdk` packages.

## Wire it in

```python
from opentelemetry.sdk.trace import TracerProvider
from ragdrift_arize import DriftConfig, DriftSpanProcessor
import numpy as np

baseline = np.load("baseline_embeddings.npy")  # (n, dim) float32

bridge = DriftSpanProcessor(
    DriftConfig(
        baseline_embeddings=baseline,
        embedding_threshold=0.05,
        flush_every=64,
    )
)

provider = TracerProvider()
provider.add_span_processor(bridge)

# In your agent runtime, after every prediction:
bridge.observe(prediction_embedding,
               response_length=token_count,
               confidence=score)
```

Every outgoing span now carries:

```
drift.embedding = 1.0324
drift.embedding.exceeded = true
drift.data = 0.012
drift.response = 0.34
drift.confidence = 0.08
drift.query = 0.07
```

Configure Arize to route on `drift.embedding.exceeded == true` and you're
done.

## The retail demo

The Rapid Agent retail challenge: a brick-and-mortar agent that
recommends items via RAG over the store's catalog. It gets *worse* as
the season turns because the embedding distribution of incoming queries
drifts week over week.

```bash
git clone https://github.com/MukundaKatta/ragdrift-arize-bridge
cd ragdrift-arize-bridge
pip install -e '.[demo]'

# Baseline week (January): all clear.
python demo/run_baseline_week.py
# Output: drift.embedding = 0.04, drift.embedding.exceeded = false

# Drifted week (May): query distribution shifted +1.5σ.
python demo/run_drifted_week.py
# Output: drift.embedding = 1.03, drift.embedding.exceeded = true
# → Arize alert fires
```

The agent doesn't know. The model doesn't know. The Phoenix dashboard
sees it 200 ms after the batch lands.

## Architecture

```
+--------------------+   +-------------------+   +-------------------+
|  retail RAG agent  |-->|  ragdrift bridge  |-->|  Arize Phoenix    |
|  (Gemini / Vertex) |   |  (Rust core+PyO3) |   |  spans + alerts   |
+--------------------+   +-------------------+   +-------------------+
                                 |
                                 v
                         five span attributes:
                         drift.{embedding,data,response,confidence,query}
```

The bridge is a SpanProcessor — same hook Phoenix already uses for its
own exporters. We compute the drift report every `flush_every`
predictions, cache the result, and stamp it onto every outgoing span
in `on_start`. So Arize sees fresh drift attributes on every span
without paying the MMD cost on every call.

## What it doesn't do

- **Doesn't replace Phoenix.** Phoenix is the visualization + retention
  layer. ragdrift is the scalar. Compose, don't compete.
- **Doesn't fit the agent.** The agent is whatever you already have.
  This bridge is the drift telemetry around it.
- **Doesn't store baselines.** Load from disk, S3, GCS, or recompute on
  a sliding window — your call. We provide the API, not the policy.

## Performance

The math runs in Rust. Indicative numbers from `ragdrift`'s own benches:

| Operation                              | Pure numpy | ragdrift-core |
|----------------------------------------|------------|---------------|
| MMD (RBF), 10k × 768 vs 10k × 768      | ~8.0 s     | ~120 ms       |
| Sliced Wasserstein, 50 projections     | ~2.4 s     | ~35 ms        |
| Full report (all 5 dims, 10k samples)  | ~12 s      | ~180 ms       |

Lambda-safe at production query rates.

## Related

- `ragdrift-py` (the Rust core via PyO3) — https://pypi.org/project/ragdrift-py/
- `ragdrift-core` (just the Rust crate) — https://crates.io/crates/ragdrift-core
- Parent project, with the math + tests + docs — https://github.com/MukundaKatta/ragdrift

## License

MIT.
