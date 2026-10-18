# Reserved KV pools are mostly empty (Alibaba xMaaS, two applications)

Data: Alibaba cluster-trace-v2026-maas, app_metrics.csv (hourly, one week, days 140 to 146).
Paper: "Model-as-a-Service in the Wild: Characterizing Production LLM Inference Clusters at
Scale", ATC'26. The trace covers 100K+ GPUs, but per-application KV metrics exist for only two
applications.

| Application / role | gpu_mem_util | kv_cache_util mean | p10 | p50 | p90 | max |
|---|---|---|---|---|---|---|
| app 42c7 (PD collocated) | 99.2% | 1.5% | 0.1% | 1.3% | 3.7% | 8.2% |
| app 5a3a, prefill | n/a | 2.2% | 1.0% | 1.8% | 4.3% | 6.5% |
| app 5a3a, decode | n/a | 45.6% | 30.2% | 46.1% | 60.1% | 70.2% |

Reading:
- GPU memory looks 99% used because the engine reserves its KV pool up front; the pool itself is
  1 to 2% used outside decode instances, and under 70% even on decode instances.
- The ATC'26 paper already reports GPU memory overprovisioning and cold-model residency, so this
  is context. Two applications are too few to generalize.
- The weekly five-minute instance table (4.5 GiB) has only gpu_mem_util and host mem_usage, which
  reflect reservation, not demand; it cannot show KV demand per instance.
