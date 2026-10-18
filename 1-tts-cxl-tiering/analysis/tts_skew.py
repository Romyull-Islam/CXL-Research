"""Is test-time-scaling KV memory concentrated in a few long samples? (MathArena AIME 2025)

Parallel sampling: the k samples of one problem start together and decode at the same rate, so
sample i holds KV that grows linearly with its tokens until it finishes. KV byte-seconds of a
sample is proportional to in*out + out^2/2 (tokens x decode steps). We report, per model config:
- length spread within a problem (max/median of output tokens),
- share of the group's KV byte-seconds held by its single longest sample,
- tail phase: share of the group's wall time during which only one sample is still running,
- how much peak-to-mean the group's KV footprint has.
"""
import numpy as np
import pandas as pd

df = pd.read_parquet("data/tts/matharena_aime25.parquet")
df["in_t"] = pd.to_numeric(df.input_tokens, errors="coerce")
df["out_t"] = pd.to_numeric(df.output_tokens, errors="coerce")
df = df.dropna(subset=["in_t", "out_t"])
df = df[df.out_t > 0]


def group_stats(g):
    out = np.sort(g.out_t.to_numpy())
    inp = g.in_t.to_numpy().mean()
    if len(out) < 2:
        return None
    bs = inp * out + out ** 2 / 2
    t = np.arange(0, out[-1] + 1, max(out[-1] // 2000, 1))
    alive_kv = sum(np.where(t <= o, inp + np.minimum(t, o), 0) for o in out)
    return dict(n=len(out), spread=out[-1] / max(np.median(out), 1), top_share=bs[-1] / bs.sum(),
                tail_phase=(out[-1] - out[-2]) / out[-1], peak_to_mean=alive_kv.max() / alive_kv.mean())


rows = []
for (cfg, prob), g in df.groupby(["model_config", "problem_idx"]):
    s = group_stats(g)
    if s:
        rows.append(dict(cfg=cfg, **s))
st = pd.DataFrame(rows)

lengths = df.groupby("model_config").out_t.describe(percentiles=[.5, .9, .99])[["count", "50%", "90%", "max"]]
summary = st.groupby("cfg").agg(groups=("n", "size"), k=("n", "median"), spread=("spread", "median"),
                                top_share=("top_share", "median"), tail_phase=("tail_phase", "median"),
                                peak_to_mean=("peak_to_mean", "median"))
out = lengths.join(summary).sort_values("50%", ascending=False)
pd.set_option("display.width", 200)
print(out.round(2).head(30).to_string())
print(f"\nAll configs: median spread {st.spread.median():.2f}, longest sample's share of byte-seconds "
      f"{st.top_share.median():.2f} (uniform would be {1 / st.n.median():.2f}), "
      f"tail phase {st.tail_phase.median():.2f}")
