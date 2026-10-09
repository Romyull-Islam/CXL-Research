"""Rebuild long_decode_64_ids.jsonl from long_decode_64_noids.jsonl: request k gets unique token
ids starting at (k + 1) * 1,000,000, prompt tokens first, then output tokens, so LLMServingSim's
prefix tier sees blocks that are never shared between requests.
Run: python make_ids_trace.py  (in this folder)"""
import json

with open("long_decode_64_noids.jsonl") as src, open("long_decode_64_ids.jsonl", "w") as dst:
    for k, line in enumerate(src):
        r = json.loads(line)
        base = (k + 1) * 1_000_000
        r["input_tok_ids"] = list(range(base, base + r["input_toks"]))
        r["output_tok_ids"] = list(range(base + r["input_toks"], base + r["input_toks"] + r["output_toks"]))
        dst.write(json.dumps(r) + "\n")
