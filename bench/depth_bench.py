#!/usr/bin/env python3
"""Measure MoE offload performance at a given context depth.

Fills the KV cache to a target depth, then reports three numbers:

  * bulk prefill   - tokens/s while evaluating the cold fill (what a long prompt costs)
  * short eval     - tokens/s for a small prompt evaluated on top of the filled cache
  * decode         - tokens/s while generating at that depth

These are the three figures used in the project README. Uses only the standard
library; talks to llama-server's /completion endpoint.

Usage:
    python3 bench/depth_bench.py --url http://127.0.0.1:8080 --depth 85000 --gen 256

Note: "depth" is approximate. The filler is expanded with the model's own
tokenizer behaviour, so the reported depth is whatever the server actually
evaluated; the script prints it.
"""

import argparse
import json
import sys
import time
import urllib.request

FILLER = "The quick brown fox jumps over the lazy dog. "


def post(url, payload, timeout):
    req = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def measure(url, depth, gen, timeout):
    # ~1.3 tokens per word for English prose; adjust the multiplier until the
    # server reports roughly the requested depth.
    words = max(1, int(depth / 1.3))
    filler = (FILLER * (words // len(FILLER.split()) + 1)).strip()

    t0 = time.time()
    fill = post(url, {
        "prompt": filler,
        "n_predict": 1,
        "temperature": 0.0,
        "cache_prompt": True,
    }, timeout)
    fill_wall = time.time() - t0

    ft = fill.get("timings", {})
    bulk = ft.get("prompt_per_second", 0.0)
    filled = ft.get("prompt_n", 0)

    # Same prefix again plus a ~64-token suffix: the prefix comes from cache, so
    # the prompt-eval figure is the "short eval at depth" number. The suffix is
    # deliberately long enough that fixed per-request overhead does not dominate.
    suffix = "\n\n" + ("Please consider the text above carefully. " * 8) + "\nReply with the single word OK."
    probe = post(url, {
        "prompt": filler + suffix,
        "n_predict": gen,
        "temperature": 0.0,
        "cache_prompt": True,
    }, timeout)

    pt = probe.get("timings", {})
    return {
        "filled": filled,
        "bulk_prefill": bulk,
        "fill_wall": fill_wall,
        "short_eval": pt.get("prompt_per_second", 0.0),
        "short_n": pt.get("prompt_n", 0),
        "decode": pt.get("predicted_per_second", 0.0),
        "gen_n": pt.get("predicted_n", 0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--depth", type=int, default=85000,
                    help="target context depth in tokens (approximate)")
    ap.add_argument("--gen", type=int, default=256, help="tokens to generate")
    ap.add_argument("--runs", type=int, default=1, help="repeat the whole measurement")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    for run in range(args.runs):
        try:
            r = measure(args.url, args.depth, args.gen, args.timeout)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        ms = 1000.0 / r["decode"] if r["decode"] else 0.0
        print(f"depth={r['filled']:>6}  "
              f"bulk prefill {r['bulk_prefill']:6.1f} t/s   "
              f"short eval {r['short_eval']:6.1f} t/s   "
              f"decode {r['decode']:5.1f} t/s  ({ms:.1f} ms/tok)"
              + (f"   [run {run + 1}/{args.runs}]" if args.runs > 1 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
