# llama.cpp + TurboQuant KV + MoE expert prefetch

A focused fork of **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)**
(itself a fork of **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)**) that makes
CPU-offloaded MoE models fast enough to be usable on small-VRAM laptops, without giving up
long-context decode speed.

The headline: a **35B-parameter MoE (3B active, 2.68 bpw) at 85K context on a 6 GB laptop GPU
and 14 GiB of RAM**, decoding at **36 tokens/s** and prefilling at **783 tokens/s**.

```bash
# the whole configuration, for the impatient
TURBO_AUTO_ASYMMETRIC=0 GGML_SCHED_PREFETCH_EXPERTS=1 ./build/bin/llama-server \
  -m model.gguf -c 85000 -fit off -ngl 99 -lm mlock \
  -ot 'blk\.([0-9]|1[0-9]|20|3[4-9])\.ffn_(up|down|gate)_exps\.weight=CPU' \
  -ctk turbo5 -ctv turbo5 -fa on -b 2048 -ub 768
```

---

## Contents

- [Tested models](#tested-models)
- [Results](#results)
- [Test hardware](#test-hardware)
- [Why this fork exists](#why-this-fork-exists)
- [What changed](#what-changed)
- [Build](#build)
- [Run](#run)
- [Benchmark it yourself](#benchmark-it-yourself)
- [Tuning for your own machine](#tuning-for-your-own-machine)
- [Mechanics: why this works](#mechanics-why-this-works)
- [Limitations](#limitations)
- [Syncing with upstream](#syncing-with-upstream)
- [Credits](#credits)
- [License](#license)

---

## Tested models

Both are 2-bit-class quants of the same base model, chosen so the weights fit alongside an 85K
KV cache on a 6 GB card — the point was a usable agentic context window, not a chat demo.

| model | file | bpw | size | result |
|---|---|---|---|---|
| <a href="https://huggingface.co/pjsmith/Qwen3.6-35B-A3B-2.6763bpw.gguf"><img alt="pjsmith/Qwen3.6-35B-A3B-2.6763bpw.gguf" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-pjsmith%2FQwen3.6--35B--A3B--2.6763bpw-FFD21E?style=for-the-badge"></a> | `Qwen3.6-35B-A3B-12Gb-2.6763bpw.gguf` | 2.6763 | 11.6 GB | 783 t/s prefill / 36.2 t/s decode @85K |
| <a href="https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"><img alt="HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-HauhauCS%20Aggressive%20IQ2__M-FFD21E?style=for-the-badge"></a> | `...-Aggressive-IQ2_M.gguf` | 2.69 | 11.7 GB | same speeds at the same depths |

The two models measure identically here to within run-to-run noise. Everything below applies to
both; the numbers quoted are from the pjsmith quant because it was measured first and most often.

## Results

Model: **Qwen3.6-35B-A3B**, 2.6763 bpw GGUF (11.6 GB), architecture `qwen35moe`
— 40 layers, 256 experts, top-8 routing, ~3B parameters active per token. The HauhauCS
`IQ2_M` quant measures the same; see [Tested models](#tested-models).
Context: 85,000 tokens. Generation: 256 tokens. Same corpus and harness for every row.

| build | bulk prefill @85K | decode @85K | note |
|---|---|---|---|
| turboquant fork, turbo5 KV, 14 GPU expert layers | 421 t/s | 36.8 t/s | starting point |
| + pinned host experts, + CUDA-graph fixes (14 layers) | 591 t/s | 36.7 t/s | previous best config |
| **+ 3-slot expert prefetch (13 layers, `-ub 768`)** | **783 t/s** | **36.2 t/s** | this repo |

Depth sweep on the final configuration:

| context | bulk prefill | decode | short eval at depth |
|---|---|---|---|
| 25,000 | 908 t/s | 39.8 t/s | 127 t/s |
| 85,000 | 783 t/s | 36.2 t/s | 104 t/s |

Quality: **perplexity 6.3191** on wikitext-2 (8 × 4096), identical with the prefetch enabled
and disabled, and identical to the build without any of these changes. Peak VRAM 5,759 MiB of
6,141 MiB.

Measurement notes, because they matter:

- Run-to-run variability on this machine is ~0.2 % for bulk prefill and ~1.7 % for decode;
  cross-session decode drift can reach ~10 % with ambient temperature.
- "Bulk prefill" is tokens actually evaluated during a cold fill (`fill_prompt_n / fill_s`),
  not the 64-token prompt-eval figure that llama.cpp prints for a cached prompt.
- Decode figures are at the stated depth, i.e. with the KV cache full.

## Test hardware

| | |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop, 6,141 MiB, Ada (SM 8.9), 75 W power limit |
| Host link | PCIe 4.0 **x8** (measured ~13 GB/s effective for H2D) |
| CPU | AMD Ryzen 7 7735HS, 8 cores / 16 threads |
| RAM | 16 GB installed, 14 GiB visible to the OS (iGPU reserves the rest) |
| OS | Pop!_OS 24.04, kernel 7.2.6 |
| CUDA | 13.4, driver 595.104.02 |
| Build | `Release`, `GGML_CUDA=ON`, `GGML_NATIVE=ON`, `GGML_CPU_REPACK=ON` |

This is a deliberately weak host: the GPU is the smallest Ada part with a 96-bit memory bus,
and the CPU feeds it from DDR5. Everything below is about making that combination work.

## Why this fork exists

Running a 35B MoE on a 6 GB card means the routed expert weights cannot fit in VRAM. The usual
answer is to leave them in system RAM and let llama.cpp copy each layer's experts to the GPU per
ubatch — "op offload". That works, but vanilla llama.cpp then hits three walls:

1. **The copies are not overlapped with compute.** The timeline is a copy burst, then a compute
   burst, then the next copy. On this box that leaves the GPU idle for 16–17 % of a prefill and
   puts every expert byte on the critical path.
2. **The copies may be pageable.** If the host weights are not in pinned memory, every
   `cudaMemcpyAsync` blocks the host for its duration. Measured here: 24 s of host time inside
   `cudaMemcpyAsync` for a 25K prefill, versus 0.6 s once the weights are pinned.
3. **CUDA graphs silently stop being used**, for reasons that are not visible from logs — see
   [What changed](#what-changed).

This fork fixes all three. It does not change the model, the quantisation, or the numerics.

## What changed

Four changes, all in `ggml/`. Each was found by profiling and verified by measurement.

### 1. Pinned host memory for offloaded experts

`ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_backend_cuda_device_supports_op()`

The fork rejected any operation whose weight tensor lived in a CUDA host (pinned) buffer, except
TurboQuant MoE experts with an opt-in env var. That policy pushed the 6.8 GB expert set into
plain pageable RAM, where every H2D copy blocks the host.

The patch admits `GGML_OP_MUL_MAT_ID` with a host-buffer `src[0]`, so the model loader places
those tensors in `CUDA_Host` (allocated with `cudaMallocHost`). CUPTI confirms the change in
copy source kind: pageable → pinned. Isolated effect on one binary at 85K: **415 t/s pageable
vs 589 t/s pinned**.

### 2. CUDA-graph compatibility now matches the dispatch

`ggml/src/ggml-cuda/ggml-cuda.cu` — new `ggml_cuda_mul_mat_id_needs_sync()`

The graph predicate declared every `MUL_MAT_ID` uncapturable when `ne[2] > mmvq_mmid_max`,
although the dispatch routes exactly that case to MMQ, which never synchronises the stream.
Result: 624 of 632 prefill graph submissions were rejected and no graph was ever launched.

The predicate now mirrors the dispatch (MMVQ / MMQ / MMF are all capture-safe). Upstream
llama.cpp already does this; the fork had drifted.

### 3. CUDA-graph cache no longer thrashes

`ggml/src/ggml-cuda/common.cuh` — `max_cuda_graphs` 64 → 512

This model produces **80 distinct graph keys per ubatch** (attention/GDN and expert splits).
With a 64-entry LRU, a graph was evicted before its next reuse, so warmup never completed and
capture never happened. After the cap change, `cudaGraphLaunch` went from **0 to 3,112** per
prefill.

### 4. Expert prefetch: overlap the copies with compute

`ggml/src/ggml-backend.cpp` — ~200 lines, opt-in with `GGML_SCHED_PREFETCH_EXPERTS`

Ported from the public work by **thecodacus / Fable** in
[llama.cpp issue #25859](https://github.com/ggml-org/llama.cpp/issues/25859)
(branch `fable5/prefetch-experts`), which measured +64 % on a 12 GB card.

Above a batch threshold the scheduler stops waiting for the routing-ids readback (a hard device
sync that buys nothing at prefill batch sizes, where the router touches ~every expert anyway),
uploads the whole expert tensor through a second backend instance on the same device (own
stream), and orders it against compute with events over a ring of staging slots.

Measured here at 85K, same binary and flags:

| config | prefill | decode | VRAM |
|---|---|---|---|
| 13 layers, no prefetch | 584 t/s | 36.2 t/s | 5,453 MiB |
| 13 layers, 2 slots | 709 t/s | 36.4 t/s | 5,641 MiB |
| **13 layers, 3 slots** | **783 t/s** | **36.2 t/s** | 5,759 MiB |
| 14 layers, 3 slots | slots do not allocate (~110 MiB free of 330 needed) | — | — |

Three slots is the sweet spot: with two, the scheduler stalls waiting for the down-projection
upload. The staging ring costs `slots × max_expert_tensor` of VRAM, which is why this
configuration trades the 14th GPU layer for the prefetch — and comes out far ahead.

## Build

Requirements: CUDA 12.4+ (13.x tested), CMake 3.20+, a C++17 compiler, Linux.

```bash
git clone https://github.com/<you>/llama-cpp-turboquant-moe-prefetch.git
cd llama-cpp-turboquant-moe-prefetch

cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CPU_REPACK=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j"$(nproc)" --target llama-server llama-cli llama-bench llama-perplexity
```

Binaries land in `build/bin/`. No extra patches or submodules are required beyond the upstream
tree; `GGML_CUDA_FA_ALL_QUANTS` is **off** in the measured build.

## Run

The exact configuration used for the numbers above:

```bash
TURBO_AUTO_ASYMMETRIC=0 \
GGML_SCHED_PREFETCH_EXPERTS=1 \
./build/bin/llama-server \
  -m /path/to/Qwen3.6-35B-A3B-12Gb-2.6763bpw.gguf \
  -c 85000 \
  -fit off \
  -ngl 99 \
  -lm mlock \
  -ot 'blk\.([0-9]|1[0-9]|20|3[4-9])\.ffn_(up|down|gate)_exps\.weight=CPU' \
  -ctk turbo5 -ctv turbo5 \
  -fa on --jinja --tools all \
  -b 2048 -ub 768 -np 1 \
  --host 127.0.0.1 --port 8080
```

Two settings are load-bearing and easy to get wrong:

- **`TURBO_AUTO_ASYMMETRIC=0`** — without it the fork rewrites K to `q8_0` when GQA ≥ 6, which
  breaks the fused TurboQuant MMA pair and drops 85K decode from ~36 to ~9 t/s.
- **`-ot ...=CPU` with 27 blocks on CPU** — 13 layers' experts stay on the GPU. This is the
  number that leaves room for the prefetch staging ring. See the tuning section.

## Benchmark it yourself

`bench/depth_bench.py` (stdlib only) fills to a target depth, then measures bulk prefill, the
short prompt-eval at depth, and decode at depth — the three numbers used throughout this README.

```bash
python3 bench/depth_bench.py --url http://127.0.0.1:8080 --depth 85000 --gen 256
```

Example output:

```
depth=85000  bulk prefill 783.4 t/s   short eval 104.3 t/s   decode 36.2 t/s  (36.2 ms/tok)
```

For quality:

```bash
./build/bin/llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw \
  -c 4096 --chunks 8 -ngl 99 -fa on -lm mlock \
  -ot 'blk\.([0-9]|1[0-9]|20|3[4-9])\.ffn_(up|down|gate)_exps\.weight=CPU' \
  -ctk turbo5 -ctv turbo5 -ub 768
```

## Tuning for your own machine

The three knobs interact; change one at a time and re-measure.

1. **Expert layers on GPU** (`-ot` pattern). More layers means less CPU work but less room for
   the prefetch ring. Count the blocks left *out* of the pattern.
2. **Prefetch slots** (`GGML_SCHED_PREFETCH_EXPERTS=1|2|3|…`). Each slot is one max-sized expert
   tensor. The code silently disables itself when the slots do not fit — check VRAM, not the
   environment variable.
3. **Ubatch** (`-ub`). Prefill improves with ubatch because fewer bytes cross PCIe per prompt
   token. It is bounded by the attention workspace; on this card `-ub 896` works without
   prefetch but faults with it, and `-ub 1024` hits an illegal memory access at head_dim 256.

A practical procedure: fix `-ub 512`, add layers until VRAM is nearly full, then walk `-ub` up
until it faults, then spend the remaining VRAM on slots.

## Mechanics: why this works

Why the prefill is PCIe-bound, and why "just use more GPU layers" stops helping:

- At ubatch `u` with top-`k` routing over `E` experts, the number of distinct experts touched is
  `E · (1 − (1 − 1/E)^(u·k))`. For `E=256, k=8, u=768` that is ≈ 100 % — **every ubatch touches
  every expert**. The working set per ubatch is therefore the whole CPU-resident expert set
  (~6.8 GB), not the handful a token routes to.
- That is 6.8 GB per 768 tokens = ~9 MB of expert weights per prompt token, against ~13 GB/s of
  PCIe 4.0 x8. Copy-bound, unless the copies are hidden behind compute.
- At decode (batch 1) the opposite is true: only ~8 experts per layer are touched, ~219 MB per
  token from DDR5, and the limit is memory bandwidth on the CPU — which is why decode barely
  moves when you tune prefill, and why the two numbers are independent.

The measured phase split for a 25K prefill before the prefetch patch: kernel busy 22.7 s,
memcpy busy 16.3 s, idle 8.3 s, wall 47.3 s, overlap 0.000 s. After the patch the same fill runs
in 27.5 s. Note that **compute, not copy, is now the binding constraint** (~1,100 t/s at 25K),
so further gains have to come from kernels, not from more PCIe trickery.

## Limitations

- **Single-GPU only.** The prefetch patch is designed for one device; multi-GPU pipeline
  parallelism uses a different code path and was not tested here.
- **The model is not included.** Download one of the [tested models](#tested-models); the
  configuration above assumes the 40-layer / 256-expert `qwen35moe` shape. Other models need
  their own `-ot` pattern and will have different expert sizes (and therefore different slot
  costs).
- **TurboQuant KV types are fork-specific.** `-ctk turbo5 -ctv turbo5` will not work on upstream
  llama.cpp; they come from the turboquant fork this is based on.
- **`-ub 1024` faults** at head_dim 256 on this build. `-ub 768` is the tested maximum with
  prefetch enabled.
- Numerics: the prefetch path produces identical tokens in testing (perplexity matched to four
  decimal places), but it is an opt-in scheduler change, so validate on your own workload.

## Syncing with upstream

`upstream` points at [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant),
so upstream work is a normal merge:

```bash
git remote -v          # origin = this repo, upstream = TheTom/llama-cpp-turboquant
git fetch upstream
git merge upstream/feature/turboquant-kv-cache
```

To restore a known-good state after an accidental local update:

```bash
git fetch origin
git reset --hard origin/main
```

The four changes are also saved as a standalone patch in `patches/fork-prefill-fixes.patch`,
which applies to upstream commit `643956891`:

```bash
git checkout 643956891
git apply patches/fork-prefill-fixes.patch
```

## Credits

- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — the engine, MIT.
- **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)** — TurboQuant
  KV cache types (`turbo3/4/5/6`) and the fused TurboQuant attention kernels this build relies
  on. Without `turbo5` the KV cache does not fit at 85K on a 6 GB card.
- **thecodacus / Fable** — the expert-prefetch scheduler ported in change 4, published in
  [llama.cpp #25859](https://github.com/ggml-org/llama.cpp/issues/25859).
- The llama.cpp contributors whose CPU/GPU kernels make the 3B-active path viable on a laptop.

## License

MIT, inherited from llama.cpp. See [LICENSE](LICENSE).
