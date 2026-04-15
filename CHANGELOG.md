# Changelog

## 2026-04-15 - Multi-thread likelihood optimization pass

### What changed

- Added persistent SIMD-packed `tPij_rr` storage on each edge and thread-local `site_dot_prod` buffers to remove repeated packing and scratch reuse in hot kernels.
- Added cached wavefront scheduling state in `t_tree` so full-traversal partial-likelihood jobs are rebuilt only when the effective layout changes.
- Cached packed eigenvectors with epoch-based invalidation and reused them from the AVX and SSE likelihood backends.
- Parallelized full-tree `Update_PMat_At_Given_Edge` updates and made the AA `PMat` path thread-safe by replacing shared scratch arrays with local storage.
- Added fused AVX/SSE `Update_Eigen_Lr + Lk_Sites` team kernels so site likelihood consumes freshly computed dot products with better locality.
- Re-enabled multi-thread `dLk`, fixed the OpenMP race in its per-site loop, and retuned the activation threshold so compressed-pattern workloads that are too small still stay serial.

### Why these decisions

- `tPij_rr` packing was still partly serialized and repeated for every local job. Keeping a packed copy per edge shifts the cost to matrix-update time and removes redundant memory traffic.
- The wavefront DAG is stable across many traversals. Caching the jobs and level offsets avoids rebuilding the same schedule on every full likelihood pass.
- Eigen packing changes only when eigenvectors change. Epoch-based invalidation keeps the fast path cheap while remaining correct when the model updates.
- Fusing eigen updates with site likelihood reduces cache misses on `dot_prod`, but the implementation still writes back to `tree->dot_prod` to preserve downstream behavior.
- Exact MT `dLk` needs deterministic per-site accumulation. The bug fix makes it correct, and the larger work threshold avoids paying OpenMP overhead on small compressed datasets.

### Benchmarks

Environment:

- `OMP_NUM_THREADS=16`
- `OMP_PROC_BIND=close`
- `OMP_PLACES=cores`

`ns=4` benchmark (`examples/nucleic`, repeated to `56704` sites, `382` compressed patterns):

- before: `11.79 12.38 12.12 11.78 12.24`
- after: `4.86 6.11 4.94 5.09 5.11`
- median speedup: `12.12 / 5.09 = 2.38x`

`ns=20` benchmark (`examples/proteic`, repeated to `35008` sites, `429` compressed patterns):

- before: `30.37 19.32 16.83 27.49 17.12`
- after: `16.45 16.35 17.57 18.90 18.52`
- median speedup: `19.32 / 17.57 = 1.10x`

### Validation

- `make -j22`
- final `ns=4` log-likelihood: `-344725.278233302757143974304`
- final `ns=20` log-likelihood: `-796355.107519890763796865940`
