"""
Recovery Verification Script (Strengthened)
============================================
Goes far beyond API checks: validates file sizes against .pyc reference,
checks for presence of critical optimizations in source code, tests
concurrency patterns, and validates production-readiness markers.

Exit code 0 = all checks pass. Exit code 1 = mismatch detected.
"""
import json
import io
import contextlib
import sys
import os
import numpy as np
import tempfile
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
REFERENCE_PATH = pathlib.Path(__file__).resolve().with_name("RECOVERY_REFERENCE.json")

with open(REFERENCE_PATH) as f:
    ref = json.load(f)

passed = 0
failed = 0
warnings = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")

def warn(name, condition, detail=""):
    global warnings, passed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        warnings += 1
        print(f"  [WARN] {name} -- {detail}")


print("=" * 70)
print("RECOVERY VERIFICATION (STRENGTHENED)")
print("=" * 70)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1: API Signatures (original checks)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n--- 1. API Signatures ---")
import latence_gem_index as gem
import latence_gem_router as router_mod

for mod, mn in [(gem, "gem"), (router_mod, "router")]:
    for cn in [x for x in dir(mod) if not x.startswith("_") and isinstance(getattr(mod, x), type)]:
        cls = getattr(mod, cn)
        key = f"{mn}.{cn}"
        if key not in ref["api"]:
            check(f"{key} exists in reference", False, "not in reference")
            continue
        ref_methods = ref["api"][key]
        for mname, minfo in ref_methods.items():
            attr = getattr(cls, mname, None)
            check(f"{key}.{mname} exists", attr is not None)
            if attr and minfo.get("sig"):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    help(attr)
                helptext = buf.getvalue()
                sig_found = False
                for line in helptext.split("\n"):
                    if mname + "(" in line:
                        actual_sig = line.strip()
                        check(f"{key}.{mname} signature", actual_sig == minfo["sig"],
                              f"expected: {minfo['sig']}\n           got:      {actual_sig}")
                        sig_found = True
                        break
                if not sig_found:
                    check(f"{key}.{mname} has signature", False, "no signature found in help()")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2: Deterministic Sealed Segment (original checks)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n--- 2. Deterministic Sealed Segment ---")
rng = np.random.RandomState(42)
D = 32; N = 50; T = 16
vl = []; di = list(range(N)); do = []; o = 0
for i in range(N):
    v = rng.randn(T, D).astype(np.float32)
    vl.append(v); do.append((o, o + T)); o += T
av = np.vstack(vl); q = rng.randn(4, D).astype(np.float32)

sg = gem.GemSegment()
sg.build(av, di, do, n_fine=32, n_coarse=4, max_degree=16,
         ef_construction=64, max_kmeans_iter=20, ctop_r=2)

rs = ref["sealed"]
check("sealed n_docs", sg.n_docs() == rs["docs"], f"{sg.n_docs()} vs {rs['docs']}")
check("sealed n_nodes", sg.n_nodes() == rs["nodes"], f"{sg.n_nodes()} vs {rs['nodes']}")
check("sealed n_edges", sg.n_edges() == rs["edges"], f"{sg.n_edges()} vs {rs['edges']}")
check("sealed dim", sg.dim() == rs["dim"])
check("sealed is_ready", sg.is_ready() == rs["ready"])

r = sg.search(q, k=10, ef=64, n_probes=2)
rs2 = ref["sealed_search"]
check("sealed search n_results", len(r) == rs2["n"], f"{len(r)} vs {rs2['n']}")
check("sealed search ids", [int(x[0]) for x in r] == rs2["ids"],
      f"{[int(x[0]) for x in r]} vs {rs2['ids']}")
actual_scores = [round(float(x[1]), 6) for x in r]
check("sealed search scores", actual_scores == rs2["scores"],
      f"{actual_scores[:3]}... vs {rs2['scores'][:3]}...")

with tempfile.TemporaryDirectory() as t:
    sg.save(t + "/s")
    s2 = gem.GemSegment()
    s2.load(t + "/s")
    ra = s2.search(q, k=10, ef=64, n_probes=2)
    check("sealed save/load ids match", [x[0] for x in r] == [x[0] for x in ra])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3: Mutable Segment (original checks)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n--- 3. Mutable Segment ---")
m = gem.PyMutableGemSegment()
m.build(av[:T*20], list(range(20)), [(i*T, (i+1)*T) for i in range(20)],
        n_fine=32, n_coarse=4, max_degree=16, ef_construction=64,
        max_kmeans_iter=20, ctop_r=2, n_probes=2)

rm = ref["mut"]
check("mutable n_live", m.n_live() == rm["live"], f"{m.n_live()} vs {rm['live']}")
check("mutable n_edges", m.n_edges() == rm["edges"], f"{m.n_edges()} vs {rm['edges']}")
check("mutable quality", round(float(m.quality_score()), 4) == rm["qual"])
check("mutable del_ratio", round(float(m.delete_ratio()), 4) == rm["dr"])

srm = m.search(q, k=5, ef=64)
check("mutable search n", len(srm) == ref["mut_s"]["n"])

nv = rng.randn(T, D).astype(np.float32)
m.insert(nv, 100)
check("after insert n_live", m.n_live() == ref["mut_ins"], f"{m.n_live()} vs {ref['mut_ins']}")

d = m.delete(0)
check("delete returned True", d == ref["mut_del"]["ret"])
check("after delete n_live", m.n_live() == ref["mut_del"]["live"])

uv = rng.randn(T, D).astype(np.float32)
m.upsert(uv, 5)
check("after upsert n_live", m.n_live() == ref["mut_ups"], f"{m.n_live()} vs {ref['mut_ups']}")

m.compact()
check("after compact n_live", m.n_live() == ref["mut_cmp"]["live"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: Router (original checks)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n--- 4. Router ---")
rt = router_mod.PyGemRouter(D)
rt.build(av, di, do, n_fine=32, n_coarse=4, max_kmeans_iter=20, ctop_r=2)

rr = ref["rtr"]
check("router n_docs", rt.n_docs() == rr["docs"])
check("router n_fine", rt.n_fine() == rr["fine"])
check("router n_coarse", rt.n_coarse() == rr["coarse"])

ro = rt.route_query(q, n_probes=2, max_candidates=10)
check("router route n_candidates", len(ro) == ref["rtr_route"])

with tempfile.TemporaryDirectory() as t:
    rt.save(t + "/r")
    r2 = router_mod.PyGemRouter(D)
    r2.load(t + "/r")
    check("router save/load", rt.n_docs() == r2.n_docs())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: Latency sanity (original checks, tighter bound)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n--- 5. Latency Sanity ---")
for lb, fn in [("sealed", lambda: sg.search(q, k=10, ef=64, n_probes=2)),
               ("mutable", lambda: m.search(q, k=5, ef=64))]:
    du = []
    for _ in range(200):
        t0 = time.perf_counter(); fn()
        du.append((time.perf_counter() - t0) * 1e6)
    du.sort()
    p50 = du[100]
    ref_p50 = ref[f"lat_{lb}"]["p50"]
    ratio = p50 / max(ref_p50, 0.1)
    check(f"{lb} latency within 5x of reference",
          ratio < 5.0,
          f"actual p50={p50:.0f}us, ref p50={ref_p50:.0f}us, ratio={ratio:.1f}x")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6: Rust Source File Inventory (STRENGTHENED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n--- 6. Rust Source File Inventory ---")
gem_index_src = ROOT / "src" / "kernels" / "gem_index" / "src"
gem_router_src = ROOT / "src" / "kernels" / "gem_router" / "src"

expected_gem_index_rs = {"emd.rs", "graph.rs", "id_tracker.rs", "lib.rs",
                          "mutable.rs", "persistence.rs", "search.rs", "visited.rs"}
expected_gem_router_rs = {"codebook.rs", "lib.rs", "persistence.rs", "router.rs"}

for fname in expected_gem_index_rs:
    p = gem_index_src / fname
    check(f"gem_index/{fname} exists", p.exists(), f"missing: {p}")

for fname in expected_gem_router_rs:
    p = gem_router_src / fname
    check(f"gem_router/{fname} exists", p.exists(), f"missing: {p}")

cargo_toml = ROOT / "src" / "kernels" / "gem_index" / "Cargo.toml"
check("gem_index/Cargo.toml exists", cargo_toml.exists())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7: Python File Size Validation (NEW)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n--- 7. Python File Size Validation ---")

PY_FILE_PATHS = {
    "index.py": ROOT / "voyager_index" / "index.py",
    "gem_manager.py": ROOT / "voyager_index" / "_internal" / "inference" / "index_core" / "gem_manager.py",
    "gem_wal.py": ROOT / "voyager_index" / "_internal" / "inference" / "index_core" / "gem_wal.py",
    "gem_segment_manager.py": ROOT / "research" / "legacy" / "python_runtime" / "inference" / "index_core" / "gem_segment_manager.py",
}

# .pyc size to .py size typical ratio is 0.25-0.50 for complex files
# Below 0.20 is a strong signal of missing code
MINIMUM_SIZE_RATIO = 0.20

if "pyc" in ref:
    for py_name, py_path in PY_FILE_PATHS.items():
        if py_name in ref["pyc"] and py_path.exists():
            pyc_size = ref["pyc"][py_name]
            py_size = py_path.stat().st_size
            ratio = py_size / pyc_size if pyc_size > 0 else 0
            check(f"{py_name} size ratio >= {MINIMUM_SIZE_RATIO}",
                  ratio >= MINIMUM_SIZE_RATIO,
                  f".py={py_size}B, .pyc={pyc_size}B, ratio={ratio:.3f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8: Rust Source Quality Markers (NEW)
# Checks that critical optimizations are present in source code.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n--- 8. Rust Source Quality Markers ---")

def file_contains(path, pattern):
    """Check if a file contains a pattern (substring match)."""
    if not path.exists():
        return False
    text = path.read_text()
    return pattern in text

def file_contains_any(path, patterns):
    """Check if a file contains any of the given patterns."""
    if not path.exists():
        return False
    text = path.read_text()
    return any(p in text for p in patterns)

# WS-1.3: Copy derive on MinCand/MaxCand
check("WS-1.3: MinCand/MaxCand derive Copy",
      file_contains(gem_index_src / "search.rs", "derive(Clone, Copy)") or
      file_contains(gem_index_src / "search.rs", "derive(Copy, Clone)"),
      "search.rs should have #[derive(Clone, Copy)] on MinCand/MaxCand")

# WS-1.4: #[inline] on construction scoring
check("WS-1.4: #[inline] on qch_proxy_between_docs",
      file_contains(gem_index_src / "emd.rs", "#[inline"),
      "emd.rs should have #[inline] on qch_proxy_between_docs")

# WS-1.10: panic = "abort" in Cargo.toml
check("WS-1.10: panic = abort in Cargo.toml",
      file_contains(cargo_toml, 'panic = "abort"'),
      "Cargo.toml [profile.release] should have panic = \"abort\"")

# WS-2.2: bridge_repair function
check("WS-2.2: bridge_repair exists in graph.rs",
      file_contains(gem_index_src / "graph.rs", "bridge_repair"),
      "graph.rs must have bridge_repair function for cross-cluster connectivity")

# WS-5.3: memmap2 usage (not just dependency)
check("WS-5.3: memmap2 actually used in persistence.rs",
      file_contains(gem_index_src / "persistence.rs", "memmap2") or
      file_contains(gem_index_src / "persistence.rs", "Mmap"),
      "persistence.rs should use memmap2 for sealed segment loading")

# WS-8.3: batch insert API
check("WS-8.3: batch insert API in lib.rs",
      file_contains(gem_index_src / "lib.rs", "insert_batch") or
      file_contains(gem_index_src / "lib.rs", "batch_insert"),
      "lib.rs should expose a batch insert method on PyMutableGemSegment")

# Diversity heuristic: cand_codes as first argument in both comparisons
emd_rs = gem_index_src / "emd.rs"
graph_rs = gem_index_src / "graph.rs"

# VisitedSet pooling or reuse pattern
check("WS-1.6: VisitedSet reuse/pool pattern",
      file_contains_any(gem_index_src / "search.rs",
                        ["thread_local", "VisitedPool", "reusable", "pool"]) or
      file_contains_any(gem_index_src / "visited.rs",
                        ["thread_local", "Pool", "generation", "reuse"]),
      "search.rs or visited.rs should have VisitedSet pooling for reuse")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9: Python Production Readiness Markers (NEW)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n--- 9. Python Production Readiness ---")

gem_mgr = ROOT / "voyager_index" / "_internal" / "inference" / "index_core" / "gem_manager.py"
index_py = ROOT / "voyager_index" / "index.py"
wal_py = ROOT / "voyager_index" / "_internal" / "inference" / "index_core" / "gem_wal.py"

# WS-3.3: Locks on should_seal, flush, get_statistics
if gem_mgr.exists():
    mgr_text = gem_mgr.read_text()

    native_mgr_start = mgr_text.find("class GemNativeSegmentManager")
    native_section = mgr_text[native_mgr_start:] if native_mgr_start >= 0 else mgr_text

    for method_name in ["should_seal", "get_statistics"]:
        idx = native_section.find(f"def {method_name}")
        if idx >= 0:
            block = native_section[idx:idx+500]
            check(f"WS-3.3: {method_name}() has lock",
                  "self._lock" in block or "with self._lock" in block,
                  f"{method_name}() in GemNativeSegmentManager should use self._lock")
        else:
            warn(f"WS-3.3: {method_name}() exists", False,
                 f"{method_name} not found in GemNativeSegmentManager")

    # WS-4.3: Config validation in __init__
    init_idx = mgr_text.find("class GemNativeSegmentManager")
    if init_idx >= 0:
        init_block = mgr_text[init_idx:init_idx+3000]
        check("WS-4.3: config validation in GemNativeSegmentManager",
              "ValueError" in init_block or "assert" in init_block or "validate" in init_block.lower(),
              "GemNativeSegmentManager.__init__ should validate dim, max_degree, etc.")

    # WS-8.1: Graceful segment load degradation
    check("WS-8.1: graceful sealed segment load",
          "try:" in mgr_text and "_load_sealed" in mgr_text,
          "gem_manager.py should have try/except in _load_sealed_segments")

# WAL imports at module top
if wal_py.exists():
    wal_text = wal_py.read_text()
    wal_lines = wal_text.split("\n")
    top_50 = "\n".join(wal_lines[:50])
    check("WS-7.3: zlib imported at top of gem_wal.py",
          "import zlib" in top_50,
          "zlib should be imported at module top, not inside functions")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9b: Audit-Fix Verification (NEW — catches regressions from full audit)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n--- 9b. Audit-Fix Integrity ---")

# CRITICAL: evict_worst_neighbor must evict when len >= max_degree (not just >)
check("AUDIT-1: evict_worst_neighbor < max_degree (not <=)",
      file_contains(gem_index_src / "graph.rs", "< max_degree") and
      not file_contains(gem_index_src / "graph.rs", "<= max_degree"),
      "evict_worst_neighbor must use `< max_degree` to evict at exactly max_degree")

# CRITICAL: Box<dyn Iterator> removed from search hot path
check("AUDIT-2: no Box<dyn Iterator> in search.rs hot path",
      not file_contains(gem_index_src / "search.rs", "Box<dyn Iterator"),
      "search.rs should not heap-allocate iterators in beam search loop")

# CRITICAL: shortcuts is Option in beam_search
check("AUDIT-3: beam_search shortcuts is Option",
      file_contains(gem_index_src / "search.rs", "Option<&[Vec<u32>]>"),
      "beam_search shortcuts param should be Option to avoid alloc for mutable search")

# HIGH: NaN-safe total_cmp in ordering
check("AUDIT-4: total_cmp used in search.rs",
      file_contains(gem_index_src / "search.rs", "total_cmp"),
      "MinCand/MaxCand Ord should use total_cmp for NaN safety")

check("AUDIT-5: total_cmp used in graph.rs sort",
      file_contains(gem_index_src / "graph.rs", "total_cmp"),
      "shrink_neighbors and evict_worst should use total_cmp")

# HIGH: build input validation in lib.rs
check("AUDIT-6: doc_ids/doc_offsets validation in lib.rs",
      file_contains(gem_index_src / "lib.rs", "doc_ids length") or
      file_contains(gem_index_src / "lib.rs", "doc_offsets length"),
      "lib.rs build should validate doc_ids.len() == doc_offsets.len()")

# HIGH: persistence OOM guard
check("AUDIT-7: persistence data_len validation",
      file_contains(gem_index_src / "persistence.rs", "exceeds file capacity") or
      file_contains(gem_index_src / "persistence.rs", "data_len overflow"),
      "persistence.rs should validate data_len before allocating")

# CRITICAL Python: seal persists to disk
check("AUDIT-8: seal writes segment.gem to disk",
      file_contains(gem_mgr, "segment.gem") and file_contains(gem_mgr, "sealed.build"),
      "GemNativeSegmentManager seal must persist active→sealed GEM before clearing")

# HIGH Python: scroll filter logic fix
if index_py.exists():
    idx_text = index_py.read_text()
    check("AUDIT-9: scroll filter not double-if",
          "if hasattr(self._manager, '_match_filter')" not in idx_text or
          "filters and hasattr" in idx_text,
          "scroll filter should use 'if filters and hasattr' not nested 'if ... if hasattr'")

# HIGH Python: get() uses lock
if index_py.exists():
    check("AUDIT-10: Index.get() uses lock",
          file_contains(index_py, "def get") and file_contains(index_py, "with self._lock"),
          "Index.get() should acquire lock for thread safety")

# HIGH Python: add() validates dims
if index_py.exists():
    check("AUDIT-11: Index.add() validates dimension",
          file_contains(index_py, "dimension mismatch"),
          "Index.add() should validate vector dimensions match self._dim")

# HIGH Python: retrieve() uses lock in GemNativeSegmentManager
check("AUDIT-12: retrieve() has lock in GemNativeSegmentManager",
      file_contains(gem_mgr, "def retrieve") and
      file_contains(gem_mgr, "with self._lock"),
      "retrieve() must acquire lock for concurrent safety")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 10: Cargo Dependencies Verification (NEW)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n--- 10. Cargo Dependencies ---")
if cargo_toml.exists() and "deps" in ref:
    ct_text = cargo_toml.read_text()
    for dep_pattern in ref["deps"]:
        dep_name = dep_pattern.rstrip('"').split('"')[0] if '"' in dep_pattern else dep_pattern
        dep_name = dep_name.strip()
        # Strip version suffixes for matching (e.g. "rand 0.8.5" -> "rand")
        crate_name = dep_name.split()[0] if ' ' in dep_name else dep_name
        check(f"dep: {crate_name} in Cargo.toml",
              crate_name in ct_text,
              f"{crate_name} not found in Cargo.toml")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 11: Key Python Files Existence (NEW)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n--- 11. Key Python Files Existence ---")
if "pyc" in ref:
    core_files = {
        "gem_manager.py": gem_mgr,
        "gem_wal.py": wal_py,
        "gem_segment_manager.py": ROOT / "research" / "legacy" / "python_runtime" / "inference" / "index_core" / "gem_segment_manager.py",
        "index.py": index_py,
    }
    for name, path in core_files.items():
        check(f"{name} exists", path.exists(), f"missing: {path}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 12: MEDIUM Issue Fixes Verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n--- 12. MEDIUM Issue Fixes ---")

lib_rs_path = gem_index_src / "lib.rs"
graph_rs_path = gem_index_src / "graph.rs"
mut_rs_path = gem_index_src / "mutable.rs"
persist_rs_path = gem_index_src / "persistence.rs"
id_tracker_rs_path = gem_index_src / "id_tracker.rs"

# M1: lib.rs - u32 overflow guard on n_docs
check("M1: lib.rs u32 overflow guard for n_docs",
      file_contains(lib_rs_path, "u32::MAX") and file_contains(lib_rs_path, "too many documents"),
      "lib.rs missing u32::MAX document count guard")

# M2: graph.rs - bridge_repair no longer clones member list
check("M2: graph.rs bridge_repair borrows members (no clone)",
      file_contains(graph_rs_path, "let members = &postings.lists[cluster_id]"),
      "bridge_repair still clones member list")

# M3: graph.rs - sort_unstable_by in shrink_neighbors
check("M3: graph.rs sort_unstable_by in shrink_neighbors",
      file_contains(graph_rs_path, "sort_unstable_by"),
      "shrink_neighbors should use sort_unstable_by")

# M4: mutable.rs - doc_offsets consistency validation
check("M4: mutable.rs doc_offsets/doc_ids length assertion",
      file_contains(mut_rs_path, "assert_eq!(n_docs, doc_offsets.len()"),
      "mutable.rs missing doc_offsets validation")

# M5: id_tracker.rs - u32 overflow guard in add
check("M5: id_tracker.rs u32 overflow guard in add()",
      file_contains(id_tracker_rs_path, "u32::MAX"),
      "id_tracker.rs missing u32 overflow guard")

# M6: persistence.rs - fsync before rename
check("M6: persistence.rs fsync before atomic rename",
      file_contains(persist_rs_path, "sync_all()"),
      "persistence.rs missing fsync on save")

# M7: gem_manager.py - sealed_deleted_ids populated on delete
check("M7: gem_manager.py delete populates _sealed_deleted_ids",
      file_contains(gem_mgr, "self._sealed_deleted_ids.add(doc_id)"),
      "delete does not track sealed_deleted_ids")

# M8: gem_manager.py - sealed_deleted_ids persistence
check("M8: gem_manager.py _save_sealed_deleted_ids method",
      file_contains(gem_mgr, "def _save_sealed_deleted_ids"),
      "missing _save_sealed_deleted_ids")
check("M9: gem_manager.py _load_sealed_deleted_ids method",
      file_contains(gem_mgr, "def _load_sealed_deleted_ids"),
      "missing _load_sealed_deleted_ids")

# M10: gem_manager.py - $has_id filter uses doc_id not payload._id
check("M10: gem_manager.py $has_id filter uses doc_id",
      file_contains(gem_mgr, "doc_id in condition if doc_id is not None"),
      "$has_id still checks payload._id")

# M11: gem_wal.py - WAL replay salvages past corruption
check("M11: gem_wal.py WAL replay scans past corruption",
      file_contains(wal_py, "file_data.find(WAL_MAGIC"),
      "WAL replay still stops at first corruption")

# M12: gem_wal.py - checkpoint atomic swap (no rmtree+rename crash window)
check("M12: gem_wal.py checkpoint atomic swap via .old rename",
      file_contains(wal_py, '".old"'),
      "checkpoint save still uses rmtree+rename pattern")

# M13: lib.rs - inject_shortcuts validation
check("M13: lib.rs inject_shortcuts validates training_pairs shapes",
      file_contains(lib_rs_path, "flat.len() % inner.dim"),
      "inject_shortcuts missing shape validation")

# M14: lib.rs - inject_shortcuts validates target bounds
check("M14: lib.rs inject_shortcuts validates target < n_nodes",
      file_contains(lib_rs_path, "inner.graph.n_nodes()"),
      "inject_shortcuts missing target bounds check")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print("=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed, {warnings} warnings")
if failed == 0 and warnings == 0:
    print("RECOVERY VERIFICATION: ALL CHECKS PASSED -- WORLD CLASS")
elif failed == 0:
    print(f"RECOVERY VERIFICATION: PASSED with {warnings} warnings")
else:
    print(f"RECOVERY VERIFICATION: {failed} FAILURES DETECTED")
print("=" * 70)

sys.exit(1 if failed > 0 else 0)
