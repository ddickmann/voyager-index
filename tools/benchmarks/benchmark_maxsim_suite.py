
import numpy as np
import torch
import time
import pandas as pd
from voyager_index._internal.inference.quantization.rotational import RotationalQuantizer, RoQConfig
from voyager_index._internal.kernels.maxsim import fast_colbert_scores
from voyager_index._internal.kernels.triton_roq import roq_maxsim_1bit, roq_maxsim_8bit, roq_maxsim_4bit, roq_maxsim_2bit

# Try importing Rust
try:
    from latence_solver import compute_max_sim_batch, compute_max_sim_batch_f32
    RUST_AVAILABLE = True
except ImportError as e:
    print(f"Rust bindings not available. Skipping Rust benchmark. Error: {e}")
    RUST_AVAILABLE = False

def generate_colbert_data(n, n_doc, n_q_tokens, n_d_tokens, dim, device='cuda'):
    # Generate correlated data for valid ranking
    torch.manual_seed(42)
    
    # Docs: (N_doc, Nt, D)
    docs = torch.randn(n_doc, n_d_tokens, dim, device=device)
    docs = torch.nn.functional.normalize(docs, p=2, dim=-1)
    
    # Queries: (N_query, Ns, D)
    # Make queries relevant to random docs
    queries = []
    truth = []
    for i in range(n):
        target_doc_idx = np.random.randint(0, n_doc)
        truth.append(target_doc_idx)
        
        target_doc = docs[target_doc_idx] # (Nt, D)
        
        # Select random tokens from doc + noise
        indices = torch.randint(0, n_d_tokens, (n_q_tokens,))
        q = target_doc[indices] + 0.1 * torch.randn(n_q_tokens, dim, device=device)
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        queries.append(q)
        
    queries = torch.stack(queries)
    # Ensure queries is tensor
    quantq = {'codes': queries.contiguous()} # Mock dict for stack_roq compatibility
    # Actually stack_roq expects dict with codes.
    # Note: We need to handle this in benchmark suite.
    # Current stack_roq stacks 'codes'. if 'codes' is float tensor, it stacks floats.
    return queries, docs, truth

def measure_speed(func, name, n_runs=5):
    # Warmup
    func()
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n_runs):
        func()
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / n_runs
    print(f"[{name}] Avg Time: {avg_time*1000:.2f} ms")
    return avg_time

def benchmark_maxsim_suite():
    print("=== MaxSim Benchmark Suite (Python vs Triton vs RoQ) ===")
    
    dim = 128
    n_queries = 20
    n_docs = 1000 # Enough for Triton speedup
    n_q_tokens = 32
    n_d_tokens = 120
    
    device = 'cuda'
    
    print(f"Config: Q=({n_queries}, {n_q_tokens}), D=({n_docs}, {n_d_tokens}), dim={dim}")
    queries, docs, truth = generate_colbert_data(n_queries, n_docs, n_q_tokens, n_d_tokens, dim, device)
    
    results = []
    
    # 1. Python Baseline (FP32)
    # Using PyTorch implementation (similar to _colbert_scores_cpu but on GPU)
    def python_maxsim():
        scores = torch.zeros(n_queries, n_docs, device=device)
        for i in range(n_queries):
            q = queries[i] # (Ns, D)
            sim = q @ docs.transpose(1, 2) # (Ns, Nd, Nt) - memory intensive if batched
            # Compute pairwise to save memory
            # q: (Ns, D), docs: (Nd, Nt, D) -> (Nd, Nt, D)
            # einsum 'sd,ntd->nt' ? No.
            # q @ d.T -> (Ns, D) @ (D, Nt) -> (Ns, Nt)
            
            # Batch doc processing
            # (Ns, D) @ (N_docs, D, Nt) -> (N_docs, Ns, Nt) 
            # docs_T: (N_docs, D, Nt)
            d_T = docs.transpose(1, 2)
            sims = torch.matmul(q.unsqueeze(0), d_T) # (1, Ns, D) @ (Nd, D, Nt) -> (Nd, Ns, Nt)
            # max over Nt
            max_sims = sims.max(dim=2).values # (Nd, Ns)
            # sum over Ns
            scores[i] = max_sims.sum(dim=1)
        return scores

    print("Running Python FP32 (GPU)...")
    t_py = measure_speed(lambda: python_maxsim(), "Python FP32 (GPU)")
    scores_py = python_maxsim()
    results.append({"Method": "Python FP32 (GPU)", "Time_ms": t_py*1000, "Recall@1": 1.0, "Correlation": 1.0})
    
    ref_scores = scores_py.cpu().numpy()

    # 1b. Python FP32 (CPU)
    print("Running Python FP32 (CPU)...")
    # Move huge tensors to CPU might be slow, do it once.
    queries_cpu = queries.cpu()
    docs_cpu = docs.cpu()
    
    def python_maxsim_cpu():
        # Identical logic but on CPU tensors
        scores = torch.zeros(n_queries, n_docs, device='cpu')
        d_T = docs_cpu.transpose(1, 2)
        for i in range(n_queries):
             q = queries_cpu[i]
             sims = torch.matmul(q.unsqueeze(0), d_T)
             scores[i] = sims.max(dim=2).values.sum(dim=1)
        return scores

    t_py_cpu = measure_speed(lambda: python_maxsim_cpu(), "Python FP32 (CPU)")
    results.append({"Method": "Python FP32 (CPU)", "Time_ms": t_py_cpu*1000, "Recall@1": 1.0, "Correlation": 1.0})
    
    # Helper to check accuracy
    def calc_metrics(scores_tensor, name, time_s):
        s = scores_tensor.cpu().numpy()
        
        # Recall@1
        hits = 0
        for i in range(n_queries):
            pred_idx = np.argmax(s[i])
            if pred_idx == truth[i]:
                hits += 1
        recall = hits / n_queries
        
        # Pearson/Spearman Correlation (avg over queries)
        corrs = []
        for i in range(n_queries):
            # Rank correlation
            df = pd.DataFrame({'ref': ref_scores[i], 'pred': s[i]})
            corr = df.corr(method='spearman').iloc[0,1]
            corrs.append(corr)
        avg_corr = np.mean(corrs)
        
        results.append({
            "Method": name,
            "Time_ms": time_s * 1000,
            "Recall@1": recall,
            "Correlation": avg_corr
        })
        print(f"  Recall@1: {recall:.2f}, Rank Corr: {avg_corr:.4f}")

    # 2. Base Triton (FP16)
    print("Running Triton FP16...")
    t_triton = measure_speed(lambda: fast_colbert_scores(queries, docs, use_quantization=False), "Triton FP16")
    scores_triton = fast_colbert_scores(queries, docs, use_quantization=False)
    calc_metrics(scores_triton, "Triton FP16", t_triton)
    
    # 3. Base Triton (INT8)
    print("Running Triton INT8 (AbsMax)...")
    t_triton_int8 = measure_speed(lambda: fast_colbert_scores(queries, docs, use_quantization=True, quantization_mode="int8"), "Triton INT8")
    scores_triton_int8 = fast_colbert_scores(queries, docs, use_quantization=True, quantization_mode="int8")
    calc_metrics(scores_triton_int8, "Triton INT8", t_triton_int8)
    
    # --- RoQ Setup ---
    roq_8 = RotationalQuantizer(RoQConfig(dim=dim, num_bits=8))
    roq_1 = RotationalQuantizer(RoQConfig(dim=dim, num_bits=1))
    
    # Flatten docs for quantization
    docs_flat = docs.view(-1, dim)
    
    # Quantize 8-bit
    print("Quantizing 8-bit...")
    q_8_res = roq_8.quantize(queries.cpu().numpy().reshape(-1, dim), store=False)
    # Convert dict to list of QuantizedVector objects for validation if needed, or just use dict
    # Wait, benchmark code expects list of QuantizedVector?
    # Line 162: codes = np.stack([x.codes for x in res])
    # But quantize returns a DICT now! {codes: ..., scales: ...}
    # My previous benchmark (benchmark_maxsim_roq.py) likely used the dict.
    # But benchmark_maxsim_suite.py assumes `q_8_res` is a list of objects?
    # "q_8_res is list of QuantizedVector" comment in line 160.
    # Ah, I need to ADAPT `benchmark_maxsim_suite.py` to work with the NEW `quantize` return type (dict).
    
    # Let's fix lines 162-172 as well.
    d_8_res = roq_8.quantize(docs_flat.cpu().numpy(), store=False)
    
    # Convert to Tensors for Triton RoQ
    # Res is dict with keys: codes, scales, offsets, norms_sq
    def stack_roq(quantizer, res, B, T):
        if quantizer.config.num_bits == 1:
            codes = torch.tensor(res['codes'].reshape(B, T, -1), dtype=torch.uint8, device=device)
            meta = torch.zeros((B, T, 4), dtype=torch.float32, device=device)
            return codes, meta
        return quantizer.stack_triton_inputs(res, batch_size=B, item_count=T, device=device)
            
    q8_codes, q8_meta = stack_roq(roq_8, q_8_res, n_queries, n_q_tokens)
    d8_codes, d8_meta = stack_roq(roq_8, d_8_res, n_docs, n_d_tokens)
    
    # 4. Triton RoQ 8-bit
    print("Running Triton RoQ 8-bit...")
    t_roq8 = measure_speed(lambda: roq_maxsim_8bit(q8_codes, q8_meta, d8_codes, d8_meta), "Triton RoQ 8-bit")
    scores_roq8 = roq_maxsim_8bit(q8_codes, q8_meta, d8_codes, d8_meta)
    calc_metrics(scores_roq8, "Triton RoQ 8-bit", t_roq8)
    
    # 4b. Triton RoQ 4-bit
    print("Quantizing 4-bit...")
    roq_4 = RotationalQuantizer(RoQConfig(dim=dim, num_bits=4))
    
    q_4_res = roq_4.quantize(queries.cpu().numpy().reshape(-1, dim), store=False)
    d_4_res = roq_4.quantize(docs_flat.cpu().numpy(), store=False)
    
    q4_codes, q4_meta = stack_roq(roq_4, q_4_res, n_queries, n_q_tokens)
    d4_codes, d4_meta = stack_roq(roq_4, d_4_res, n_docs, n_d_tokens)
    
    print("Running Triton RoQ 4-bit...")
    t_roq4 = measure_speed(lambda: roq_maxsim_4bit(q4_codes, q4_meta, d4_codes, d4_meta), "Triton RoQ 4-bit")
    scores_roq4 = roq_maxsim_4bit(q4_codes, q4_meta, d4_codes, d4_meta)
    scores_roq4 = roq_maxsim_4bit(q4_codes, q4_meta, d4_codes, d4_meta)
    calc_metrics(scores_roq4, "Triton RoQ 4-bit", t_roq4)

    # 4c. Triton RoQ 2-bit (scalar affine path for kernel parity)
    print("Quantizing 2-bit...")
    roq_2 = RotationalQuantizer(RoQConfig(dim=dim, num_bits=2, group_size=dim))
    
    q_2_res = roq_2.quantize(queries.cpu().numpy().reshape(-1, dim), store=False)
    d_2_res = roq_2.quantize(docs_flat.cpu().numpy(), store=False)
    
    q2_codes, q2_meta = stack_roq(roq_2, q_2_res, n_queries, n_q_tokens)
    d2_codes, d2_meta = stack_roq(roq_2, d_2_res, n_docs, n_d_tokens)
    
    print("Running Triton RoQ 2-bit...")
    t_roq2 = measure_speed(lambda: roq_maxsim_2bit(q2_codes, q2_meta, d2_codes, d2_meta), "Triton RoQ 2-bit")
    scores_roq2 = roq_maxsim_2bit(q2_codes, q2_meta, d2_codes, d2_meta)
    calc_metrics(scores_roq2, "Triton RoQ 2-bit", t_roq2)

    # 5. Triton RoQ 1-bit (asymmetric query-screening kernel)
    print("Quantizing 1-bit...")
    roq_1 = RotationalQuantizer(RoQConfig(dim=dim, num_bits=1))
    d_1_res = roq_1.quantize(docs_flat.cpu().numpy(), store=False)
    q1_codes, q1_meta = roq_1.build_1bit_query_triton_inputs(
        queries.cpu().numpy().reshape(-1, dim),
        batch_size=n_queries,
        item_count=n_q_tokens,
        device=device,
        include_norm_sq=False,
    )
    d1_codes = roq_1.build_1bit_doc_triton_inputs(
        d_1_res,
        batch_size=n_docs,
        item_count=n_d_tokens,
        device=device,
    )
    
    print("Running Triton RoQ 1-bit...")
    t_roq1 = measure_speed(lambda: roq_maxsim_1bit(q1_codes, d1_codes, q1_meta), "Triton RoQ 1-bit")
    scores_roq1 = roq_maxsim_1bit(q1_codes, d1_codes, q1_meta)
    calc_metrics(scores_roq1, "Triton RoQ 1-bit", t_roq1)

    # 6a. Rust FP32 (Baseline)
    if RUST_AVAILABLE:
        print("Running Rust FP32 (Simulated FP16)...")
        # Reuse Triton inputs (already on GPU, move to CPU)
        q_f32 = queries.cpu().numpy()
        d_f32 = docs.cpu().numpy()
        
        t_rust_f32 = measure_speed(lambda: compute_max_sim_batch_f32(q_f32, d_f32), "Rust FP32")
        scores_rust_f32 = compute_max_sim_batch_f32(q_f32, d_f32)
        # Verify vs Python
        # calc_metrics(scores_rust_f32, "Rust FP32", t_rust_f32) 
        
    # 6b. Rust RoQ 8-bit
    if RUST_AVAILABLE:
        print("Running Rust RoQ 8-bit...")
        # Prepare numpy inputs on CPU
        q8_c_np = q8_codes.cpu().numpy()
        q8_m_np = q8_meta.cpu().numpy()
        d8_c_np = d8_codes.cpu().numpy()
        d8_m_np = d8_meta.cpu().numpy()
        
        t_rust8 = measure_speed(lambda: compute_max_sim_batch(q8_c_np, q8_m_np, d8_c_np, d8_m_np, 8, dim), "Rust RoQ 8-bit")
        scores_rust8_np = compute_max_sim_batch(q8_c_np, q8_m_np, d8_c_np, d8_m_np, 8, dim)
        scores_rust8 = torch.from_numpy(scores_rust8_np).to(device)
        calc_metrics(scores_rust8, "Rust RoQ 8-bit", t_rust8)
        
        # 7. Rust RoQ 1-bit
        print("Running Rust RoQ 1-bit...")
        q1_c_np = q1_codes.cpu().numpy()
        d1_c_np = d1_codes.cpu().numpy()
        # Meta not used but required by signature
        
        t_rust1 = measure_speed(lambda: compute_max_sim_batch(q1_c_np, q8_m_np, d1_c_np, d8_m_np, 1, dim), "Rust RoQ 1-bit")
        scores_rust1_np = compute_max_sim_batch(q1_c_np, q8_m_np, d1_c_np, d8_m_np, 1, dim)
        scores_rust1 = torch.from_numpy(scores_rust1_np).to(device)
        calc_metrics(scores_rust1, "Rust RoQ 1-bit", t_rust1)

        # 8. Rust RoQ 4-bit (New)
        print("Running Rust RoQ 4-bit...")
        q4_c_np = q4_codes.cpu().numpy()
        q4_m_np = q4_meta.cpu().numpy()
        d4_c_np = d4_codes.cpu().numpy()
        d4_m_np = d4_meta.cpu().numpy()
        
        t_rust4 = measure_speed(lambda: compute_max_sim_batch(q4_c_np, q4_m_np, d4_c_np, d4_m_np, 4, dim), "Rust RoQ 4-bit")
        scores_rust4_np = compute_max_sim_batch(q4_c_np, q4_m_np, d4_c_np, d4_m_np, 4, dim)
        scores_rust4 = torch.from_numpy(scores_rust4_np).to(device)
        calc_metrics(scores_rust4, "Rust RoQ 4-bit", t_rust4)
    
    # Print Table
    df = pd.DataFrame(results)
    print("\n=== Final Benchmark Results ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    benchmark_maxsim_suite()
