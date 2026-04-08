
import numpy as np
import torch
from voyager_index._internal.inference.quantization.rotational import RotationalQuantizer, RoQConfig

def benchmark_colbert_roq():
    # Setup
    dim = 128 # ColBERT token dim
    n_docs = 100
    n_doc_tokens = 120
    n_query_tokens = 32
    
    # Generate data (Correlated)
    # We want "Doc A" to be relevant to "Query A".
    # Create random docs
    torch.manual_seed(42)
    docs = torch.randn(n_docs, n_doc_tokens, dim)
    docs = torch.nn.functional.normalize(docs, p=2, dim=-1)
    
    # Create query that matches Doc 0
    # Query tokens are close to random Doc 0 tokens
    # Q_tokens = D_0_tokens[random] + noise
    query = torch.zeros(n_query_tokens, dim)
    doc0 = docs[0]
    indices = torch.randint(0, n_doc_tokens, (n_query_tokens,))
    query = doc0[indices] + 0.1 * torch.randn(n_query_tokens, dim)
    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    
    # 1. Exact MaxSim
    # Sim: (Nq, Nd, Nt)
    # Actually ColBERT scores: for each doc, compute maxsim
    scores_exact = []
    
    for i in range(n_docs):
        d = docs[i] # (Nt, D)
        sim = query @ d.T # (Nq, Nt)
        max_sim = sim.max(dim=1).values.sum()
        scores_exact.append(max_sim.item())
        
    scores_exact = torch.tensor(scores_exact)
    top_doc_exact = scores_exact.argmax().item()
    print(f"Exact Top Doc: {top_doc_exact} (Score: {scores_exact[top_doc_exact]:.4f})")
    print(f"Doc 0 Score: {scores_exact[0]:.4f}")
    
    # 2. RoQ 1-Bit
    config = RoQConfig(dim=dim, num_bits=1)
    roq = RotationalQuantizer(config)
    
    # Flatten for quantization
    docs_flat = docs.view(-1, dim)
    docs_q = roq.quantize(docs_flat, store=True) # Dict with 'codes'
    
    query_q = roq.quantize(query, store=False)
    
    # 1-Bit MaxSim
    # Unpack bits
    q_codes = query_q['codes'] # (Nq, n_bytes)
    q_bits = np.unpackbits(q_codes, axis=1) # (Nq, dim)
    
    d_codes = docs_q['codes'] # (Nd*Nt, n_bytes)
    d_bits = np.unpackbits(d_codes, axis=1) # (Nd*Nt, dim)
    
    # Reshape doc bits
    d_bits = d_bits.reshape(n_docs, n_doc_tokens, -1)
    
    # Verify match
    dim_bits = q_bits.shape[1]
    
    scores_roq = []
    for i in range(n_docs):
        d_b = d_bits[i] # (Nt, dim)
        
        # Hamming Distance matrix
        # (Nq, 1, dim) x (1, Nt, dim) -> (Nq, Nt, dim)
        # Using XOR sum
        # Efficient way: matrix mult for float? 
        # popcount(a^b) = a.sum + b.sum - 2a.b
        # Let's use float matmul for convenience (bits are 0/1)
        q_float = torch.tensor(q_bits, dtype=torch.float32)
        d_float = torch.tensor(d_b, dtype=torch.float32)
        
        dot = q_float @ d_float.T
        q_sum = q_float.sum(dim=1, keepdim=True)
        d_sum = d_float.sum(dim=1).unsqueeze(0)
        hamming = q_sum + d_sum - 2 * dot
        
        # Sim ~ dim - 2*hamming (linear proxy)
        # Actually Hamming = 0 -> Sim = 1. Hamming = dim -> Sim = -1.
        # Sim = 1 - 2*(Hamming/dim)
        sim = 1.0 - 2.0 * (hamming / dim_bits)
        
        max_sim = sim.max(dim=1).values.sum()
        scores_roq.append(max_sim.item())
        
    scores_roq = torch.tensor(scores_roq)
    top_doc_roq = scores_roq.argmax().item()
    print(f"RoQ 1-Bit Top Doc: {top_doc_roq} (Score: {scores_roq[top_doc_roq]:.4f})")
    
    # Rank Correlation
    rank_exact = scores_exact.argsort(descending=True)
    rank_roq = scores_roq.argsort(descending=True)
    
    # Check if Doc 0 is in top 1/5/10
    pos_exact = (rank_exact == 0).nonzero(as_tuple=True)[0].item()
    pos_roq = (rank_roq == 0).nonzero(as_tuple=True)[0].item()
    
    print(f"Doc 0 Rank Exact: {pos_exact}")
    print(f"Doc 0 Rank RoQ: {pos_roq}")
    
    if pos_roq < 5:
        print("SUCCESS: 1-Bit RoQ preserves ColBERT ranking!")
    else:
        print("FAILURE: 1-Bit RoQ lost the signal.")

if __name__ == "__main__":
    benchmark_colbert_roq()
