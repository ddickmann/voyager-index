"""Out-of-band sanity check: confirm score_b1_fused returns a real
tensor (i.e. the fused CUDA kernel actually dispatches) at the
production whole-corpus shape used by the running bench. If it returns
None, the bench is silently falling back to Triton."""
import os
os.environ.setdefault("VOYAGER_RROQ158_USE_B1_FUSED", "1")

import torch
from voyager_index._internal.kernels import cuda_b1_rroq158

print("env =", os.environ.get("VOYAGER_RROQ158_USE_B1_FUSED"))
print("is_available =", cuda_b1_rroq158.is_available())
print("sm =", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print("device =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

# Production whole-corpus shape used by the bench:
#   arguana: B=1401, T_pad=512 (padded to nearest 32 from p95=273)
#   rroq158_gs128 + dim=128 -> n_words=4, n_groups=1, query_bits=4
#   S=32 query tokens
B, T = 1401, 512
n_words, n_groups, query_bits, S = 4, 1, 4, 32
K = 8192
device = "cuda"

torch.manual_seed(0)
ds = torch.randint(-(1 << 30), 1 << 30, (B, T, n_words), dtype=torch.int32, device=device)
dn = torch.randint(-(1 << 30), 1 << 30, (B, T, n_words), dtype=torch.int32, device=device)
dsc = torch.randn(B, T, n_groups, device=device)
cid = torch.randint(0, K, (B, T), dtype=torch.int32, device=device)
cos_n = torch.randn(B, T, device=device)
sin_n = torch.randn(B, T, device=device)
dm = torch.ones(B, T, device=device)
qp = torch.randint(-(1 << 30), 1 << 30, (S, query_bits, n_words), dtype=torch.int32, device=device)
qm = torch.randn(S, 2, device=device)
qct = torch.randn(S, K, device=device)

scores = cuda_b1_rroq158.score_b1_fused(
    docs_sign=ds, docs_nz=dn, docs_scl=dsc,
    docs_cid=cid, docs_cos=cos_n, docs_sin=sin_n,
    docs_mask=dm, q_planes=qp, q_meta=qm, qc_table=qct,
)
print()
print("score_b1_fused return type:", type(scores).__name__)
if scores is None:
    print("RESULT: FUSED KERNEL NOT DISPATCHED (returned None) -- bench is on Triton fallback")
else:
    print(f"RESULT: FUSED KERNEL DISPATCHED -- scores.shape={tuple(scores.shape)}, "
          f"scores.dtype={scores.dtype}, scores.device={scores.device}")
    print(f"        first 5 scores: {scores[:5].cpu().tolist()}")
