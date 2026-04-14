import torch
import torch.nn.functional as F


class TabuPotentialField:
    """
    Implements the Neuro-Symbolic Control Law.
    Minimizes E(v) to find the optimal search vector under constraints.
    """

    def __init__(
        self,
        lambda_repulsion: float = 1.0,
        trust_region: float = 0.1,
        lr: float = 0.1,
        steps: int = 5,
    ):
        self.lambda_repulsion = lambda_repulsion
        self.trust_region = trust_region
        self.lr = lr
        self.steps = steps

    def solve(
        self,
        v_prop: torch.Tensor,
        memory_vectors: torch.Tensor,
        memory_intents: torch.Tensor,
        current_intent: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            v_prop: (B, D) - Proposed vector from Neural Core
            memory_vectors: (B, M, D) - Visited vectors
            memory_intents: (B, M) - Discrete intent IDs of visits
            current_intent: (B) - Current intent ID
            sigmas: (B, M) - Local density of visited nodes

        Returns:
            v_opt: (B, D) - Optimized query vector
        """
        B, _ = v_prop.shape
        v = v_prop.clone().detach().requires_grad_(True)

        for _ in range(self.steps):
            v_norm = F.normalize(v, p=2, dim=1)
            fidelity = 1.0 - torch.sum(v_norm * v_prop, dim=1)

            if memory_vectors is not None and memory_vectors.shape[1] > 0:
                mask = (memory_intents == current_intent.unsqueeze(1)).float()
                dists = 1.0 - torch.sum(v_norm.unsqueeze(1) * memory_vectors, dim=2)
                repulsion = torch.exp(-dists / (2 * sigmas.pow(2) + 1e-6))
                repulsion_energy = torch.sum(mask * repulsion, dim=1)
            else:
                repulsion_energy = torch.zeros(B, device=v.device)

            total_energy = fidelity + self.lambda_repulsion * repulsion_energy
            grad = torch.autograd.grad(total_energy.sum(), v)[0]

            with torch.no_grad():
                update = -self.lr * grad
                update_norm = update.norm(p=2, dim=1, keepdim=True)
                scale = torch.clamp(update_norm, max=self.trust_region) / (update_norm + 1e-8)
                update = update * scale

                v.add_(update)
                v.div_(v.norm(p=2, dim=1, keepdim=True) + 1e-8)

            v.requires_grad_(True)

        return v.detach()
