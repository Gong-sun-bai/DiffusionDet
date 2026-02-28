import torch
import torch.nn as nn
import math

class LowRankLinear(nn.Module):
    """
    y = (B @ (A @ x^T))^T + bias
    A: (r, in) , B: (out, r)
    """
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.A = nn.Linear(in_features, rank, bias=False)      # (r, in)
        self.B = nn.Linear(rank, out_features, bias=bias)      # (out, r)

    @torch.no_grad()
    def init_from_linear_svd(self, linear: nn.Linear):
        """
        SVD init so that B(A(x)) approx linear(x)
        W ≈ U_r S_r V_r^T
        Let A = sqrt(S) V^T, B = U sqrt(S)
        """
        W = linear.weight.data  # (out, in)
        device = W.device
        dtype = W.dtype

        # do SVD in float32 for stability
        W32 = W.float()
        U, S, Vh = torch.linalg.svd(W32, full_matrices=False)  # U:(out,k), S:(k,), Vh:(k,in)
        r = self.rank
        U_r = U[:, :r]                 # (out, r)
        S_r = S[:r]                    # (r,)
        Vh_r = Vh[:r, :]               # (r, in)

        # sqrt(S)
        Sr_sqrt = torch.sqrt(S_r + 1e-12)  # (r,)

        # Set A.weight: (r, in)  = diag(sqrt(S)) @ Vh
        A_w = (Sr_sqrt[:, None] * Vh_r)            # (r, in)
        # Set B.weight: (out, r) = U @ diag(sqrt(S))
        B_w = (U_r * Sr_sqrt[None, :])             # (out, r)

        self.A.weight.data.copy_(A_w.to(device=device, dtype=dtype))
        self.B.weight.data.copy_(B_w.to(device=device, dtype=dtype))

        if linear.bias is not None and self.B.bias is not None:
            self.B.bias.data.copy_(linear.bias.data)

    def forward(self, x):
        return self.B(self.A(x))
