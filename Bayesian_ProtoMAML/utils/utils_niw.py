import torch
import math

KAPPA0 = 0.01
NU0_OFFSET = 2
LAMBDA0_SCALE = 1.0

def niw_posterior(z):
    K, d = z.shape
    device, dtype = z.device, z.dtype

    mu0 = torch.zeros(d, device=device, dtype=dtype)
    Lambda0 = LAMBDA0_SCALE * torch.eye(d, device=device, dtype=dtype)

    z_bar = z.mean(dim=0)
    S = (z - z_bar).T @ (z - z_bar)

    kappa_n = KAPPA0 + K
    nu_n = d + NU0_OFFSET + K

    mu_n = (KAPPA0 * mu0 + K * z_bar) / kappa_n
    diff = (z_bar - mu0).unsqueeze(1)

    Lambda_n = Lambda0 + S + (KAPPA0 * K / kappa_n) * (diff @ diff.T)

    dof = nu_n - d + 1
    Sigma_n = ((kappa_n + 1) / (kappa_n * dof)) * Lambda_n

    return mu_n, Sigma_n, dof

def log_student_t(x, mu, Sigma, dof):
    d = x.shape[1]
    device, dtype = x.device, x.dtype

    if not torch.is_tensor(dof):
        dof = torch.tensor(dof, device=device, dtype=dtype)
    else:
        dof = dof.to(device=device, dtype=dtype)

    d_t = torch.tensor(float(d), device=device, dtype=dtype)

    xm = x - mu.unsqueeze(0)

    jitter = 1e-6 * torch.eye(d, device=device, dtype=dtype)
    L = torch.linalg.cholesky(Sigma + jitter)
    invSigma = torch.cholesky_inverse(L)
    logdet = 2 * torch.sum(torch.log(torch.diag(L)))

    delta = torch.sum((xm @ invSigma) * xm, dim=1)

    return (
        torch.lgamma((dof + d_t) / 2)
        - torch.lgamma(dof / 2)
        - 0.5 * d_t * torch.log(dof * torch.tensor(math.pi, device=device))
        - 0.5 * logdet
        - ((dof + d_t) / 2) * torch.log1p(delta / dof)
    )
