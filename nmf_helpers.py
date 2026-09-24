"""Shared plotting and matching utilities for the NMF course figures.

Used by generate_figures_intuition.py, generate_figures_practice.py,
generate_figures_mathematics.py, and generate_figures_exercises.py so that
component colours, heatmap conventions, and the permutation/sign matching
used before any true-vs-estimated comparison stay identical across chapters.
"""
import numpy as np

# Consistent component colours used in every figure across the whole course.
# Component 1 is always this blue, Component 2 always this orange, and so on.
COMPONENT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

FIG_WIDTH_1PANEL = 5.5
FIG_WIDTH_2PANEL = 10.0
FIG_WIDTH_3PANEL = 13.5
DPI = 200


def match_components(H_true, H_est):
    """Match estimated components to true components by correlation.

    H_true, H_est: (k, p) arrays with the same p. Returns a permutation
    `perm` such that H_est[perm[r]] best corresponds to H_true[r], found by
    a greedy highest-correlation assignment (k is always small in this
    course, so a full Hungarian solver is unnecessary).
    """
    k = H_true.shape[0]
    corr = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ti, ei = H_true[i], H_est[j]
            corr[i, j] = np.corrcoef(ti, ei)[0, 1]

    perm = [-1] * k
    used_est = set()
    for true_idx in np.argsort(-corr.max(axis=1)):
        order = np.argsort(-corr[true_idx])
        for est_idx in order:
            if est_idx not in used_est:
                perm[true_idx] = est_idx
                used_est.add(est_idx)
                break
    return np.array(perm), corr


def normalize_wh(W, H):
    """Rescale each component so H's rows have unit L2 norm, moving the
    scale into W. This removes the WH = (WD)(D^-1 H) scaling ambiguity
    before comparing magnitudes across fits or against ground truth."""
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    H_norm = H / norms
    W_norm = W * norms.T
    return W_norm, H_norm


def nndsvd_init(X, k):
    """Basic NNDSVD initialization (Boutsidis & Gallopoulos, 2008).

    Returns (W0, H0) with W0 (n, k) and H0 (k, p), both non-negative. Used
    to give both the R and the Python fit in practice.qmd an identical,
    structured starting point instead of two independent random draws.
    Deliberately produces exact zero entries when a singular vector has no
    positive (or no negative) part to draw on for a given component; this
    is the same basic NNDSVD behaviour discussed in the mathematics
    chapter's note on zero locking under multiplicative updates.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n, p = X.shape
    W0 = np.zeros((n, k))
    H0 = np.zeros((k, p))

    W0[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H0[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])

    for j in range(1, k):
        u, v = U[:, j], Vt[j, :]
        u_p, u_n = np.maximum(u, 0), np.maximum(-u, 0)
        v_p, v_n = np.maximum(v, 0), np.maximum(-v, 0)
        u_p_norm, u_n_norm = np.linalg.norm(u_p), np.linalg.norm(u_n)
        v_p_norm, v_n_norm = np.linalg.norm(v_p), np.linalg.norm(v_n)
        term_p = u_p_norm * v_p_norm
        term_n = u_n_norm * v_n_norm

        if term_p >= term_n:
            u_final = u_p / u_p_norm if u_p_norm > 0 else u_p
            v_final = v_p / v_p_norm if v_p_norm > 0 else v_p
            sigma = term_p
        else:
            u_final = u_n / u_n_norm if u_n_norm > 0 else u_n
            v_final = v_n / v_n_norm if v_n_norm > 0 else v_n
            sigma = term_n

        W0[:, j] = np.sqrt(S[j] * sigma) * u_final
        H0[j, :] = np.sqrt(S[j] * sigma) * v_final

    return W0, H0


def make_pca_nmf_toy(seed=7, n=150):
    """A small 2-feature non-negative toy dataset used only for the PCA-vs-NMF
    and NMF-cone geometry illustrations (Chapters 1 and 3). Points are
    positive combinations of two basis directions plus a little noise, kept
    non-negative by construction, so both a PCA line and an NMF cone can be
    drawn through the same 2D cloud.
    """
    rng = np.random.default_rng(seed)
    basis1 = np.array([1.0, 0.25])
    basis2 = np.array([0.3, 1.0])
    c1 = rng.uniform(0.2, 1.8, n // 2)
    c2 = rng.uniform(0.2, 1.8, n // 2)
    pts_a = np.outer(c1, basis1) + np.outer(rng.uniform(0, 0.3, n // 2), basis2)
    pts_b = np.outer(rng.uniform(0, 0.3, n - n // 2), basis1) + np.outer(c2, basis2)
    pts = np.vstack([pts_a, pts_b])
    pts += rng.normal(0, 0.05, pts.shape)
    pts = np.clip(pts, 0.01, None)
    return pts, basis1, basis2


def savefig(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    print(f"Saved {path}")
