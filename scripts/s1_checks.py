"""S1 smoke tests.   Usage:  python scripts/s1_checks.py {env|sigma|gauss|all}
  env    environment, pinned vendor code, imports, weight-file hash
  sigma  VI-non-IID noise-level estimator on a patch with KNOWN noise
  gauss  authors' posterior sampler vs. a closed-form Gaussian posterior (no trained network needed)"""
import hashlib
import importlib
import pathlib
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from common import ROOT, VENDOR, get_device, use_vendor

EXPECTED_W_SHA = "c3beee608d08188a"        # sha256[:16] of Non-IID-Unet/model_state_10 (identical in both repos)


def _where(obj):
    f = pathlib.Path(sys.modules[obj.__module__].__file__).resolve()
    try:
        return str(f.relative_to(ROOT))
    except ValueError:
        return f"OUTSIDE REPO (shadowed!): {f}"


def check_env():
    print(f"python {platform.python_version()} on {platform.system()}")
    import torch
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GiB, compute capability {p.major}.{p.minor}")
    missing = []
    for m in ("numpy", "scipy", "skimage", "yaml", "h5py", "tqdm", "segyio"):
        try:
            importlib.import_module(m)
        except Exception:
            missing.append(m)
    print("missing packages:", missing or "none")
    for line in (ROOT / "vendor.lock").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, _, sha = line.split()
        d = VENDOR / name
        if not (d / ".git").exists():
            print(f"vendor/{name}: NOT FETCHED (python scripts/fetch_vendor.py)")
            continue
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True).stdout.strip()
        print(f"vendor/{name}: {'pinned OK' if head == sha else 'WRONG COMMIT ' + head[:8]}")
    use_vendor("IPS-RNA")
    from models import general_anneal_Langevin_dynamics_den as sampler
    from utils.networks import VDN
    print("sampler from:", _where(sampler))
    print("VDN from:    ", _where(VDN))
    from sigma_est import WEIGHTS
    sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()[:16]
    print("weights sha256[:16]:", sha, "(OK)" if sha == EXPECTED_W_SHA else f"(EXPECTED {EXPECTED_W_SHA})")


def check_sigma():
    import scipy.io as sio
    from sigma_est import estimate_sigma_map, load_vdn
    dev = get_device()
    mat = VENDOR / "VI-Non-IID" / "test_data" / "overthrust" / "overthrust_204.mat"
    if not mat.exists():
        sys.exit("run first: python scripts/fetch_vendor.py VI-Non-IID")
    x = sio.loadmat(mat)["data"].astype(np.float32)[30:158, 300:428]     # 128x128 patch
    x = x / np.abs(x).max()                                              # scale to [-1, 1]
    net = load_vdn(dev)
    rng = np.random.default_rng(0)
    print(f"device={dev}  patch std={x.std():.3f}")
    print(f"{'true':>7} {'median':>8} {'min':>8} {'max':>8} {'rel.err':>8}")
    for s in (0.05, 0.10, 0.1819):
        y = x + s * rng.standard_normal(x.shape).astype(np.float32)
        sm = estimate_sigma_map(net, y, dev)
        med = float(np.median(sm))
        print(f"{s:7.4f} {med:8.4f} {sm.min():8.4f} {sm.max():8.4f} {abs(med - s) / s:8.1%}")


def check_gauss(n=64, tau=0.3, sigma_y=0.1819, size=128, seed=0):
    import torch
    dev = get_device()
    use_vendor("IPS-RNA")
    from models import general_anneal_Langevin_dynamics_den as authors
    L, s1, sL, eps, M = 500, 27.0, 0.01, 3.3e-6, 5                       # paper settings (Marmousi model)
    sig = np.exp(np.linspace(np.log(s1), np.log(sL), L)).astype(np.float32)
    sig_t = torch.tensor(sig, device=dev)
    i0 = int(np.abs(sig - sigma_y).argmin())
    s_ln = float(sig[i0])

    def score(x, lab):      # exact score of N(0, tau^2 + sigma^2) data at noise level sigma[lab]
        return -x / (tau ** 2 + sig_t[lab].view(-1, 1, 1, 1) ** 2)

    g = torch.Generator().manual_seed(seed)
    x_true = tau * torch.randn(1, 1, size, size, generator=g)
    y = (x_true + sigma_y * torch.randn(1, 1, size, size, generator=g)).to(dev)
    yb = y.repeat(n, 1, 1, 1)

    def run_authors(batch, final_only=True):
        return authors(batch.reshape(batch.shape[0], 1, -1), torch.rand_like(batch), score, sig,
                       n_steps_each=M, step_lr=eps, final_only=final_only, verbose=False,
                       denoise=True, c_begin=0, sigma_0=sigma_y)

    def run_paper(batch):   # literal Algorithm 2; scale i0 itself is skipped (1/(s_ln^2 - s_i0^2) is singular)
        x = batch.clone()
        for i in range(i0 + 1, L):
            lab = torch.full((batch.shape[0],), i, device=dev, dtype=torch.long)
            a = eps * float(sig[i]) ** 2 / sL ** 2
            for _ in range(M):
                gr = score(x, lab) + (y - x) / (s_ln ** 2 - float(sig[i]) ** 2)
                x = x + a * gr + (2 * a) ** 0.5 * torch.randn_like(x)
        return x + sL ** 2 * score(x, torch.full((batch.shape[0],), L - 1, device=dev, dtype=torch.long))

    def target(s_ref):      # exact law at scale L, then the final denoising step: (mean factor, variance)
        prec = 1 / (tau ** 2 + sL ** 2) + 1 / (s_ref ** 2 - sL ** 2)
        c = tau ** 2 / (tau ** 2 + sL ** 2)
        return c * (1 / (s_ref ** 2 - sL ** 2)) / prec, c * c / prec

    def stats(xs):
        xs = xs.to(dev).float().reshape(-1, 1, size, size)
        return float((xs.mean(0) * y[0]).sum() / (y[0] ** 2).sum()), float(xs.var(0).mean()), bool(torch.isfinite(xs).all())

    t = time.time(); torch.manual_seed(seed)
    out, _ = run_authors(yb)
    a_stats = stats(out[0]); ta = time.time() - t
    t = time.time(); torch.manual_seed(seed + 1)
    b_stats = stats(run_paper(yb)); tb = time.time() - t
    wien = (tau ** 2 / (tau ** 2 + sigma_y ** 2), tau ** 2 * sigma_y ** 2 / (tau ** 2 + sigma_y ** 2))

    print(f"\ndevice={dev}  sigma_y={sigma_y}  start index Ln={i0} (sigma_Ln={s_ln:.6f})  n={n} samples")
    print(f"{'':36}{'shrink':>9}{'target':>9}{'var':>10}{'target':>10}  finite")
    for name, st, tg in (("A authors' code (sigma_y in 1/(..))", a_stats, target(sigma_y)),
                         ("B paper-literal (sigma_Ln, Ln+1..)", b_stats, target(s_ln))):
        print(f"{name:36}{st[0]:9.4f}{tg[0]:9.4f}{st[1]:10.5f}{tg[1]:10.5f}  {st[2]}")
    print(f"{'exact Wiener posterior (sigma_y)':36}{'':9}{wien[0]:9.4f}{'':10}{wien[1]:10.5f}")
    print(f"timing: A {ta:.1f}s, B {tb:.1f}s")

    torch.manual_seed(seed)
    imgs = run_authors(y, final_only=False)
    ys = y.cpu()
    ks = [0, 1, 2, 4, 5, 9, 24, 99, len(imgs) - 1]
    print("authors' code, rms|x-y| after inner step k (k=len-1 is after the final denoise):")
    print("  " + "  ".join(f"{k + 1}:{float((imgs[k] - ys).pow(2).mean().sqrt()):.3f}" for k in ks))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("env", check_env), ("sigma", check_sigma), ("gauss", check_gauss)):
        if what in (name, "all"):
            print(f"\n===== {name} =====")
            fn()
