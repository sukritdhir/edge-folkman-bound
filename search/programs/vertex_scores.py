"""Which vertex should be deleted next under method B?  Subspace estimate for every vertex, exact value for a few.
usage: python3 exp/vertex_scores.py stage.pkl [k]"""
import sys, pickle, time, numpy as np, scipy.linalg as sl
sys.path.insert(0, "exp")
from sublib import *

d = pickle.load(open(sys.argv[1], "rb")); mu = np.array(d["mu"]); D = np.array(d["D"]); X = list(d["X"])
KV = int(sys.argv[2]) if len(sys.argv) > 2 else 128
T = Target(X)
D = D * (3 * (T.mf * mu).sum() / (T.nf * D).sum())
N2, dme = sparse_N2(T, mu, D)
lam0 = lanczos_min(N2, T.ein, 1500)
rng = np.random.default_rng(5)
V = rng.standard_normal((NE, KV))
th, V, res, rm, rd, tim = evaluate(T, mu, D, V, lam0 - GAP0, 4)
th, V, res, rm, rd, tim = evaluate(T, mu, D, V, th[0] - GAP0, 2)
print("|X| = %d: lambda_min(N'') = %.6f (margin %+.6f), theta[k-1] = %.6f" % (len(X), th[0], th[0] + 2 / 3, th[-1])); sys.stdout.flush()
NV = N2 @ V
bad = np.zeros(n, dtype=bool); bad[X] = True
De = np.repeat(D, q); wt = mu[torb]
at = T.tin; ae = T.ein
W = wt[at].sum(); Ds = De[ae].sum()
wl = np.zeros(n); dl = np.zeros(n)
for j in range(3): np.add.at(wl, tverts[at, j], wt[at])
aei = np.nonzero(ae)[0]
for j in range(2): np.add.at(dl, ends[aei, j], De[aei])
theta = 3 * (W - wl) / (Ds - dl)
edges_at = [[] for _ in range(n)]
for p in aei:
    edges_at[ends[p, 0]].append(p); edges_at[ends[p, 1]].append(p)
cand = [x for x in range(n) if Sx.keep[x] and not bad[x]]
est = {}
t0 = time.time()
N2c = N2.tocsr()
for x in cand:
    rows = np.array(edges_at[x])
    Vr = V[rows]; NVr = NV[rows]
    A = Vr.T @ NVr
    Bm = Vr.T @ (N2c[rows][:, rows] @ Vr)
    H = np.diag(th) - A - A.T + Bm
    Gx = np.eye(KV) - Vr.T @ Vr
    lam = sl.eigh((H + H.T) / 2, (Gx + Gx.T) / 2, eigvals_only=True, subset_by_index=[0, 0])[0]
    est[x] = (lam, theta[x], lam / theta[x] + 2 / 3, float((Vr ** 2).sum()))
print("estimates for %d vertices [%.0fs]" % (len(cand), time.time() - t0)); sys.stdout.flush()
order = sorted(cand, key=lambda x: -est[x][2])
greedy = max(cand, key=lambda x: theta[x])
print("budget-greedy vertex %d: theta %.6f, estimated margin %+.6f (rank %d)" % (greedy, theta[greedy], est[greedy][2], order.index(greedy)))
ths = np.array([theta[x] for x in cand]); ms = np.array([est[x][2] for x in cand]); ls = np.array([est[x][0] for x in cand])
print("theta: min %.6f max %.6f; raw subspace lambda after deletion: min %.6f max %.6f; est margin: min %+.6f median %+.6f max %+.6f"
      % (ths.min(), ths.max(), ls.min(), ls.max(), ms.min(), np.median(ms), ms.max()))
pickle.dump({"est": est, "X": X}, open("scan/vertex_scores.pkl", "wb"))


def exact(x):
    T2 = Target(X + [x])
    D2 = D * (3 * (T2.mf * mu).sum() / (T2.nf * D).sum())
    N22, _ = sparse_N2(T2, mu, D2)
    return lanczos_min(N22, T2.ein, 1500) + 2 / 3


for x in order[:5] + [greedy] + order[len(order) // 2: len(order) // 2 + 2] + order[-2:]:
    print("vertex %4d (coset %s): theta %.6f  mass %.4f  est margin %+.6f  exact margin %+.6f" % (x, "-", est[x][1], est[x][3], est[x][2], exact(x))); sys.stdout.flush()
