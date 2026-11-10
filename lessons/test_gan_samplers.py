"""Self-test for gan_samplers.py (about 1 minute). Run:  python test_gan_samplers.py
Every line must say PASS. Uses synthetic data; no files needed."""
import numpy as np
from sklearn.datasets import make_classification
from gan_samplers import GANSampler

X, y = make_classification(n_samples=1000, n_features=9, n_informative=4, weights=[0.61], flip_y=0.3, random_state=0)
X[:, 1] = X[:, 1] * 8000 + 20000                       # one big-scale column, like Solids
need = int(np.bincount(y).max() - np.bincount(y).min())
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name)


for mode in ["gan", "smotified_gan", "ganified_smote"]:
    s = GANSampler(mode, epochs=20, random_state=1)
    Xo, yo = s.fit_resample(X, y)
    syn = Xo[len(X):]
    check(f"{mode}: classes balanced", np.bincount(yo)[0] == np.bincount(yo)[1])
    check(f"{mode}: exactly {need} synthetic rows, all labelled minority", len(syn) == need and (yo[len(y):] == 1).all())
    check(f"{mode}: original rows unchanged and first", np.array_equal(Xo[:len(X)], X) and np.array_equal(yo[:len(y)], y))
    check(f"{mode}: synthetic values inside the real range", (syn >= X.min(0) - 1e-6).all() and (syn <= X.max(0) + 1e-6).all())
    check(f"{mode}: no NaN", not np.isnan(Xo).any())
    check(f"{mode}: same seed gives same output",
          np.allclose(Xo, GANSampler(mode, epochs=20, random_state=1).fit_resample(X, y)[0]))

# GANified-SMOTE (published version): the SMOTE rows must lie between GAN rows (inside their per-feature range)
s = GANSampler("ganified_smote", epochs=20, gan_fraction=0.5, random_state=2)
Xo, yo = s.fit_resample(X, y)
syn = Xo[len(X):]
n_gan = int(round(need * 0.5))
gan_part, smote_part = syn[:n_gan], syn[n_gan:]
check("ganified_smote: SMOTE rows lie inside the range of the GAN rows",
      (smote_part >= gan_part.min(0) - 1e-6).all() and (smote_part <= gan_part.max(0) + 1e-6).all())
check("ganified_smote: split is ~half GAN / half SMOTE", abs(len(gan_part) - len(smote_part)) <= 1)

# test rows must never influence the sampler: output depends only on the rows it is given
Xa, ya = GANSampler("smotified_gan", epochs=10, random_state=3).fit_resample(X[:600], y[:600])
Xb, yb = GANSampler("smotified_gan", epochs=10, random_state=3).fit_resample(np.vstack([X[:600]]), y[:600])
check("sampler output depends only on the rows passed in", np.allclose(Xa, Xb))
print("\nALL TESTS PASSED" if ok_all else "\nSOME TESTS FAILED - send me this output")
