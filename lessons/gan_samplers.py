"""GAN-based oversamplers for small tabular data (PyTorch), usable inside an imblearn Pipeline.

Modes (cross-checked against the original papers):
  gan            : plain GAN trained on the minority training rows; its output fills the whole gap.
  smotified_gan  : SMOTified-GAN (Sharma, Singh & Chandra, IEEE Access 2022, arXiv 2108.03235). SMOTE makes pre-sampled
                   minority rows u. The generator takes u as its INPUT (not random noise); the discriminator compares the
                   generator output with the REAL minority rows. Generator output G(u) = the new minority rows.
  ganified_smote : GANified-SMOTE (Cheah, Yang & Lee, Int. J. Financial Studies 2023, 11(3):110, Sect. 3.2.5).
                   A GAN first generates minority rows; SMOTE is then applied TO THE GAN-GENERATED rows (they are the
                   seeds that SMOTE interpolates between); GAN rows + SMOTE rows are added to the original training data.
                   gan_fraction = share of the gap produced directly by the GAN (0.5 = the 394 GAN + 394 SMOTE split of their Test B).
                   smote_on="real_plus_gan" is a variant (SMOTE seeds = real + GAN rows), NOT the published version.

Leak-free: everything is learned inside fit_resample() from the rows it receives (= training rows of one fold).
It never sees validation or test rows.
"""
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from torch import nn

torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


class _Generator(nn.Module):
    """3 hidden layers x 128 neurons with BatchNorm + ReLU (as in Sharma et al.), sigmoid output in [0,1]."""

    def __init__(self, d_in, d_out, hidden=128):
        super().__init__()
        layers, last = [], d_in
        for _ in range(3):
            layers += [nn.Linear(last, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
            last = hidden
        layers += [nn.Linear(last, d_out), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _Discriminator(nn.Module):
    """Linear -> LeakyReLU(0.2) -> Linear (two layers, as in Sharma et al.)."""

    def __init__(self, d, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


def _train_gan(real, d_in, pool, epochs, batch_size, lr, seed):
    """Train G and D. If pool is None the generator input is fresh N(0,1) noise; otherwise rows drawn from pool."""
    torch.manual_seed(seed)
    d = real.shape[1]
    G, D = _Generator(d_in, d), _Discriminator(d)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    real_t = torch.tensor(real, dtype=torch.float32)
    pool_t = None if pool is None else torch.tensor(pool, dtype=torch.float32)
    n = len(real_t)
    G.train()
    D.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            if len(idx) < 2:                      # BatchNorm needs at least 2 rows
                continue
            xb = real_t[idx]
            zb = torch.randn(len(idx), d_in) if pool_t is None else pool_t[torch.randint(0, len(pool_t), (len(idx),))]
            fake = G(zb)
            opt_d.zero_grad()                     # discriminator: real -> 1, fake -> 0
            loss_d = bce(D(xb), torch.ones(len(idx), 1)) + bce(D(fake.detach()), torch.zeros(len(idx), 1))
            loss_d.backward()
            opt_d.step()
            opt_g.zero_grad()                     # generator: make the discriminator say 1 for fakes
            loss_g = bce(D(fake), torch.ones(len(idx), 1))
            loss_g.backward()
            opt_g.step()
    G.eval()
    return G


def _generate(G, d_in, pool, n):
    with torch.no_grad():
        if pool is None:
            z = torch.randn(n, d_in)
        else:
            idx = np.random.RandomState(0).randint(0, len(pool), n) if len(pool) != n else np.arange(n)
            z = torch.tensor(pool[idx], dtype=torch.float32)
        return G(z).numpy()


class GANSampler(BaseEstimator):
    def __init__(self, mode="smotified_gan", epochs=200, batch_size=64, lr=2e-4, latent_dim=16,
                 gan_fraction=0.5, k_neighbors=5, smote_on="gan_only", random_state=0):
        self.mode = mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.latent_dim = latent_dim
        self.gan_fraction = gan_fraction
        self.k_neighbors = k_neighbors
        self.smote_on = smote_on
        self.random_state = random_state

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        minority, n_min, n_maj = classes[counts.argmin()], counts.min(), counts.max()
        need = int(n_maj - n_min)
        if need <= 0:
            return X, y
        scaler = MinMaxScaler().fit(X)                     # learned from these (training) rows only
        Xs = scaler.transform(X)
        Xmin = Xs[y == minority]
        k = max(1, min(self.k_neighbors, n_min - 1))

        if self.mode == "gan":
            G = _train_gan(Xmin, self.latent_dim, None, self.epochs, self.batch_size, self.lr, self.random_state)
            synth = _generate(G, self.latent_dim, None, need)

        elif self.mode == "smotified_gan":
            Xr, _ = SMOTE(k_neighbors=k, random_state=self.random_state).fit_resample(Xs, y)
            u = Xr[len(Xs):]                               # SMOTE's pre-sampled minority rows (appended after originals)
            G = _train_gan(Xmin, Xs.shape[1], u, self.epochs, self.batch_size, self.lr, self.random_state)
            synth = _generate(G, Xs.shape[1], u, len(u))

        elif self.mode == "ganified_smote":
            n_gan = min(max(int(round(need * self.gan_fraction)), 2), need)
            G = _train_gan(Xmin, self.latent_dim, None, self.epochs, self.batch_size, self.lr, self.random_state)
            gan_rows = _generate(G, self.latent_dim, None, n_gan)
            if n_gan >= need:
                synth = gan_rows
            elif self.smote_on == "gan_only":              # published version: SMOTE interpolates between GAN rows
                maj_label = classes[counts.argmax()]
                k2 = max(1, min(self.k_neighbors, n_gan - 1))
                X_maj = Xs[y != minority][: k2 + 1]        # SMOTE needs the other class present; it never interpolates it
                X_tmp = np.vstack([gan_rows, X_maj])
                y_tmp = np.concatenate([np.full(n_gan, minority), np.full(len(X_maj), maj_label)])
                X_res, _ = SMOTE(sampling_strategy={minority: need}, k_neighbors=k2,
                                 random_state=self.random_state).fit_resample(X_tmp, y_tmp)
                synth = np.vstack([gan_rows, X_res[len(X_tmp):]])
            else:                                          # variant: SMOTE seeds = real + GAN rows
                X_aug = np.vstack([Xs, gan_rows])
                y_aug = np.concatenate([y, np.full(len(gan_rows), minority)])
                X_res, _ = SMOTE(k_neighbors=k, random_state=self.random_state).fit_resample(X_aug, y_aug)
                synth = X_res[len(Xs):]
        else:
            raise ValueError(f"unknown mode {self.mode}")

        synth = np.clip(synth, 0.0, 1.0)
        self.report_ = {"n_synthetic": len(synth),
                        "std_ratio_vs_real": float(np.mean(synth.std(0) / (Xmin.std(0) + 1e-9)))}
        X_new = scaler.inverse_transform(synth)
        return np.vstack([X, X_new]), np.concatenate([y, np.full(len(X_new), minority)])
