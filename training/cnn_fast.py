"""GPU-resident training of detector.phaseogram_cnn.FileModel with the paper's recipe (two augmented views, BCE with
pos_weight + W_CONS * MSE consistency, Adam 1e-3, weight decay 1e-4, batch 24, 40 epochs, torch Generator seeded
with detector.phaseogram_cnn.SEED), without the per-batch host-side gathering of detector.phaseogram_cnn._train_fold:

  * all phaseograms (clean + K signal-augmented copies) and the preprocessed features live on the device (or in
    pinned host memory, copied per batch, when they do not fit) and a batch is an index gather;
  * a file holds up to K_NOTES notes as a padded block with a mask; the note embeddings are pooled by mean and
    standard deviation exactly as FileModel.forward (std = 0 for a one-note file), vectorised over the batch;
  * the model is FileModel itself (same parameters, same state_dict), only its forward is evaluated in vectorised
    form: head(cat[mean, std, features]).

Signal-domain augmentation: with probability p_aug a training file is replaced by one of its room copies -- the
same copy for the phaseograms and the features.  The view keeps the phase roll, the row mask and the peak
normalisation (no image-domain channel emulation), as training/notes_cnn._view_signal_aug.
Not bit-identical to detector.phaseogram_cnn._train_fold (different device, gather order and RNG use), equivalent recipe.
"""
from __future__ import annotations
import numpy as np


def _pool(e, mask):
    """e: (B, K, D) note embeddings, mask: (B, K) float -> (B, 2D) [mean, std] over the valid notes; std = 0 when a
    file has one note (FileModel's single-note branch).  sqrt is clamped so a zero variance has a finite gradient."""
    import torch
    n = mask.sum(1, keepdim=True).clamp_min(1.0)
    mean = (e * mask[..., None]).sum(1) / n
    var = (((e - mean[:, None]) ** 2) * mask[..., None]).sum(1) / n
    std = torch.sqrt(var.clamp_min(1e-12)) * (n > 1).float()
    return torch.cat([mean, std], 1)


def _forward(model, x, mask, ana):
    """x: (B, K, C, P) phaseograms, already viewed; mask (B, K); ana (B, F) or None."""
    import torch
    B, K = x.shape[:2]
    e = model.cnn(x.reshape(B * K, 1, *x.shape[2:])).reshape(B, K, -1)
    z = _pool(e, mask)
    if model.use_analytic:
        z = torch.cat([z, ana], 1)
    return model.head(z).squeeze(1)


def _view(x, mask, gen, train):
    """Phase roll, peak normalization per note and two masks of up to seven rows."""
    import torch
    import detector.phaseogram_cnn as rn
    if train:
        x = torch.roll(x, int(torch.randint(0, rn.P, (1,), generator=gen).item()), dims=-1)
    x = x / x.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    if train:
        x = x.clone()
        for _ in range(2):
            w = int(torch.randint(0, 8, (1,), generator=gen).item())
            if w:
                r0 = int(torch.randint(0, max(1, rn.C - w), (1,), generator=gen).item())
                x[:, :, r0:r0 + w, :] = 0.0
    return x * mask[..., None, None]


class Data:
    """Phaseograms ph (N, K, C, P), note mask (N, K), features X (N, F); optional room copies ph_aug
    (N, A, K, C, P) / X_aug (N, A, F) for training and a test copy ph_test / X_test for scoring."""

    def __init__(self, ph, mask, X, y, ph_aug=None, X_aug=None, ph_test=None, X_test=None, device='cuda',
                 on_device=True):
        import torch
        self.dev = torch.device(device)
        put = (lambda a: torch.as_tensor(np.ascontiguousarray(a)).to(self.dev)) if on_device else \
              (lambda a: torch.as_tensor(np.ascontiguousarray(a)).pin_memory() if self.dev.type == 'cuda'
               else torch.as_tensor(np.ascontiguousarray(a)))
        self.on_device = on_device
        self.ph, self.mask = put(np.asarray(ph, np.float32)), put(np.asarray(mask, np.float32))
        self.y = np.asarray(y, int)
        self.X_raw, self.X_aug_raw, self.X_test_raw = X, X_aug, X_test
        self.ph_aug = put(np.asarray(ph_aug, np.float32)) if ph_aug is not None else None
        self.ph_test = put(np.asarray(ph_test, np.float32)) if ph_test is not None else None
        self._put = put

    def batch(self, idx, copy=None, test=False):
        """phaseograms of files idx; copy[i] = -1 (clean) or the room copy index; test=True -> the test copy."""
        import torch
        ii = torch.as_tensor(idx)
        if test:
            x = self.ph_test.index_select(0, ii.to(self.ph_test.device))
        else:
            x = self.ph.index_select(0, ii.to(self.ph.device))
            if copy is not None and self.ph_aug is not None and (copy >= 0).any():
                sel = np.flatnonzero(copy >= 0)
                xa = self.ph_aug[ii[sel].to(self.ph_aug.device), torch.as_tensor(copy[sel]).to(self.ph_aug.device)]
                x = x.clone()
                x[torch.as_tensor(sel).to(x.device)] = xa
        m = self.mask.index_select(0, ii.to(self.mask.device))
        if not self.on_device:
            x, m = x.to(self.dev, non_blocking=True), m.to(self.dev, non_blocking=True)
        return x, m


def train(data, tr, use_analytic, epochs=40, p_aug=0.5, seed=0, bs=24, lr=1e-3):
    """Returns (model, (imputer, scaler)); tr = training file indices."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import detector.phaseogram_cnn as rn
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    tr = np.asarray(tr)
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(data.X_raw[tr])
    sc = StandardScaler().fit(imp.transform(data.X_raw[tr]))
    prep = lambda A: torch.as_tensor(sc.transform(imp.transform(A)).astype(np.float32)).to(data.dev)
    Xc = prep(data.X_raw)
    Xa = None
    if data.X_aug_raw is not None:
        N, A, Fd = data.X_aug_raw.shape
        Xa = prep(data.X_aug_raw.reshape(N * A, Fd)).reshape(N, A, -1)
    ytr = data.y[tr]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = rn.FileModel(data.X_raw.shape[1], use_analytic=use_analytic).to(data.dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())],
                                                         dtype=torch.float32, device=data.dev))
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    yt = torch.as_tensor(data.y.astype(np.float32), device=data.dev)
    n_aug = 0 if data.ph_aug is None else data.ph_aug.shape[1]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=g).numpy()
        for b in range(0, len(tr), bs):
            idx = tr[perm[b:b + bs]]
            copy = np.full(len(idx), -1)
            if n_aug:
                hit = rng.random(len(idx)) < p_aug
                copy[hit] = rng.integers(n_aug, size=int(hit.sum()))
            x, m = data.batch(idx, copy)
            ii = torch.as_tensor(idx, device=data.dev)
            ana = Xc.index_select(0, ii)
            if Xa is not None and (copy >= 0).any():
                s = np.flatnonzero(copy >= 0)
                ana = ana.clone()
                ana[torch.as_tensor(s, device=data.dev)] = Xa[ii[s], torch.as_tensor(copy[s], device=data.dev)]
            ys = yt.index_select(0, ii)
            l1 = _forward(model, _view(x, m, g, True), m, ana)
            l2 = _forward(model, _view(x, m, g, True), m, ana)
            loss = 0.5 * (lossf(l1, ys) + lossf(l2, ys)) + rn.W_CONS * F.mse_loss(torch.sigmoid(l1), torch.sigmoid(l2))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    model._prep = prep
    return model, (imp, sc)


def predict(model, data, rows, test=False, bs=256):
    import torch
    rows = np.asarray(rows)
    X = data.X_test_raw if test else data.X_raw
    out = []
    with torch.no_grad():
        for b in range(0, len(rows), bs):
            idx = rows[b:b + bs]
            x, m = data.batch(idx, test=test)
            ana = model._prep(X[idx])
            out.append(torch.sigmoid(_forward(model, _view(x, m, None, False), m, ana)).cpu().numpy())
    return np.concatenate(out)
