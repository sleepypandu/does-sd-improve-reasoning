import numpy as np
import pytest
from safetensors.numpy import save_file

from gradprobe.analyze import report
from gradprobe.stats import cosine, gram, noise_scale, pooled_cosine

SHAPES = {"embed": (13, 6), "layer.0.w": (5, 7), "layer.0.b": (7,), "norm": (3,), "scale": ()}


def _gradients(n, seed=0):
    rng = np.random.default_rng(seed)
    shared = {k: np.asarray(rng.normal(size=s), dtype=np.float32) for k, s in SHAPES.items()}
    return [{k: np.asarray(0.5 * v + rng.normal(size=v.shape), dtype=np.float32) for k, v in shared.items()}
            for _ in range(n)]


def _flat(g):
    return np.concatenate([g[k].astype(np.float64).reshape(-1) for k in sorted(g)])


def _write_single(tmp_path, grads):
    paths = []
    for i, g in enumerate(grads):
        d = tmp_path / f"single-seed-{i}"
        d.mkdir()
        save_file(g, str(d / "gradient.safetensors"))
        paths.append(str(d))
    return paths


def _write_sharded(tmp_path, grads, ranks=3):
    # Like FSDP: every rank holds a different, same-named piece of each tensor.
    paths = []
    for i, g in enumerate(grads):
        d = tmp_path / f"sharded-seed-{i}"
        d.mkdir()
        flat = {k: v.reshape(-1) for k, v in g.items()}
        for r in range(ranks):
            save_file({k: np.array_split(v, ranks)[r] for k, v in flat.items() if np.array_split(v, ranks)[r].size},
                      str(d / f"gradient_rank{r}.safetensors"))
        paths.append(str(d))
    return paths


def _dense(grads):
    X = np.stack([_flat(g) for g in grads])
    return X @ X.T, X


@pytest.mark.parametrize("chunk", [1, 4, 1 << 22])
def test_gram_matches_dense(tmp_path, chunk):
    grads = _gradients(5)
    G_dense, _ = _dense(grads)
    np.testing.assert_allclose(gram(_write_single(tmp_path, grads), chunk=chunk), G_dense, rtol=1e-12)


def test_sharded_matches_single(tmp_path):
    grads = _gradients(4)
    single = gram(_write_single(tmp_path, grads), chunk=5)
    sharded = gram(_write_sharded(tmp_path, grads), chunk=5)
    np.testing.assert_allclose(sharded, single, rtol=1e-12)


def test_parts_sum_to_whole(tmp_path):
    paths = _write_single(tmp_path, _gradients(4))
    whole = gram(paths)
    np.testing.assert_allclose(sum(gram(paths, part=k, nparts=3) for k in range(3)), whole, rtol=1e-12)


def test_layout_mismatch(tmp_path):
    a, b = _gradients(2)
    del b["norm"]
    with pytest.raises(ValueError):
        gram(_write_single(tmp_path, [a, b]))


def _cos(u, v):
    return u @ v / np.linalg.norm(u) / np.linalg.norm(v)


def test_statistics_match_dense():
    grads = _gradients(8, seed=1)
    G, X = _dense(grads)
    ref, method = [0, 1, 2, 3], [4, 5, 6, 7]
    c, se, reps = pooled_cosine(G, ref, method)
    assert c == pytest.approx(_cos(X[ref].mean(0), X[method].mean(0)), rel=1e-12)
    dense_reps = [_cos(np.delete(X[ref], s, 0).mean(0), np.delete(X[method], s, 0).mean(0)) for s in range(4)]
    np.testing.assert_allclose(reps, dense_reps, rtol=1e-12)
    assert se == pytest.approx(np.sqrt(3 / 4 * ((np.array(dense_reps) - np.mean(dense_reps)) ** 2).sum()), rel=1e-12)
    m = X[ref].mean(0)
    trace = ((X[ref] - m) ** 2).sum() / 3
    assert noise_scale(G, ref) == pytest.approx(trace / (m @ m), rel=1e-12)
    assert trace == pytest.approx(np.trace(np.cov(X[ref], rowvar=False)), rel=1e-12)
    assert cosine(G, ref, ref) == pytest.approx(1.0)


def test_report_pairs_by_seed():
    grads = _gradients(4, seed=2)
    G, _ = _dense(grads)
    labels = ["reference", "m", "reference", "m"]
    rows = report(G, labels, [0, 1, 1, 0])
    c, se, _ = pooled_cosine(G, [0, 2], [3, 1])
    assert rows["m"]["cosine"] == pytest.approx(c) and rows["m"]["jackknife_se"] == pytest.approx(se)
    with pytest.raises(ValueError):
        report(G[:3, :3], labels[:3], [0, 1, 1])
