#!/usr/bin/env python3
"""paradigm_space.py -- figures and interactive map of the experimental paradigm space.

One input: the literature workbook. One row of the 'Articles' sheet is one
experimental paradigm, so a paper running four experiments occupies four rows
and four points. Rung ladders, decision rules and axis names are read from the
'Axis Fla criteria' sheet, so the rubric and the figures cannot drift apart.

    python3 paradigm_space.py literature_database_scored_v3.xlsx -o out

To run this script with the corersponding option of clustering you need to 
update the excel source (the one updated is the version in the files_3 folder), 
    files_3/literature_database_scored_v5.xlsx

and run: 
    python3 paradigm_space.py literature_database_scored_v5.xlsx -o figs_k3 -k 3
where we can change the number of clusters and also the folder where to save it. 


Written to <out>/:
    fig1_projections.pdf   the corpus in the six principal planes
    fig2_regions.pdf       candidate regions, conditionally projected, plus funnels
    fig3_cube.pdf          3D views whose three axes carry every constraint
    fig4_task_axes.pdf     motor plane, non-motor plane, displacement between them
    fig5_gaps.pdf          coverage deficit, frontier and island gaps
    fig6_accounts.pdf      the h layer, the pushforward, f at the thesis point
    fig7_audit.pdf         dispositions, confidence, collinearity, label determinism
    figA1_ladders.pdf      occupancy of every rung of every ladder
    figA2_year.pdf         principal axes against publication year
    paradigm_space.html    interactive map: move the walls of the box yourself, and,
                           when -k is given, the clusters the code found next to the
                           labels assigned by hand
    paradigm_scores.csv    the tidy table every figure is drawn from
    scoring_report.txt     every number quoted in the prose
    captions.tex           figure environments with those numbers already filled in

Requires numpy, pandas, matplotlib, openpyxl. plotly is optional: if it is
installed its javascript bundle is inlined so the page works offline, otherwise
the page loads plotly from the CDN.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

PRINCIPAL = ["x", "y", "z", "t"]
ALL_AXES = ["x", "y", "z", "t", "s", "x1", "y1", "r"]

# how each axis is found in the 'Articles' header (prefix match, case folded)
COLUMN_PREFIX = {
    "x": "x axis", "y": "y axis", "z": "z axis", "t": "tr axis",
    "s": "ts axis", "x1": "x_1 axis", "y1": "y_1 axis", "r": "r axis",
}
# extra event blocks, used only with --per-event
EVENT_BLOCKS = [
    {"z": "z1 axis", "t": "tr1 axis", "s": "ts1 axis"},
    {"z": "z2 axis", "t": "tr2 axis", "s": "ts2 axis"},
]

# panel labels are deliberately shorter than the criteria-sheet names, which are
# long enough to collide between neighbouring panels
FALLBACK_LABEL = {
    "x": "motor difficulty $x$",
    "y": "motor timescale $y$",
    "z": "surprise hierarchy $z$",
    "t": "task relevance $t$",
    "s": "surprise magnitude $s$",
    "x1": "non-motor difficulty $x_1$",
    "y1": "non-motor timescale $y_1$",
    "r": "endo/exo $r$",
}
SHORT = {"x": "x", "y": "y", "z": "z", "t": "t", "s": "s",
         "x1": "x1", "y1": "y1", "r": "r"}

WEIGHT = {"hi": 1.0, "md": 0.7, "lo": 0.4}
SIGMA = 0.09          # density bandwidth, reported not fitted
SIGMA_REACH = 0.18    # nearest-neighbour scale of the reachability kernel
EMPTY_DEFICIT = 0.95  # a cell counts as uncovered above this coverage deficit
GRID = 13             # cells per axis in the 4D gap search

CLUSTERS = {
    "motor":    dict(label="motor control", color="#157F7F"),
    "surprise": dict(label="environmental surprise",       color="#C1425A"),
    "bridge":   dict(label="surprise during action",    color="#D2892A"),
    "thesis":   dict(label="thesis experiments",        color="#4B2E83"),
}
CLUSTER_OF_FILE = {
    "saliency": "surprise",
    "unexpected + motor control": "bridge",
    "thesis": "thesis",
}
CONF_MARKER = {"hi": "o", "md": "s", "lo": "^"}

# candidate regions. Order of the constraints is the order of the funnel.
REGIONS = {
    "G1": dict(
        title="engaging control interrupted by an irrelevant event",
        constraints=[("x", ">=", 0.67), ("y", ">=", 0.57), ("t", "<=", 0.17)],
        color="#C1425A"),
    "G2": dict(
        title="the same, with a surprise horizon deeper than a fixed-p deviant",
        constraints=[("x", ">=", 0.67), ("t", "<=", 0.17), ("z", ">=", 0.5)],
        color="#6A3D9A"),
}

ACCOUNTS = ["PC", "AIF", "OFC", "SDT", "DDM", "AC", "SAL", "RL"]
ACCOUNT_ALIAS = {
    "salience detection": "SAL", "saliency": "SAL", "salience": "SAL",
    "affordance competition": "AC", "predictive coding": "PC",
    "active inference": "AIF", "optimal feedback control": "OFC",
    "drift diffusion": "DDM", "reinforcement learning": "RL",
    "statistical decision theory": "SDT",
}

THESIS_FLAGS = ("thesis",)  # matched against 'Thematic file' / 'Entry source'


@dataclass
class Config:
    sigma: float = SIGMA
    sigma_reach: float = SIGMA_REACH
    grid: int = GRID
    drop_lo: bool = False
    per_event: bool = False
    axes: list = field(default_factory=lambda: list(PRINCIPAL))


# ---------------------------------------------------------------------------
# 1. loading
# ---------------------------------------------------------------------------

def _find_column(columns, prefix):
    p = prefix.lower()
    for c in columns:
        if str(c).strip().lower().startswith(p):
            return c
    return None


def _clean(s):
    return str(s).strip() if pd.notna(s) else ""


_ARITHMETIC = re.compile(r"^=[\d\s.+\-*/()]+$")


def _numeric(col):
    """Numbers, plus the arithmetic-only formulas the workbook holds in the axis cells.

    Several hand-scored rows record a mean over event blocks as '=(0.33+0.17*2+0.5)/4'.
    Any tool that rewrites the file without recalculating drops the cached value and
    pandas then reads the cell as missing, which silently removes those paradigms from
    every coverage statistic. Formulas that reference other cells stay missing.
    """
    v = pd.to_numeric(col, errors="coerce")
    for i in col.index[v.isna() & col.notna()]:
        s = str(col[i]).strip()
        if _ARITHMETIC.match(s):
            try:
                v[i] = float(eval(s[1:], {"__builtins__": {}}, {}))
            except Exception:
                pass
    return v


def load_ladders(xlsx, sheet="Axis Fla criteria"):
    """Rung ladders and axis names, read straight from the criteria sheet.

    Returns {axis_key: {'name': str, 'rungs': [(value, label, test), ...]}}.
    """
    try:
        raw = pd.read_excel(xlsx, sheet_name=sheet, header=None)
    except Exception:
        return {}
    keys = {"x": "x", "y": "y", "z": "z", "t": "t", "s": "s",
            "x_1": "x1", "y_1": "y1", "r": "r"}
    ladders, current = {}, None
    for _, row in raw.iterrows():
        a, b, c = (_clean(row.get(0)), _clean(row.get(1)), _clean(row.get(2)))
        head = re.match(r"^([a-z]_?1?)\s{1,3}(.+)$", a)
        if head and head.group(1) in keys and not b and not c:
            current = keys[head.group(1)]
            ladders[current] = {"name": head.group(2).strip(), "rungs": []}
            continue
        if current and a and a.lower() != "value":
            try:
                v = float(a)
            except ValueError:
                continue
            ladders[current]["rungs"].append((v, b, c))
    return ladders


def load_rules(xlsx, sheet="Axis Fla criteria"):
    """The R1-R8 decision rules, for the html and the report."""
    try:
        raw = pd.read_excel(xlsx, sheet_name=sheet, header=None)
    except Exception:
        return []
    out = []
    for _, row in raw.iterrows():
        a, b = _clean(row.get(0)), _clean(row.get(1))
        m = re.match(r"^(R\d)\s+(.*)$", a)
        if m and b:
            out.append((m.group(1), m.group(2), b))
    return out


def load_corpus(xlsx, cfg: Config, sheet="Articles"):
    """The 'Articles' sheet as a tidy frame, one row per paradigm.

    Every column the figures use is derived here, so nothing downstream has to
    know about the workbook's column names.
    """
    df = pd.read_excel(xlsx, sheet_name=sheet)
    cols = {k: _find_column(df.columns, p) for k, p in COLUMN_PREFIX.items()}
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        raise SystemExit(f"axis columns not found in '{sheet}': {missing}")

    out = pd.DataFrame(index=df.index)
    out["citekey"] = df.get("CiteKey", pd.Series(dtype=object)).map(_clean)
    out["title"] = df.get("Title", pd.Series(dtype=object)).map(_clean)
    out["year"] = pd.to_numeric(df.get("Year"), errors="coerce")
    out["topic"] = df.get("Topic", pd.Series(dtype=object)).map(_clean)
    out["file"] = df.get("Thematic file", pd.Series(dtype=object)).map(_clean)
    out["entry"] = df.get("Entry source", pd.Series(dtype=object)).map(_clean)
    out["study"] = df.get("Study type", pd.Series(dtype=object)).map(_clean)
    out["summary"] = df.get("One-line summary", pd.Series(dtype=object)).map(_clean)
    out["task_note"] = df.get("Note on task", pd.Series(dtype=object)).map(_clean)
    out["event_note"] = df.get("Note on the surprise axis criteria",
                               pd.Series(dtype=object)).map(_clean)
    out["scoring_note"] = df.get("scoring note", pd.Series(dtype=object)).map(_clean)
    out["provenance"] = df.get("score provenance", pd.Series(dtype=object)).map(_clean)
    out["confidence"] = (df.get("confidence", pd.Series(dtype=object))
                         .map(_clean).str.lower().replace("", np.nan))
    out["scorable"] = (df.get("scorable", pd.Series(dtype=object))
                       .map(_clean).str.lower())
    for k, c in cols.items():
        out[k] = _numeric(df[c])

    # computational account. The workbook carries a distribution over accounts in
    # P_PC ... P_RL; where it is missing, fall back to the dominant label, hand
    # column first. A distribution beats an argmax: with one-hot labels the kernel
    # regression below collapses onto whichever account the two or three nearest
    # rows happen to carry.
    # reindexed on df, not taken as-is: an absent column returns an empty Series,
    # and zip() over it would silently produce zero accounts for every row
    blank = pd.Series("", index=df.index, dtype=object)
    acc = df.get("Com Account (dominant)", blank).reindex(df.index).map(_clean)
    fallback = df.get("dominant_account", blank).reindex(df.index).map(_clean)
    out["account"] = [_normalise_account(a) or _normalise_account(b)
                      for a, b in zip(acc, fallback)]
    for a in ACCOUNTS:
        col = f"P_{a}"
        out[f"p_{a}"] = (pd.to_numeric(df[col], errors="coerce")
                         if col in df.columns else np.nan)

    out["row"] = df.index + 2  # spreadsheet row, for the audit
    out = out[out["scorable"] == "yes"].copy()

    if cfg.per_event:
        out = _expand_events(df, out)

    # one identifier per paradigm: a paper with four experiments gets four
    out["paradigm_id"] = _unique_ids(out)
    out["cluster"] = [_cluster_of(f, t) for f, t in zip(out["file"], out["topic"])]
    thesis = out["file"].str.lower().isin(THESIS_FLAGS) | \
        out["entry"].str.lower().isin(THESIS_FLAGS)
    out.loc[thesis, "cluster"] = "thesis"
    out["thesis"] = thesis
    out["confidence"] = out["confidence"].fillna("lo")
    out["w"] = out["confidence"].map(WEIGHT).fillna(WEIGHT["lo"])
    if cfg.drop_lo:
        out = out[out["confidence"] != "lo"].copy()

    out["off_lattice"] = _off_lattice(out)
    return out.reset_index(drop=True)


def _expand_events(df, out):
    """--per-event: a second and third manipulated event become their own points."""
    extra = []
    for block in EVENT_BLOCKS:
        cols = {k: _find_column(df.columns, p) for k, p in block.items()}
        if not all(cols.values()):
            continue
        sub = out.copy()
        for k, c in cols.items():
            sub[k] = pd.to_numeric(df.loc[out.index, c], errors="coerce")
        sub = sub[sub["z"].notna()]
        if len(sub):
            sub["event_note"] = sub["event_note"] + " [second event block]"
            extra.append(sub)
    return pd.concat([out] + extra) if extra else out


def _unique_ids(df):
    ids, seen = [], {}
    for k, r in zip(df["citekey"], df["row"]):
        base = k or f"row{r}"
        seen[base] = seen.get(base, 0) + 1
        ids.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
    return ids


def _cluster_of(file_, topic):
    key = file_.strip().lower()
    if key in CLUSTER_OF_FILE:
        return CLUSTER_OF_FILE[key]
    if "salien" in key or "salien" in topic.lower():
        return "surprise"
    return "motor"


def _normalise_account(v):
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.upper() in ACCOUNTS:
        return s.upper()
    return ACCOUNT_ALIAS.get(s.lower(), "")


def _off_lattice(df, tol=1e-3):
    """Rule R7: every score should sit on a rung. Flag the ones that do not."""
    lat6 = np.array([i / 6 for i in range(7)])
    lat7 = np.array([i / 7 for i in range(8)])
    lattice = {"x": lat6, "z": lat6, "t": lat6, "s": lat6, "x1": lat6,
               "y": lat7, "y1": lat7, "r": np.array([0, 0.5, 1.0])}
    flag = pd.Series(False, index=df.index)
    for a, grid in lattice.items():
        v = df[a].to_numpy(float)
        d = np.abs(v[:, None] - grid[None, :]).min(axis=1)
        flag |= pd.Series((d > 0.02) & np.isfinite(v), index=df.index)
    return flag


# ---------------------------------------------------------------------------
# 2. geometry
# ---------------------------------------------------------------------------

def matrix(df, axes):
    """Points and weights for the rows scored on every axis in `axes`."""
    sub = df.dropna(subset=axes)
    return sub[axes].to_numpy(float), sub["w"].to_numpy(float), sub


def density(points, weights, axes_idx, grids, sigma):
    """Exact marginal of the Gaussian mixture onto the axes in `axes_idx`.

    A marginal of a mixture of isotropic Gaussians is a mixture of Gaussians,
    so every projection below is exact rather than histogrammed.
    """
    mesh = np.meshgrid(*grids, indexing="ij")
    out = np.zeros(mesh[0].shape)
    for p, w in zip(points, weights):
        q = np.zeros(mesh[0].shape)
        for k, g in zip(axes_idx, mesh):
            q += (g - p[k]) ** 2
        out += w * np.exp(-q / (2 * sigma ** 2))
    return out / weights.sum()


def reachability(u, points, sigma_r):
    """exp(-d^2/2 sigma_r^2) to the nearest existing paradigm."""
    u = np.atleast_2d(np.asarray(u, float))
    d2 = ((u[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    j = d2.argmin(axis=1)
    near = d2[np.arange(len(u)), j]
    return np.exp(-near / (2 * sigma_r ** 2)), np.sqrt(near), j


def feasible(u, axes):
    """The structural constraints of appendix A.1, evaluated on a grid."""
    idx = {a: i for i, a in enumerate(axes)}
    ok = np.ones(len(u), bool)
    if "x" in idx and "y" in idx:
        ok &= u[:, idx["y"]] <= u[:, idx["x"]] + 1 / 3 + 1e-9
        ok &= u[:, idx["x"]] <= u[:, idx["y"]] + 0.5 + 1e-9
    if "z" in idx and "t" in idx:
        ok &= ~((u[:, idx["z"]] < 0.05) & (u[:, idx["t"]] > 0.05))
    return ok


def gap_search(df, cfg, k=2, separation=0.25):
    """Frontier gaps (empty but a short walk away) and island gaps (empty and far).

    Searched separately, as in appendix A.5: taking arg max of the coverage
    deficit alone returns the far corner of the cube, which is empty and
    uninformative.
    """
    axes = cfg.axes
    P, w, _ = matrix(df, axes)
    lin = [np.linspace(0, 1, cfg.grid)] * len(axes)
    cells = np.stack(np.meshgrid(*lin, indexing="ij"), -1).reshape(-1, len(axes))
    keep = feasible(cells, axes)
    cells = cells[keep]

    p = np.zeros(len(cells))
    for pt, wt in zip(P, w):
        p += wt * np.exp(-((cells - pt) ** 2).sum(1) / (2 * cfg.sigma ** 2))
    deficit = 1 - p / p.max()
    reach, dist, near = reachability(cells, P, cfg.sigma_reach)

    step = 1.0 / (cfg.grid - 1)
    # a cell whose nearest paradigm is inside it is not empty, however low the
    # smoothed density there: deficit saturates near 1 over most of a 4D cube.
    empty = (deficit >= EMPTY_DEFICIT) & (dist > 0.5 * step)
    out = []
    for kind, order in (("frontier", np.argsort(-reach)), ("island", np.argsort(reach))):
        chosen = []
        for i in order:
            if not empty[i]:
                continue
            u = cells[i]
            if any(np.linalg.norm(u - c) < separation for c in chosen):
                continue
            chosen.append(u)
            out.append(dict(kind=kind, reach=float(reach[i]),
                            deficit=float(deficit[i]),
                            **{a: float(v) for a, v in zip(axes, u)}))
            if len(chosen) == k:
                break
    return pd.DataFrame(out), dict(cells=cells, deficit=deficit, reach=reach)


def account_matrix(sub):
    """Rows as distributions over ACCOUNTS, and a mask of the rows that carry one."""
    cols = [f"p_{a}" for a in ACCOUNTS]
    Y = sub[cols].to_numpy(float) if set(cols) <= set(sub.columns) \
        else np.full((len(sub), len(ACCOUNTS)), np.nan)
    Y = np.nan_to_num(Y)
    for i, a in enumerate(sub["account"]):
        if Y[i].sum() <= 0 and a in ACCOUNTS:      # fall back to the dominant label
            Y[i, ACCOUNTS.index(a)] = 1.0
    keep = Y.sum(1) > 0
    Y[keep] /= Y[keep].sum(1, keepdims=True)
    return Y, keep


def account_field(df, u, axes=None, sigma=SIGMA, exclude_thesis=True):
    """f(u) = P(account | design), Nadaraya-Watson onto the account simplex."""
    axes = axes or PRINCIPAL
    sub = df.dropna(subset=axes)
    if exclude_thesis:
        sub = sub[~sub["thesis"]]
    Y, keep = account_matrix(sub)
    sub, Y = sub[keep], Y[keep]
    if not len(sub):
        return {a: 0.0 for a in ACCOUNTS}, 0.0
    P = sub[axes].to_numpy(float)
    w = sub["w"].to_numpy(float)
    kern = w * np.exp(-((np.asarray(u, float) - P) ** 2).sum(1) / (2 * sigma ** 2))
    if kern.sum() <= 0:
        return {a: 0.0 for a in ACCOUNTS}, 0.0
    v = kern @ Y / kern.sum()
    # how many rows the estimate actually rests on: with a narrow kernel over a
    # lattice this can be two, and a confident-looking f is then one paper's label
    n_eff = float(kern.sum() ** 2 / (kern ** 2).sum())
    return {a: float(vi) for a, vi in zip(ACCOUNTS, v)}, n_eff


def pushforward(df):
    """Which accounts the corpus actually spends its mass on."""
    Y, keep = account_matrix(df)
    sub, Y = df[keep], Y[keep]
    if not len(sub):
        return {a: 0.0 for a in ACCOUNTS}
    w = sub["w"].to_numpy(float)
    m = (w[:, None] * Y).sum(0)
    m = m / m.sum() if m.sum() else m
    return {a: float(v) for a, v in zip(ACCOUNTS, m)}


# ---------------------------------------------------------------------------
# 2b. clustering: do our cluters map to the tentative claimed literature?
# ---------------------------------------------------------------------------
#
# Each paper came from a search from a specific literature. 
# Everything here is numpy so the script keeps its dependency list; scipy is used
# for Ward (the hierarchical clustering)
# Some metrics are computed here to test the statistical significance
# of the clusters identified from the paradigm space. 
########## 
# To formally prove the bridge cluster is statistically 
# distinct from both motor and surprise across all dimensions, 
# we can add a multivariate analysis of variance (MANOVA) 
# or pairwise distance permutation test.
from statsmodels.multivariate.manova import MANOVA
from scipy.spatial.distance import cdist

from statsmodels.multivariate.manova import MANOVA

def test_cluster_separation(sub_df, axes, group_col="cluster"):
    """Evaluates MANOVA and pairwise multivariate separation."""
    clean_df = sub_df.dropna(subset=axes).copy()
    lines = []

    # ---------------------------------------------------------
    # 1. Multivariate Analysis of Variance (MANOVA)
    # ---------------------------------------------------------
    lines.append("\n=== MANOVA Test (Overall Cluster Separation) ===")
    formula = f"{' + '.join(axes)} ~ {group_col}"
    try:
        maov = MANOVA.from_formula(formula, data=clean_df)
        manova_summary = str(maov.mv_test())
        lines.append(manova_summary)
        print("\n=== MANOVA Test (Overall Cluster Separation) ===")
        print(manova_summary)
    except Exception as e:
        err_msg = f"MANOVA could not be computed: {e}"
        lines.append(err_msg)
        print(err_msg)

    # ---------------------------------------------------------
    # 2. Pairwise Centroid Distances & Permutation p-values
    # ---------------------------------------------------------
    lines.append("\n=== Pairwise Centroid Distances & Permutation Tests ===")
    print("\n=== Pairwise Centroid Distances & Permutation Tests ===")
    
    groups = [g for g in clean_df[group_col].unique() if pd.notna(g)]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = clean_df[clean_df[group_col] == groups[i]][axes].to_numpy(float)
            g2 = clean_df[clean_df[group_col] == groups[j]][axes].to_numpy(float)
            
            if len(g1) < 2 or len(g2) < 2:
                continue
                
            obs_dist = np.linalg.norm(g1.mean(0) - g2.mean(0))
            
            # Permutation test on distance between centroids
            pooled = np.vstack([g1, g2])
            n1 = len(g1)
            null_dists = []
            for _ in range(1000):
                np.random.shuffle(pooled)
                null_dists.append(np.linalg.norm(pooled[:n1].mean(0) - pooled[n1:].mean(0)))
                
            p_val = (np.array(null_dists) >= obs_dist).mean()
            line = f"{groups[i]:<14} vs {groups[j]:<14} | Centroid Distance: {obs_dist:.3f} | p = {p_val:.4f}"
            lines.append(line)
            print(line)

    return "\n".join(lines) + "\n"

def _kmeanspp(X, w, k, rng):
    idx = [rng.choice(len(X), p=w / w.sum())]
    for _ in range(k - 1):
        d2 = ((X - X[idx][:, None, :]) ** 2).sum(-1).min(0)
        p = w * d2
        idx.append(rng.choice(len(X)) if p.sum() <= 0 else rng.choice(len(X), p=p / p.sum()))
    return X[idx].copy()


def weighted_kmeans(X, w, k, n_init=40, iters=200, seed=0):
    """Lloyd's algorithm with confidence weights and k-means++ starts.

    Weighted because a row inferred from a title should not pull a centroid as hard
    as one whose Methods section was read — the same weights the density uses.
    """
    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    for _ in range(n_init):
        C = _kmeanspp(X, w, k, rng)
        lab = np.zeros(len(X), int)
        for _ in range(iters):
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            new_lab = d2.argmin(1)
            if (new_lab == lab).all():
                break
            lab = new_lab
            for j in range(k):
                m = lab == j
                if m.any():
                    C[j] = (w[m, None] * X[m]).sum(0) / w[m].sum()
                else:                       # empty cluster: reseed on the worst point
                    C[j] = X[(w * d2.min(1)).argmax()]
        inertia = float((w * ((X - C[lab]) ** 2).sum(1)).sum())
        if inertia < best[0]:
            best = (inertia, lab.copy(), C.copy())
    return best[1], best[2], best[0]


def ward_labels(X, k):
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
    except Exception:
        return None
    Z= linkage(X, method="ward")
    return fcluster(Z, k, criterion="maxclust") - 1, Z


def silhouette(X, lab):
    """Mean silhouette. 
    Silhouette Score: Evaluates whether clusters occupy distinct, 
    separated regions in the multidimensional space 
    (ranges [−1,1], with >0.35 indicating genuine geometric separation).

    Ties on a lattice give zero-distance neighbours, so this is
    a conservative read: identical designs in different clusters are penalised."""
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    out = np.zeros(len(X))
    for i in range(len(X)):
        same = lab == lab[i]
        if same.sum() <= 1:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, lab == j].mean() for j in np.unique(lab) if j != lab[i])
        out[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(out.mean())


def contingency(a, b):
    ua, ub = np.unique(a), np.unique(b)
    M = np.zeros((len(ua), len(ub)), int)
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            M[i, j] = int(((a == x) & (b == y)).sum())
    return M, ua, ub


def adjusted_rand(a, b):
    M, _, _ = contingency(a, b)
    n = M.sum()
    if n < 2:
        return float("nan")
    comb2 = lambda v: (v * (v - 1) / 2).sum()
    idx = comb2(M.astype(float))
    ea, eb = comb2(M.sum(1).astype(float)), comb2(M.sum(0).astype(float))
    exp = ea * eb / (n * (n - 1) / 2)
    mx = (ea + eb) / 2
    return float((idx - exp) / (mx - exp)) if mx != exp else 1.0


def adjusted_mutual_info(a, b):
    from math import lgamma
    M, _, _ = contingency(a, b)
    n = M.sum()
    ai, bj = M.sum(1), M.sum(0)
    nz = M > 0
    P = M[nz] / n
    mi = float((P * np.log(n * M[nz] / np.outer(ai, bj)[nz])).sum())
    ent = lambda v: float(-((v[v > 0] / n) * np.log(v[v > 0] / n)).sum())
    ha, hb = ent(ai), ent(bj)
    lf = lambda m: lgamma(m + 1)
    emi = 0.0
    for i, A in enumerate(ai):
        for j, B in enumerate(bj):
            for m in range(max(1, A + B - n), min(A, B) + 1):
                term = (m / n) * np.log(n * m / (A * B))
                logw = (lf(A) + lf(B) + lf(n - A) + lf(n - B) - lf(n) - lf(m)
                        - lf(A - m) - lf(B - m) - lf(n - A - B + m))
                emi += term * np.exp(logw)
    denom = (ha + hb) / 2 - emi
    return float((mi - emi) / denom) if denom != 0 else 1.0


def best_agreement(found, given):
    """Hungarian-free best one-to-one match: k is small, so try every assignment."""
    from itertools import permutations
    M, ua, ub = contingency(found, given)
    n, best, mapping = M.sum(), -1, None
    rows, cols = M.shape
    for perm in permutations(range(cols), min(rows, cols)):
        hit = sum(M[i, perm[i]] for i in range(len(perm)))
        if hit > best:
            best, mapping = hit, {ua[i]: ub[perm[i]] for i in range(len(perm))}
    return best / n, mapping, M, ua, ub


def permutation_p(found, given, stat=adjusted_rand, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = stat(found, given)
    null = np.array([stat(found, rng.permutation(given)) for _ in range(n_perm)])
    return obs, float((null >= obs).sum() + 1) / (n_perm + 1), null


def bootstrap_stability(X, w, k, n_boot=60, seed=0):
    """
    Bootstrap Cluster Stability: 
    This fucntion Resamples the dataset with replacement 60 times. 
    A score near 1.0 (>0.85) proves the clusters are robust and 
    not artifacts of a few specific papers.
    
    How often two paradigms land together across resamples.

    A partition that only exists in this particular sample will score near the rate
    you would get by chance; one that reflects real structure will not.
    """
    rng = np.random.default_rng(seed)
    co = np.zeros((len(X), len(X)))
    seen = np.zeros((len(X), len(X)))
    for b in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        uniq = np.unique(idx)
        if len(uniq) <= k:
            continue
        lab, _, _ = weighted_kmeans(X[uniq], w[uniq], k, n_init=6, seed=int(seed + b))
        same = (lab[:, None] == lab[None, :]).astype(float)
        co[np.ix_(uniq, uniq)] += same
        seen[np.ix_(uniq, uniq)] += 1
    with np.errstate(invalid="ignore"):
        frac = np.where(seen > 0, co / np.maximum(seen, 1), np.nan)
    iu = np.triu_indices(len(X), 1)
    v = frac[iu]
    v = v[np.isfinite(v)]
    # 1 = every pair always agrees, 0.5 = coin flip
    return float(np.mean(np.maximum(v, 1 - v)))

def cluster_corpus(df, cfg, k=2, method="ward", 
                   axes=None, sweep=range(2, 9), out=None):
    """Cluster the corpus without looking at the labels, then compare to them."""
    axes = axes or cfg.axes
    
    # 1. Fill missing values on secondary axes so rows aren't dropped prematurely
    sub = df.dropna(subset=["x", "y", "z", "t"]).copy()
    for col in ["s", "x1", "y1", "r"]:
        if col in axes and col in sub.columns:
            sub[col] = sub[col].fillna(0.0)
            
    sub = sub.dropna(subset=axes).reset_index(drop=True)
    X_raw = sub[axes].to_numpy(float)

    print("\n=== Clustering axis statistics  (Raw) ===")
    print(sub[axes].describe().T)
    print("\nVariance:")
    print(sub[axes].var())
    print("\nStandard deviation:")
    print(sub[axes].std())

    # 2. Standardize features to unit variance so t does not overpower s, x1, y1
    std_devs = np.std(X_raw, axis=0)
    std_devs[std_devs == 0] = 1.0  # prevent division by zero
    X = (X_raw - np.mean(X_raw, axis=0)) / std_devs

    print("\n=== Clustering axis statistics (Standardized) ===")
    print(pd.DataFrame(X, columns=axes).describe().T)

    w = sub["w"].to_numpy(float)

    if method == "ward":
        res = ward_labels(X, k)
        if res is None:
            print("scipy not available, falling back to k-means")
            method = "kmeans"
        else:
            lab, Z = res
            from scipy.cluster.hierarchy import dendrogram
            fig = plt.figure(figsize=(12, 5))
            dendrogram(
                Z,
                labels=sub["paradigm_id"].values if "paradigm_id" in sub else None,
                leaf_rotation=90,
                leaf_font_size=6
            )
            plt.title(f"Hierarchical Clustering Dendrogram (Ward's Method, k={k})")
            plt.xlabel("Paradigm Identifier")
            plt.ylabel("Ward Linkage Distance")
            plt.tight_layout()
            if out:
                fig.savefig(Path(out) / "fig9_dendrogram.pdf", dpi=300)
            plt.close(fig)

    if method == "kmeans":
        lab, C, _ = weighted_kmeans(X, w, k, seed=0)
    else:
        C = np.stack([(w[lab == j, None] * X[lab == j]).sum(0) / w[lab == j].sum()
                      for j in np.unique(lab)])

    # Order clusters by motor difficulty (x)
    order = np.argsort(C[:, axes.index("x")] if "x" in axes else C[:, 0])
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[v] for v in lab])
    C = C[order]

    # We compute the Adjusted Rand Index (ARI) with Permutation Testing:
    # Measures whether the discovered geometric clusters match your curated 
    # classes (motor, salience, bridge, thesis) better than pure chance
    # GEtting p-value (p_ari) <0.05 proves the alignment between empirical 
    # geometry and literature categories is statistically significant.
    given = sub["cluster"].to_numpy()
    agree, mapping, M, ua, ub = best_agreement(lab, given)
    ari, p_ari, _ = permutation_p(lab, given)

    
    curve = []
    for kk in sweep:
        if kk >= len(X):
            break
        l2, _, inertia = weighted_kmeans(X, w, kk, n_init=12, seed=1)
        curve.append(dict(k=kk, silhouette=silhouette(X, l2), inertia=inertia,
                          stability=bootstrap_stability(X, w, kk, n_boot=30, seed=2)))
                          
    return dict(sub=sub, X=X, w=w, axes=axes, k=k, method=method, labels=lab,
                centers=C, given=given, contingency=M, found_names=ua, given_names=ub,
                agreement=agree, mapping=mapping, ari=ari, p_ari=p_ari,
                ami=adjusted_mutual_info(lab, given),
                silhouette=silhouette(X, lab),
                stability=bootstrap_stability(X, w, k, n_boot=60, seed=2),
                curve=pd.DataFrame(curve))


# ---------------------------------------------------------------------------
# 3. regions
# ---------------------------------------------------------------------------

def satisfies(df, constraints):
    m = pd.Series(True, index=df.index)
    for a, op, v in constraints:
        col = df[a]
        m &= (col >= v) if op == ">=" else (col <= v)
    return m.fillna(False)


def funnel(df, constraints, axes=None):
    """How the corpus empties as the constraints are applied one at a time."""
    axes = axes or PRINCIPAL
    sub = df.dropna(subset=axes)
    steps = [("all scored", len(sub))]
    for i in range(len(constraints)):
        m = satisfies(sub, constraints[:i + 1])
        a, op, v = constraints[i]
        steps.append((f"{a} {'≥' if op == '>=' else '≤'} {v:g}", int(m.sum())))
    return steps


def region_members(df, name, axes=None):
    axes = axes or PRINCIPAL
    sub = df.dropna(subset=axes)
    return sub[satisfies(sub, REGIONS[name]["constraints"])]


def rect_on_plane(constraints, a, b):
    """The shadow of a region on the (a, b) plane, and the constraints it drops."""
    span = {}
    off = []
    for ax, op, v in constraints:
        if ax in (a, b):
            lo, hi = span.get(ax, (0.0, 1.0))
            span[ax] = (max(lo, v), hi) if op == ">=" else (lo, min(hi, v))
        else:
            off.append((ax, op, v))
    xa = span.get(a, (0.0, 1.0))
    xb = span.get(b, (0.0, 1.0))
    return xa, xb, off


def centroid(name, axes=None):
    """Centre of the region's box, clipped to the unit cube."""
    axes = axes or PRINCIPAL
    lo = dict.fromkeys(axes, 0.0)
    hi = dict.fromkeys(axes, 1.0)
    for a, op, v in REGIONS[name]["constraints"]:
        if a in lo:
            lo[a] = max(lo[a], v) if op == ">=" else lo[a]
            hi[a] = min(hi[a], v) if op == "<=" else hi[a]
    return np.array([(lo[a] + hi[a]) / 2 for a in axes])


# ---------------------------------------------------------------------------
# 4. figures
# ---------------------------------------------------------------------------

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.patches import Rectangle, Patch          # noqa: E402
from matplotlib.lines import Line2D                      # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

INK = "#101820"
RULE = "#DCE3E6"
GREY = "#AEB8BC"
PANEL_W = 1.95   # inches per square panel; every panel in the set is identical


def use_style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300,
        "font.size": 7.4, "axes.titlesize": 8.0, "axes.labelsize": 7.6,
        "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 7.0,
        "axes.edgecolor": INK, "axes.linewidth": 0.6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def label_of(axis, ladders=None, short=False, ladder=False):
    """short: the symbol alone. ladder: the criteria-sheet name, used where the
    panel is wide enough for it. Otherwise the short panel label above."""
    sym = SHORT[axis].replace("x1", "x_1").replace("y1", "y_1")
    if short:
        return f"${sym}$"
    if ladder and ladders and ladders.get(axis, {}).get("name"):
        name = re.sub(r"\s*\(.*?\)\s*$", "", ladders[axis]["name"]).strip()
        return f"{name} ${sym}$"
    return FALLBACK_LABEL[axis]


def square(ax, xlab, ylab, rungs=None):
    """Every panel in every figure gets the same box, the same limits, the same grid."""
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.set_box_aspect(1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "", ".5", "", "1"])
    ax.set_yticklabels(["0", "", ".5", "", "1"])
    if rungs:
        for v in rungs[0]:
            ax.axvline(v, color=RULE, lw=0.4, zorder=0)
        for v in rungs[1]:
            ax.axhline(v, color=RULE, lw=0.4, zorder=0)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)


def panel_title(ax, text, sub=None, width=30, fontsize=7.2, pad=4.0, tail=None):
    """A left-aligned panel title that cannot run into the panel next door.

    `loc="left"` is what makes long titles collide: matplotlib anchors the string
    at the left spine and lets it grow rightwards over its neighbour, because it
    has no idea the neighbour is there. Wrapping at a width the panel can hold is
    the only fix that survives a change of figure size, so the width is given in
    characters and each figure picks it from its own panel width.

    `tail` is appended to the last wrapped line if it fits there — used for the
    axis symbol, which reads badly stranded on a line of its own. `sub` is a
    smaller grey second line for counts, which are secondary to the name.
    """
    lines = textwrap.wrap(text, width=width) or [text]
    if tail:
        # "$x_1$" is five characters of source and about two of ink
        drawn = len(re.sub(r"[$_{}\\]", "", tail))
        if len(lines[-1]) + drawn + 1 <= width + 3:
            lines[-1] = f"{lines[-1]} {tail}"
        else:
            lines.append(tail)
    if sub:
        ax.text(0, 1.008, sub, transform=ax.transAxes, fontsize=fontsize - 1.1,
                color="#6C7C85", va="bottom", ha="left")
        pad = pad + fontsize + 1.5
    ax.set_title("\n".join(lines), loc="left", fontsize=fontsize, pad=pad,
                 linespacing=1.20)


def spread(a, b, radius=0.013):
    """Fan coincident paradigms out on a small spiral.

    The axes are ladders, so exact ties are the rule rather than the exception:
    without this a rung holding nine paradigms and a rung holding one look the
    same. The offset is deterministic and always smaller than half a rung.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    out_a, out_b = a.copy(), b.copy()
    key = {}
    for i, (u, v) in enumerate(zip(a, b)):
        key.setdefault((round(u, 4), round(v, 4)), []).append(i)
    golden = np.pi * (3 - np.sqrt(5))
    for idx in key.values():
        n = len(idx)
        if n == 1:
            continue
        for j, i in enumerate(idx):
            rad = radius * np.sqrt((j + 0.5) / n) * np.sqrt(n)
            ang = j * golden
            out_a[i] = a[i] + rad * np.cos(ang)
            out_b[i] = b[i] + rad * np.sin(ang)
    return out_a, out_b


def scatter_corpus(ax, df, a, b, live=None, size=17, zorder=3, jitter=True):
    """Cluster colour, confidence marker; rows failing `live` are drawn as context."""
    if jitter:
        xa, xb = spread(df[a].to_numpy(float), df[b].to_numpy(float))
    else:
        xa, xb = df[a].to_numpy(float), df[b].to_numpy(float)
    live = np.ones(len(df), bool) if live is None else np.asarray(live, bool)
    for conf, marker in CONF_MARKER.items():
        for cl, style in CLUSTERS.items():
            m = (df["confidence"].to_numpy() == conf) & \
                (df["cluster"].to_numpy() == cl)
            if not m.any():
                continue
            on = m & live
            off = m & ~live
            if off.any():
                ax.scatter(xa[off], xb[off], s=size * 0.55, marker=marker,
                           facecolors="none", edgecolors=GREY, linewidths=0.5,
                           alpha=0.55, zorder=zorder - 1)
            if on.any():
                ax.scatter(xa[on], xb[on], s=size, marker=marker,
                           color=style["color"], edgecolors="white",
                           linewidths=0.4, alpha=0.92, zorder=zorder)


def corpus_legend(fig, ncol=4, y=-0.02, context=False, confidence=True):
    handles = [Line2D([], [], marker="o", ls="", color=s["color"],
                      markeredgecolor="white", markersize=5, label=s["label"])
               for s in CLUSTERS.values()]
    if confidence:   # 3D scatter draws one marker per call, so shape carries nothing there
        handles += [Line2D([], [], marker=m, ls="", color=INK, markersize=4.4,
                           label=f"confidence {c}") for c, m in CONF_MARKER.items()]
    if context:
        handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="none",
                              markeredgecolor=GREY, markersize=4.4,
                              label="excluded by an off-plane constraint"))
    fig.legend(handles=handles, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, y), handletextpad=0.3, columnspacing=1.2)


# --- figure 1: the corpus in projection -----------------------------------

def fig_projections(df, ladders, out):
    """Six planes, six identical square panels, no region boxes.

    The old version of this figure drew the candidate regions here, which is
    what made it read as emptier than the data: a box in the (x, y) plane is a
    slab in the space, and points that a third constraint excludes still fall
    inside its shadow. Region membership now has a figure of its own.
    """
    planes = [("x", "y"), ("x", "z"), ("x", "t"),
              ("y", "z"), ("y", "t"), ("z", "t")]
    sub = df.dropna(subset=PRINCIPAL)
    fig, axes = plt.subplots(2, 3, figsize=(3 * PANEL_W + 1.1, 2 * PANEL_W + 1.0))
    for ax, (a, b) in zip(axes.ravel(), planes):
        rungs = ([v for v, *_ in ladders.get(a, {}).get("rungs", [])],
                 [v for v, *_ in ladders.get(b, {}).get("rungs", [])])
        square(ax, label_of(a, ladders), label_of(b, ladders), rungs)
        scatter_corpus(ax, sub, a, b)
        r = np.corrcoef(sub[a], sub[b])[0, 1]
        ax.text(0.035, 0.965, f"$r$ = {r:+.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.4, color="#54646D",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
    fig.suptitle(f"the corpus in the six principal planes  ·  n = {len(sub)} paradigms",
                 fontsize=8, y=1.005, color="#54646D")
    corpus_legend(fig, ncol=4, y=-0.035)
    save(fig, out, "fig1_projections")
    return len(sub)


# --- figure 2: the candidate regions, tested ------------------------------

def fig_regions(df, ladders, out):
    """One row per region: two conditional planes and the constraint funnel.

    A panel shows only the paradigms that satisfy the constraints on the axes
    *not* drawn, so what is inside the box on screen is inside the region in
    the space. The funnel then reports the same thing as a count.
    """
    sub = df.dropna(subset=PRINCIPAL)
    panels = {"G1": [("x", "y"), ("x", "t")], "G2": [("x", "z"), ("z", "t")]}
    names = list(REGIONS)
    fig = plt.figure(figsize=(3 * PANEL_W + 1.5, len(names) * PANEL_W + 1.2))
    gs = fig.add_gridspec(len(names), 3, width_ratios=[1, 1, 1.35],
                          wspace=0.45, hspace=0.62)
    counts = {}
    for i, name in enumerate(names):
        spec = REGIONS[name]
        members = satisfies(sub, spec["constraints"])
        counts[name] = int(members.sum())
        for j, (a, b) in enumerate(panels[name]):
            ax = fig.add_subplot(gs[i, j])
            (xlo, xhi), (ylo, yhi), off = rect_on_plane(spec["constraints"], a, b)
            live = satisfies(sub, off) if off else pd.Series(True, index=sub.index)
            rungs = ([v for v, *_ in ladders.get(a, {}).get("rungs", [])],
                     [v for v, *_ in ladders.get(b, {}).get("rungs", [])])
            square(ax, label_of(a, ladders, short=True),
                   label_of(b, ladders, short=True), rungs)
            inside = int((members & live).sum()) if off else counts[name]
            empty = inside == 0
            ax.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                                   facecolor=spec["color"],
                                   alpha=0.08 if empty else 0.16,
                                   edgecolor=spec["color"], lw=0.9,
                                   ls="--" if empty else "-", zorder=1))
            scatter_corpus(ax, sub, a, b, live=live)
            ax.text((xlo + xhi) / 2, yhi - 0.055, f"{inside}",
                    ha="center", va="top", fontsize=9, color=spec["color"],
                    fontweight="bold", zorder=6)
            cond = ", ".join(f"${ax_}$ {'≥' if op == '>=' else '≤'} {v:g}"
                             for ax_, op, v in off)
            ax.set_title(f"among designs with {cond}" if off
                         else "no constraint off this plane",
                         fontsize=6.5, color="#54646D", pad=3)

        ax = fig.add_subplot(gs[i, 2])
        steps = funnel(sub, spec["constraints"])
        labels = [s for s, _ in steps]
        vals = [v for _, v in steps]
        pos = np.arange(len(vals))[::-1]
        shades = [matplotlib.colors.to_rgba(spec["color"], a)
                  for a in np.linspace(0.30, 0.95, len(vals))]
        ax.barh(pos, vals, height=0.62, color=shades, zorder=2)
        for p, v, s in zip(pos, vals, labels):
            ax.text(v + max(vals) * 0.02, p, f"{v}", va="center",
                    fontsize=6.8, color=INK)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels, fontsize=6.6)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.set_xlabel("paradigms surviving")
        ax.set_title("\n".join(textwrap.wrap(f"{name}  ·  {spec['title']}", 44)),
                     fontsize=7.2, color=spec["color"], loc="left", pad=4)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    corpus_legend(fig, ncol=4, y=-0.06, context=True)
    save(fig, out, "fig2_regions")
    return counts


# --- figure 3: the cube ---------------------------------------------------

def _box3d(ax, bounds, color, alpha=0.10):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
             [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    ax.add_collection3d(Poly3DCollection([v[f] for f in faces], facecolor=color,
                                         alpha=alpha, edgecolor=color, lw=0.7,
                                         zsort="min"))


def fig_cube(df, ladders, out):
    """Two 3D views, each carrying every constraint of the region it shows.

    Choosing the triple this way is the whole point: in (x, y, t) the G1 box is
    the region, not its shadow, so a marker inside the box is a paradigm inside
    the region and the picture cannot mislead.
    """
    sub = df.dropna(subset=PRINCIPAL)
    views = [("G1", ("x", "y", "t"), 22, -58), ("G2", ("x", "z", "t"), 22, -58)]
    fig = plt.figure(figsize=(7.0, 3.5))
    for i, (name, (a, b, c), elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        spec = REGIONS[name]
        lo = {k: 0.0 for k in (a, b, c)}
        hi = {k: 1.0 for k in (a, b, c)}
        for k, op, v in spec["constraints"]:
            if op == ">=":
                lo[k] = v
            else:
                hi[k] = v
        _box3d(ax, [(lo[a], hi[a]), (lo[b], hi[b]), (lo[c], hi[c])], spec["color"])
        inside = satisfies(sub, spec["constraints"]).to_numpy()
        pa, pb = spread(sub[a].to_numpy(float), sub[b].to_numpy(float))
        pc = sub[c].to_numpy(float)
        colors = np.array([CLUSTERS[c_]["color"] for c_ in sub["cluster"]])
        # shadow on the floor: three dimensions on a flat page need a depth cue
        ax.scatter(pa, pb, np.zeros_like(pc), s=5, color=GREY, alpha=0.35,
                   depthshade=False, zorder=1)
        ax.scatter(pa[~inside], pb[~inside], pc[~inside], s=15,
                   c=colors[~inside], edgecolors="white", linewidths=0.3,
                   alpha=0.9, depthshade=False)
        if inside.any():
            ax.scatter(pa[inside], pb[inside], pc[inside], s=42, c=colors[inside],
                       edgecolors=INK, linewidths=0.9, depthshade=False)
        for k, ax_name in zip((a, b, c), ("x", "y", "z")):
            getattr(ax, f"set_{ax_name}lim")(0, 1)
            getattr(ax, f"set_{ax_name}ticks")([0, 0.5, 1])
            getattr(ax, f"set_{ax_name}ticklabels")(["0", ".5", "1"])
        ax.set_xlabel(label_of(a, short=True), labelpad=-4)
        ax.set_ylabel(label_of(b, short=True), labelpad=-4)
        ax.set_zlabel(label_of(c, short=True), labelpad=-4)
        ax.tick_params(pad=0.6, labelsize=6.0)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{'AB'[i]}  ({a}, {b}, {c}) — {name}: "
                     f"{int(inside.sum())} of {len(sub)} inside",
                     fontsize=7.4, loc="left", pad=-4)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor("white")
            pane.set_edgecolor(RULE)
    corpus_legend(fig, ncol=4, y=0.02, confidence=False)
    save(fig, out, "fig3_cube")


# --- figure 0: what the space is -----------------------------------------

def _arrow3(ax, start, end, color=INK, lw=1.1):
    ax.plot(*zip(start, end), color=color, lw=lw, solid_capstyle="round", zorder=2)
    ax.scatter(*[[e] for e in end], s=12, color=color, marker="o", depthshade=False)


def fig_space(df, ladders, out, axes=("x", "y", "t")):
    """What the coordinates mean, and where the corpus sits in them.

    A gives each ladder its own column so the rung labels cannot collide, with the
    occupancy of every rung drawn as a bar to the left of the spine: the reader sees
    the definition and the coverage of an axis in one object. B places the corpus in
    three dimensions, with all three marginals cast onto the back walls, so the shape
    of each projection is visible without hunting for it in another figure.
    """
    a, b, c = axes
    sub = df.dropna(subset=PRINCIPAL)
    fig = plt.figure(figsize=(7.4, 4.1))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 3.05], wspace=0.06,
                          left=0.035, right=0.985, bottom=0.10, top=0.90)

    # ---- A: the ladders, one column each
    for i, k in enumerate((a, b, c)):
        ax = fig.add_subplot(gs[0, i])
        rungs = ladders.get(k, {}).get("rungs", [])
        vals = sub[k].dropna()
        counts = [int((np.abs(vals - v) < 0.03).sum()) for v, *_ in rungs] or [0]
        top = max(counts + [1])
        ax.plot([0, 0], [-0.02, 1.02], color=RULE, lw=1.2, solid_capstyle="round", zorder=1)
        for slab, spec in REGIONS.items():
            for kk, op, v in spec["constraints"]:
                if kk != k:
                    continue
                lo, hi = (v, 1.02) if op == ">=" else (-0.02, v)
                ax.add_patch(Rectangle((-0.055, lo), 0.11, hi - lo, facecolor=spec["color"],
                                       alpha=0.15, lw=0, zorder=0))
                ax.text(-0.075, (lo + hi) / 2, slab, fontsize=5.6, rotation=90,
                        ha="center", va="center", color=spec["color"])
        for (v, lab, _), n in zip(rungs, counts):
            if n:                                   # occupancy bar, drawn inward
                ax.add_patch(Rectangle((0, v - 0.019), 0.26 * n / top, 0.038,
                                       facecolor=CLUSTERS["thesis"]["color"], alpha=0.28,
                                       lw=0, zorder=2))
            ax.plot([-0.035, 0.035], [v, v], color=INK, lw=0.85, zorder=3,
                    solid_capstyle="butt")
            txt = re.sub(r"\s*/.*$", "", str(lab)).strip()
            txt = textwrap.shorten(txt, width=25, placeholder="…")
            ax.text(0.33, v + 0.016, f"{v:g}", fontsize=5.4, va="bottom", color=INK)
            ax.text(0.33, v - 0.016, txt, fontsize=5.2, va="top", color="#6C7C85")
            if n:
                ax.text(0.26 * n / top + 0.025, v, str(n), fontsize=4.8, va="center",
                        ha="left", color=CLUSTERS["thesis"]["color"])
        ax.set_xlim(-0.15, 1.60)
        ax.set_ylim(-0.075, 1.10)
        ax.axis("off")
        name = re.sub(r"\s*\(.*?\)\s*$", "", ladders.get(k, {}).get("name", "")).strip()
        ax.text(0, 1.13, label_of(k, short=True), fontsize=10, ha="center", color=INK)
        ax.text(0.28, 1.135, textwrap.shorten(name or FALLBACK_LABEL[k], 27, placeholder="…"),
                fontsize=5.6, ha="left", va="center", color="#6C7C85")
        if i == 0:
            ax.text(-0.15, 1.215, "A   the rung ladders, and how many paradigms sit on "
                    "each rung", fontsize=7.6, ha="left", color=INK)

    # ---- B: the corpus in three dimensions, with its marginals on the walls
    ax3 = fig.add_subplot(gs[0, 3], projection="3d")
    pa, pb = spread(sub[a].to_numpy(float), sub[b].to_numpy(float))
    pc = sub[c].to_numpy(float)
    colors = np.array([CLUSTERS[cl]["color"] for cl in sub["cluster"]])
    wall = dict(s=7, color=GREY, alpha=0.35, depthshade=False, zorder=1)
    ax3.scatter(pa, pb, np.zeros_like(pc), **wall)          # floor: (x, y)
    ax3.scatter(pa, np.ones_like(pb), pc, **wall)           # back wall: (x, t)
    ax3.scatter(np.zeros_like(pa), pb, pc, **wall)          # side wall: (y, t)
    for name, spec in REGIONS.items():
        if not all(k in (a, b, c) for k, _, _ in spec["constraints"]):
            continue
        lo = {k: 0.0 for k in (a, b, c)}
        hi = {k: 1.0 for k in (a, b, c)}
        for k, op, v in spec["constraints"]:
            lo[k] = max(lo[k], v) if op == ">=" else lo[k]
            hi[k] = min(hi[k], v) if op == "<=" else hi[k]
        _box3d(ax3, [(lo[a], hi[a]), (lo[b], hi[b]), (lo[c], hi[c])], spec["color"], 0.16)
        inside = int(satisfies(sub, spec["constraints"]).sum())
        ax3.text((lo[a] + hi[a]) / 2, (lo[b] + hi[b]) / 2, hi[c] + 0.08,
                 f"{name}: {inside}", fontsize=7.2, color=spec["color"], ha="center",
                 fontweight="bold", zorder=8)
    for cl, style in CLUSTERS.items():
        m = (sub["cluster"] == cl).to_numpy()
        if m.any():
            ax3.scatter(pa[m], pb[m], pc[m], s=17, color=style["color"], alpha=0.95,
                        edgecolors="white", linewidths=0.35, depthshade=False, zorder=5)
    for k, nm in zip((a, b, c), ("x", "y", "z")):
        getattr(ax3, f"set_{nm}lim")(0, 1)
        getattr(ax3, f"set_{nm}ticks")([0, 0.5, 1])
        getattr(ax3, f"set_{nm}ticklabels")(["0", ".5", "1"])
    ax3.set_xlabel(label_of(a), labelpad=-2, fontsize=6.6)
    ax3.set_ylabel(label_of(b), labelpad=-2, fontsize=6.6)
    ax3.set_zlabel(label_of(c), labelpad=-2, fontsize=6.6)
    ax3.tick_params(pad=0.4, labelsize=6.0)
    ax3.set_box_aspect((1, 1, 1))
    ax3.view_init(elev=20, azim=-58)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor(RULE)
    ax3.text2D(0.0, 1.02, f"B   the corpus on ({a}, {b}, {c}); grey points are the same "
               f"{len(sub)} paradigms\n      cast onto each wall",
               transform=ax3.transAxes, fontsize=7.6, va="bottom", color=INK)
    corpus_legend(fig, ncol=4, y=0.005, confidence=False)
    save(fig, out, "fig0_space")


# --- figure 4: motor vs non-motor ----------------------------------------

def fig_task_axes(df, ladders, out):
    sub = df.dropna(subset=["x", "y", "x1", "y1"])
    fig, axs = plt.subplots(1, 3, figsize=(3 * PANEL_W + 1.5, PANEL_W + 1.0))
    square(axs[0], label_of("x", ladders), label_of("y", ladders))
    scatter_corpus(axs[0], sub, "x", "y")
    axs[0].set_title("A  motor plane", loc="left", fontsize=7.6)
    square(axs[1], label_of("x1", ladders), label_of("y1", ladders))
    scatter_corpus(axs[1], sub, "x1", "y1")
    axs[1].set_title("B  non-motor plane", loc="left", fontsize=7.6)

    square(axs[2], "difficulty", "timescale")
    d = np.hypot(sub["x"] - sub["x1"], sub["y"] - sub["y1"])
    for (_, row), col in zip(sub.iterrows(), [CLUSTERS[c]["color"] for c in sub["cluster"]]):
        axs[2].annotate("", xy=(row["x1"], row["y1"]), xytext=(row["x"], row["y"]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=0.55,
                                        alpha=0.6, shrinkA=0, shrinkB=0,
                                        mutation_scale=5))
    axs[2].set_title("C  motor → non-motor displacement", loc="left", fontsize=7.6)
    axs[2].text(0.03, 0.96, f"mean displacement {d.mean():.2f}",
                transform=axs[2].transAxes, fontsize=6.6, va="top", color="#54646D")
    corpus_legend(fig, ncol=4, y=-0.10)
    save(fig, out, "fig4_task_axes")
    return float(d.mean())


# --- figure 5: coverage deficit and gaps ---------------------------------

def fig_gaps(df, cfg, ladders, gaps, out):
    planes = [("x", "y"), ("x", "z"), ("x", "t"),
              ("y", "z"), ("y", "t"), ("z", "t")]
    axes = cfg.axes
    P, w, sub = matrix(df, axes)
    lin = np.linspace(0, 1, 90)
    fig, axs = plt.subplots(2, 3, figsize=(3 * PANEL_W + 1.6, 2 * PANEL_W + 1.0))
    im = None
    for ax, (a, b) in zip(axs.ravel(), planes):
        ia, ib = axes.index(a), axes.index(b)
        d = density(P, w, (ia, ib), (lin, lin), cfg.sigma)
        deficit = 1 - d / d.max()
        im = ax.contourf(lin, lin, deficit.T, levels=np.linspace(0, 1, 11),
                         cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.contour(lin, lin, deficit.T, levels=[0.5, 0.8], colors="white",
                   linewidths=0.5)
        square(ax, label_of(a, ladders, short=True), label_of(b, ladders, short=True))
        for _, g in gaps.iterrows():
            marker = "o" if g["kind"] == "frontier" else "*"
            ax.scatter(g[a], g[b], marker=marker,
                       s=34 if marker == "o" else 78,
                       facecolors="none" if marker == "o" else "#3B2E80",
                       edgecolors="#3B2E80", linewidths=0.9, zorder=5)
        ja, jb = spread(sub[a].to_numpy(float), sub[b].to_numpy(float), radius=0.010)
        ax.scatter(ja, jb, s=4.5, color="white", edgecolors=INK, linewidths=0.25,
                   alpha=0.9, zorder=4)
    cb = fig.colorbar(im, ax=axs, fraction=0.022, pad=0.015,
                      ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label("coverage deficit", fontsize=7)
    cb.outline.set_linewidth(0.4)
    handles = [Line2D([], [], marker="o", ls="", markerfacecolor="none",
                      markeredgecolor="#3B2E80", markersize=5,
                      label="frontier gap (adjacent to existing work)"),
               Line2D([], [], marker="*", ls="", color="#3B2E80", markersize=8,
                      label="island gap (isolated)"),
               Line2D([], [], marker=".", ls="", color=GREY, markersize=5,
                      label="scored paradigm")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    save(fig, out, "fig5_gaps")


# --- figure 6: account space ---------------------------------------------

def fig_accounts(df, thesis_point, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6),
                            gridspec_kw=dict(width_ratios=[1.6, 1, 1], wspace=0.6))
    sub = df[df["topic"].astype(bool)]
    Y, keep = account_matrix(sub)
    sub, Y = sub[keep], Y[keep]
    tab = (pd.DataFrame(Y * sub["w"].to_numpy()[:, None], columns=ACCOUNTS,
                        index=sub.index)
           .groupby(sub["topic"]).sum())
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index[:8]]
    norm = tab.div(tab.sum(axis=1), axis=0)
    keepc = [a for a in ACCOUNTS if norm[a].sum() > 0]
    norm = norm[keepc]
    im = axs[0].imshow(norm.to_numpy(), cmap="magma_r", vmin=0, vmax=1, aspect="auto")
    axs[0].set_xticks(range(norm.shape[1]))
    axs[0].set_xticklabels(norm.columns, fontsize=6.4)
    axs[0].set_yticks(range(norm.shape[0]))
    axs[0].set_yticklabels([t[:28] for t in norm.index], fontsize=5.8)
    axs[0].set_title("A  $P$(account | topic): the $h$ layer", loc="left", fontsize=7.4)
    fig.colorbar(im, ax=axs[0], fraction=0.035, pad=0.02)

    push = pushforward(df)
    keys = [k for k in ACCOUNTS if push[k] > 0.004]
    axs[1].bar(keys, [push[k] for k in keys], color="#7A5EA8", width=0.62)
    for i, k in enumerate(keys):
        axs[1].text(i, push[k] + 0.012, f"{push[k]:.2f}", ha="center", fontsize=6.2)
    axs[1].set_ylabel("share of corpus")
    axs[1].set_title(r"B  pushforward $f_\#\mu$", loc="left", fontsize=7.4)
    axs[1].tick_params(axis="x", labelsize=6.2, rotation=90)

    f, n_eff = account_field(df, thesis_point)
    keys = [k for k in ACCOUNTS if push[k] > 0.004 or f[k] > 0.005]
    axs[2].barh(keys[::-1], [f[k] for k in keys[::-1]], color="#7A5EA8", height=0.62)
    axs[2].set_xlim(0, 1)
    axs[2].set_xlabel(r"$P(c \mid u)$")
    axs[2].tick_params(axis="y", labelsize=6.2)
    axs[2].set_title(f"C  $f$ at the thesis paradigm\neffective $n$ = {n_eff:.1f} rows",
                     loc="left", fontsize=7.4)
    save(fig, out, "fig6_accounts")
    return f, n_eff


# --- figure 7: audit ------------------------------------------------------

# Panel D used to print r in every cell. At eight axes the cells are ~11 pt wide,
# so the numbers had to be set in 4.6 pt and ran into each other; the colour scale
# already carries the pattern, and the exact coefficients are tabulated in
# scoring_report.txt, so nothing is lost by leaving them out. Set this to True to
# get them back — only the coefficients above AUDIT_CORR_FLOOR are then printed.
AUDIT_ANNOTATE_CORR = False
AUDIT_CORR_FLOOR = 0.30


def fig_audit(raw, df, out):
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.55),
                            gridspec_kw=dict(wspace=0.62))
    disp = raw["score provenance"].map(_clean).replace("", "unrecorded").value_counts()
    axs[0].barh(range(len(disp))[::-1], disp.to_numpy(), color="#7A5EA8", height=0.6)
    axs[0].set_yticks(range(len(disp))[::-1])
    axs[0].set_yticklabels([textwrap.shorten(d, 24, placeholder="…")
                            for d in disp.index], fontsize=5.8)
    axs[0].set_xlim(0, disp.max() * 1.16)
    for i, v in zip(range(len(disp))[::-1], disp.to_numpy()):
        axs[0].text(v + disp.max() * 0.02, i, str(v), va="center", fontsize=6.0)
    axs[0].set_xlabel("records")
    panel_title(axs[0], "A  what happened to each record", width=21)

    conf = df["confidence"].value_counts().reindex(["hi", "md", "lo"]).fillna(0)
    axs[1].bar(conf.index, conf.to_numpy(), color=["#157F7F", "#7A5EA8", "#C1425A"],
               width=0.6)
    axs[1].set_ylim(0, max(conf.max() * 1.14, 1))
    for i, v in enumerate(conf.to_numpy()):
        axs[1].text(i, v + conf.max() * 0.02, f"{int(v)}", ha="center", fontsize=6.4)
    axs[1].set_ylabel("scored paradigms")
    panel_title(axs[1], "B  scoring confidence (rule R8)", width=21)

    sub = df.dropna(subset=PRINCIPAL)
    coord = sub[PRINCIPAL].round(2).astype(str).agg("|".join, axis=1)
    per_topic = sub.assign(c=coord).groupby("topic")["c"].nunique()
    n_rows = sub.groupby("topic").size()
    axs[2].scatter(n_rows, per_topic, s=16, color="#4B2E83", alpha=0.8,
                   edgecolors="white", linewidths=0.4)
    lim = max(n_rows.max(), per_topic.max()) + 1
    axs[2].plot([0, lim], [0, lim], color=GREY, lw=0.6, ls="--")
    axs[2].set_xlabel("paradigms in topic")
    axs[2].set_ylabel("distinct coordinates")
    axs[2].set_box_aspect(1)
    panel_title(axs[2], "C  label determinism", width=21,
                sub=f"{int((per_topic == 1).sum())} of {len(per_topic)} topics "
                    f"at one point")

    corr = df[ALL_AXES].dropna(how="all").corr()
    # the diagonal is 1 by construction and, at this size, the only thing the eye
    # sees; blanking it lets the off-diagonal structure carry the panel
    shown = corr.to_numpy().copy()
    np.fill_diagonal(shown, np.nan)
    cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#F2F4F5")
    im = axs[3].imshow(shown, cmap=cmap, vmin=-1, vmax=1)
    ticks = [f"${SHORT[a].replace('x1', 'x_1').replace('y1', 'y_1')}$"
             for a in ALL_AXES]
    axs[3].set_xticks(range(len(ALL_AXES)), ticks, fontsize=6.4)
    axs[3].set_yticks(range(len(ALL_AXES)), ticks, fontsize=6.4)
    axs[3].set_xticks(np.arange(len(ALL_AXES) + 1) - 0.5, minor=True)
    axs[3].set_yticks(np.arange(len(ALL_AXES) + 1) - 0.5, minor=True)
    axs[3].grid(which="minor", color="white", lw=0.5)
    axs[3].tick_params(which="minor", length=0)
    if AUDIT_ANNOTATE_CORR:
        for i in range(len(ALL_AXES)):
            for j in range(len(ALL_AXES)):
                v = corr.to_numpy()[i, j]
                if np.isfinite(v) and i != j and abs(v) >= AUDIT_CORR_FLOOR:
                    axs[3].text(j, i, f"{v:.2f}".lstrip("0").replace("-0", "-"),
                                ha="center", va="center", fontsize=5.0,
                                color="white" if abs(v) > 0.55 else INK)
    panel_title(axs[3], "D  axis collinearity", width=21)
    cb = fig.colorbar(im, ax=axs[3], fraction=0.045, pad=0.04,
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Pearson $r$", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.8)
    cb.outline.set_linewidth(0.4)
    save(fig, out, "fig7_audit")
    return corr


# --- figure 8: clustering vs the labels ----------------------------------

CLUSTER_PALETTE = ["#1B3A5C", "#E08A2E", "#6B8E4E", "#A03050", "#5D4E8C",
                   "#2E8B8B", "#8C6239", "#B0447A"]


def fig_clusters(res, ladders, out):
    """What the corpus splits into on its own, next to what the labels say.

    B and C are the same points in the same coordinates, coloured two different ways:
    anywhere the colours disagree is a paradigm the label puts in one literature and
    the geometry puts in the other.
    """
    sub, X, lab, given = res["sub"], res["X"], res["labels"], res["given"]
    axes, C = res["axes"], res["centers"]
    a, b = axes[0], axes[1]
    fig = plt.figure(figsize=(7.4, 5.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.0], hspace=0.52, wspace=0.44)

    # A -- choosing k
    ax = fig.add_subplot(gs[0, 0])
    cur = res["curve"]
    ax.plot(cur["k"], cur["silhouette"], "-o", ms=3.2, lw=1.1, color=INK,
            label="silhouette")
    ax.plot(cur["k"], cur["stability"], "-s", ms=3.2, lw=1.1, color="#7A5EA8",
            label="bootstrap stability")
    ax.axvline(res["k"], color=CLUSTERS["surprise"]["color"], lw=1.0, ls="--", zorder=0)
    ax.text(res["k"] + 0.1, 0.965, f"k = {res['k']}", fontsize=6.4, ha="left",
            va="top", color=CLUSTERS["surprise"]["color"],
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("number of clusters $k$")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6, loc="lower right")
    ax.set_title("A  choosing $k$", loc="left", fontsize=7.6)

    # B / C -- the same points, coloured two ways
    pa, pb = spread(sub[a].to_numpy(float), sub[b].to_numpy(float))
    for col, (title, colours, cents) in enumerate([
            ("B  clusters found by the data", [CLUSTER_PALETTE[v % 8] for v in lab], C),
            ("C  the labels assigned by hand",
             [CLUSTERS[g]["color"] for g in given], None)]):
        ax = fig.add_subplot(gs[0, col + 1])
        square(ax, label_of(a, short=True), label_of(b, short=True))
        ax.scatter(pa, pb, s=17, c=colours, edgecolors="white", linewidths=0.35,
                   alpha=0.92, zorder=3)
        if cents is not None:
            for j, cc in enumerate(cents):
                ax.scatter([cc[axes.index(a)]], [cc[axes.index(b)]], s=80, marker="X",
                           color=CLUSTER_PALETTE[j % 8], edgecolors=INK, linewidths=0.7,
                           zorder=5)
                ax.text(cc[axes.index(a)] + 0.06, cc[axes.index(b)] + 0.06, f"c{j}",
                        fontsize=6.8, ha="left", color=CLUSTER_PALETTE[j % 8],
                        fontweight="bold", zorder=6)
        ax.set_title(title, loc="left", fontsize=7.6)

    # D -- contingency
    ax = fig.add_subplot(gs[1, 0])
    M = res["contingency"]
    ax.imshow(M / M.sum(1, keepdims=True), cmap="Purples", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(res["given_names"], fontsize=6.0, rotation=35, ha="right")
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([f"c{i}" for i in res["found_names"]], fontsize=6.4)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=6.4,
                    color="white" if M[i, j] / M[i].sum() > 0.55 else INK)
    ax.set_title("D  found $\\times$ assigned", loc="left", fontsize=7.6)
    pstr = "$p$ < 0.001" if res["p_ari"] < 0.001 else f"$p$ = {res['p_ari']:.3f}"
    ax.text(0, -0.30, f"ARI {res['ari']:.2f} ({pstr})\nAMI {res['ami']:.2f} · "
            f"{res['agreement']:.0%} of rows agree", transform=ax.transAxes,
            fontsize=6.4, va="top", color="#42525B")

    # E -- what each discovered cluster is, on every axis
    ax = fig.add_subplot(gs[1, 1:])
    xs = np.arange(len(axes))
    for j, cc in enumerate(C):
        n = int((lab == j).sum())
        ax.plot(xs, cc, "-o", ms=4, lw=1.4, color=CLUSTER_PALETTE[j % 8],
                label=f"c{j}  (n = {n})")
        for xi, v in zip(xs, cc):
            lo, hi = np.percentile(X[lab == j, xi], [25, 75])
            ax.plot([xi, xi], [lo, hi], color=CLUSTER_PALETTE[j % 8], lw=3.4, alpha=0.22,
                    solid_capstyle="round", zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([label_of(k, short=True) for k in axes], fontsize=8.5)
    for xi, k in enumerate(axes):
        ax.text(xi, -0.20, textwrap.shorten(
            re.sub(r"\s*\(.*?\)\s*$", "", ladders.get(k, {}).get("name", "")
                   or FALLBACK_LABEL[k]), 20, placeholder="…"),
            fontsize=5.4, ha="center", color="#6C7C85", transform=ax.get_xaxis_transform())
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("rung")
    ax.legend(fontsize=6.2, ncol=min(4, len(C)), loc="upper left")
    ax.set_title("E  centroid of each discovered cluster, with its interquartile range",
                 loc="left", fontsize=7.6)
    save(fig, out, "fig8_clusters")


# --- appendix figures -----------------------------------------------------

def fig_ladders(df, ladders, out):
    """Occupancy of every rung, drawn as counts on the lattice rather than a KDE.

    A smooth density over a discrete ladder invents mass between rungs where no
    design can sit; bars at the admissible values do not.
    """
    keys = [a for a in ALL_AXES if a in ladders] or ALL_AXES
    n = len(keys)
    ncol = int(np.ceil(n / 2))
    fig, axs = plt.subplots(2, ncol, figsize=(2.15 * ncol + 0.6, 4.5))
    axs = np.atleast_2d(axs)
    for pos, (ax, a) in enumerate(zip(axs.ravel(), keys)):
        rungs = [v for v, *_ in ladders.get(a, {}).get("rungs", [])] or \
            sorted(df[a].dropna().unique())
        vals = df[a].dropna()
        counts = []
        for v in rungs:
            counts.append(int((np.abs(vals - v) < 0.03).sum()))
        stray = int(len(vals) - sum(counts))
        ax.bar(rungs, counts, width=0.075, color="#7A5EA8", alpha=0.85)
        for cl, style in CLUSTERS.items():
            m = df["cluster"] == cl
            ax.plot(df.loc[m, a], np.full(m.sum(), -max(counts + [1]) * 0.10), "|",
                    color=style["color"], ms=3.4, mew=0.8)
        ax.set_xlim(-0.08, 1.08)
        ax.set_xticks(rungs)
        ax.set_xticklabels([f"{v:g}" for v in rungs], fontsize=5.4, rotation=90)
        # the criteria-sheet names are sentences, so they are wrapped to the panel
        # and the count is demoted to a second line rather than trailing the name
        name = re.sub(r"\s*\(.*?\)\s*$", "",
                      ladders.get(a, {}).get("name", "") or FALLBACK_LABEL[a]).strip()
        sym = SHORT[a].replace("x1", "x_1").replace("y1", "y_1")
        panel_title(ax, name, tail=f"${sym}$", width=27, fontsize=6.5, pad=3.0,
                    sub=f"n = {len(vals)}" + (f" · {stray} off-rung" if stray else ""))
        ax.tick_params(axis="y", labelsize=5.6)
        ax.spines["left"].set_visible(False)
        if pos % ncol == 0:
            ax.set_ylabel("paradigms", fontsize=6.0)
    for ax in axs.ravel()[len(keys):]:
        ax.set_visible(False)
    fig.tight_layout(h_pad=1.9, w_pad=1.1)
    save(fig, out, "figA1_ladders")


def fig_year(df, ladders, out):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for ax, a in zip(axs, ["x", "z"]):
        sub = df.dropna(subset=["year", a]).sort_values("year")
        for cl, style in CLUSTERS.items():
            m = sub["cluster"] == cl
            ax.scatter(sub.loc[m, "year"], sub.loc[m, a], s=13,
                       color=style["color"], alpha=0.85, edgecolors="white",
                       linewidths=0.3, label=style["label"])
        roll = sub[a].rolling(9, min_periods=4, center=True).mean()
        ax.plot(sub["year"], roll, color=INK, lw=1.0)
        ax.set_xlabel("publication year")
        ax.set_ylabel(label_of(a, ladders))
        ax.set_ylim(-0.05, 1.05)
    axs[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    save(fig, out, "figA2_year")


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(Path(out) / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. interactive page
# ---------------------------------------------------------------------------

def plotly_bundle():
    """Inline plotly if it is installed locally, otherwise fall back to the CDN."""
    try:
        import plotly
        js = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        if js.exists():
            return f"<script>{js.read_text(encoding='utf-8')}</script>", True
    except Exception:
        pass
    return ('<script src="https://cdn.plot.ly/plotly-3.0.0.min.js" '
            'charset="utf-8"></script>', False)


def html_payload(df, ladders, cfg, gaps, raw, clus=None):
    def num(v):
        return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4)

    fit = None
    if clus is not None:
        # the partition is carried per paradigm_id, so the page can put the label a
        # human assigned and the cluster the code found on the same point. Rows the
        # clustering could not use (a missing coordinate on one of its axes) simply
        # have no entry, and the page draws them as 'not clustered' rather than
        # silently folding them into a cluster.
        sub = clus["sub"]
        fit = {
            "k": int(clus["k"]), "method": clus["method"],
            "axes": list(clus["axes"]),
            "found": {str(i): int(v) for i, v in
                      zip(sub["paradigm_id"], clus["labels"])},
            "foundNames": [f"c{int(v)}" for v in clus["found_names"]],
            "givenNames": [str(v) for v in clus["given_names"]],
            "contingency": [[int(v) for v in row] for row in clus["contingency"]],
            "centers": [[num(v) for v in row] for row in clus["centers"]],
            "mapping": {str(int(k)): str(v) for k, v in (clus["mapping"] or {}).items()},
            "ari": num(clus["ari"]), "pAri": num(clus["p_ari"]),
            "ami": num(clus["ami"]), "agreement": num(clus["agreement"]),
            "silhouette": num(clus["silhouette"]), "stability": num(clus["stability"]),
            "curve": [{"k": int(r["k"]), "silhouette": num(r["silhouette"]),
                       "stability": num(r["stability"])}
                      for _, r in clus["curve"].iterrows()],
        }

    pts = []
    for _, r in df.iterrows():
        pts.append({
            "id": r["paradigm_id"], "key": r["citekey"] or r["paradigm_id"],
            "title": r["title"][:90], "year": num(r["year"]),
            "cluster": r["cluster"], "conf": r["confidence"], "w": num(r["w"]),
            "acc": r["account"], "topic": r["topic"],
            "note": (r["task_note"] or r["scoring_note"] or r["summary"])[:180],
            **{a: num(r[a]) for a in ALL_AXES},
        })
    disp = (raw["score provenance"].map(_clean).replace("", "unrecorded")
            .value_counts().items())
    return {
        "pts": pts,
        "axes": {a: {"label": re.sub(r"\$", "", label_of(a, ladders)),
                     "rungs": [v for v, *_ in ladders.get(a, {}).get("rungs", [])]}
                 for a in ALL_AXES},
        "clusters": CLUSTERS,
        "regions": {k: {"title": v["title"], "color": v["color"],
                        "constraints": v["constraints"]} for k, v in REGIONS.items()},
        "cfg": {"sigma": cfg.sigma, "sigmaReach": cfg.sigma_reach,
                "principal": cfg.axes, "grid": cfg.grid,
                "emptyDeficit": EMPTY_DEFICIT},
        "accounts": ACCOUNTS,
        "gaps": gaps.to_dict("records"),
        "audit": {"dispositions": [[k, int(v)] for k, v in disp],
                  "records": int(len(raw)),
                  "offLattice": [r["paradigm_id"] for _, r in
                                 df[df["off_lattice"]].iterrows()]},
        "fit": fit,
    }


HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Towards a formal literature review — surprising events during ongoing motor control</title>
<style>
:root {
  --ink:#101820; --paper:#FFFFFF; --panel:#F4F6F7;
  --motor:#157F7F; --sal:#C1425A; --gap:#6A3D9A; --rule:#DCE3E6; --mut:#6C7C85;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink);
  font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  line-height:1.55; -webkit-font-smoothing:antialiased; }
.wrap { max-width:1060px; margin:0 auto; padding:0 28px 96px; }
header { padding:60px 0 30px; border-bottom:2px solid var(--ink); }
h1 { font-family:Georgia,'Iowan Old Style',Palatino,serif; font-weight:400;
  font-size:clamp(28px,4.2vw,44px); line-height:1.1; margin:0 0 14px; letter-spacing:-.01em; }
h1 em { font-style:italic; color:var(--gap); }
.sub { font-size:15px; max-width:64ch; color:#3C4B54; margin:0; }
.coord { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px;
  background:var(--panel); border:1px solid var(--rule); border-radius:2px;
  padding:3px 8px; display:inline-block; }
section { padding:48px 0 8px; border-bottom:1px solid var(--rule); }
.eyebrow { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11px;
  text-transform:uppercase; letter-spacing:.13em; color:var(--mut); margin:0 0 8px; }
h2 { font-family:Georgia,'Iowan Old Style',Palatino,serif; font-weight:400; font-size:26px; margin:0 0 10px; }
p.lead { max-width:70ch; margin:0 0 18px; }
p.note { max-width:70ch; margin:14px 0 0; font-size:13.5px; color:#4A5A63; }
.plot { background:var(--panel); border:1px solid var(--rule); border-radius:3px;
  padding:6px 4px; margin:16px 0 6px; }
.controls { display:flex; gap:22px; flex-wrap:wrap; align-items:flex-end;
  background:var(--panel); border:1px solid var(--rule); padding:14px 18px; border-radius:3px; }
.ctl label { display:block; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11px; color:#54646D; text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }
input[type=range] { width:150px; accent-color:var(--gap); }
select { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px;
  padding:5px 6px; border:1px solid var(--rule); background:var(--paper); color:var(--ink); }
.readout { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px;
  margin-left:auto; text-align:right; }
.readout b { font-size:30px; font-weight:400; display:block; line-height:1; color:var(--gap);
  font-family:Georgia,'Iowan Old Style',Palatino,serif; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:26px; }
.card { border-left:3px solid var(--rule); padding:2px 0 2px 16px; }
.card.motor { border-color:var(--motor); } .card.sal { border-color:var(--sal); }
.card.gap { border-color:var(--gap); }
.card h3 { font-size:14px; margin:0 0 6px; } .card p { margin:0; font-size:13.5px; color:#42525B; }
table.audit { border-collapse:collapse; font-size:13.5px; width:100%; margin-top:8px; }
table.audit th, table.audit td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--rule); }
table.audit th { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11px;
  text-transform:uppercase; letter-spacing:.08em; color:var(--mut); font-weight:400; }
table.audit td.num { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
.flag { background:#FFF6E5; border-left:3px solid #E0A32E; padding:14px 18px; font-size:13.5px; margin-top:20px; }
.flag b { display:block; margin-bottom:4px; }
button.seg { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11px;
  letter-spacing:.08em; text-transform:uppercase; background:var(--paper); border:1px solid var(--rule);
  padding:7px 12px; cursor:pointer; color:#54646D; border-radius:2px; }
button.seg:hover { border-color:var(--ink); color:var(--ink); }
button.seg[aria-pressed=true] { background:var(--ink); color:#fff; border-color:var(--ink); }
.chips { display:flex; gap:8px; flex-wrap:wrap; margin:12px 0 0; }
.hit { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11.5px;
  background:var(--panel); border:1px solid var(--rule); padding:3px 7px; border-radius:2px; }
.searchbar { display:flex; align-items:center; gap:12px; margin:16px 0 0; }
.searchbar input { flex:1; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:13px; padding:9px 12px; border:1px solid var(--rule); border-radius:2px;
  background:var(--paper); color:var(--ink); }
.searchbar input:focus { outline:none; border-color:var(--gap); }
.hits { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11.5px;
  color:var(--mut); white-space:nowrap; }
.qlist { max-height:260px; overflow-y:auto; margin-top:10px; }
.qlist:empty { display:none; }
.qrow { display:grid; grid-template-columns:150px 1fr 190px; gap:12px; align-items:baseline;
  padding:7px 10px; border-bottom:1px solid var(--rule); font-size:12.5px; cursor:pointer; }
.qrow:hover { background:var(--panel); }
.qrow .k { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11.5px; }
.qrow .c { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11px;
  color:var(--mut); text-align:right; }
.qrow .t { color:#42525B; }
.dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; }
.funnel { margin-top:14px; }
.frow { display:grid; grid-template-columns:150px 1fr 42px; gap:10px; align-items:center;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11.5px; margin-bottom:5px; }
.bar { height:13px; background:var(--gap); opacity:.85; border-radius:1px; transition:width .18s; }
.frow span.lab { color:#54646D; } .frow span.n { text-align:right; }
table.cross { border-collapse:collapse; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:12px; margin-top:16px; }
table.cross th { font-weight:400; color:var(--mut); font-size:11px; text-transform:uppercase;
  letter-spacing:.06em; padding:6px 10px; text-align:center; white-space:nowrap; }
table.cross th.rowhead { text-align:right; }
table.cross td { padding:0; }
table.cross td button { width:100%; min-width:74px; border:1px solid var(--rule); background:var(--paper);
  font-family:inherit; font-size:12.5px; color:var(--ink); padding:9px 6px; cursor:pointer; }
table.cross td button:hover { border-color:var(--ink); }
table.cross td button[data-on=true] { outline:2px solid var(--gap); outline-offset:-2px; }
table.cross td.matched button { font-weight:700; }
table.cross td.zero button { color:#B7C2C7; }
table.cross td.tot, table.cross th.tot { color:var(--mut); text-align:center; padding:6px 10px; }
.swatches { display:flex; gap:14px; flex-wrap:wrap; margin:12px 0 0; font-size:12px; color:#42525B; }
.swatches span.s { display:inline-flex; align-items:center; gap:6px; }
.qrow.fitrow { grid-template-columns:150px 1fr 200px 26px; }
.qrow .ok { text-align:center; font-size:13px; }
footer { padding-top:30px; font-size:12.5px; color:var(--mut); }
@media (max-width:760px){ .two{grid-template-columns:1fr;} .wrap{padding:0 16px 60px;} .readout{margin-left:0;} }
@media (prefers-reduced-motion:reduce){ *{transition:none!important;} }
</style>
__PLOTLY__
</head><body><div class="wrap">

<header>
  <p class="eyebrow">paradigm space · __NPARA__ scored paradigms · __NRECORDS__ records</p>
  <h1>Towards a formal literature review <br>
      <em>surprising events during ongoing motor control</em></h1>
  <p class="sub">We have developed a low-dimensional geometrical representation of the 
    experimental paradigms used to study the processing and integration of surprising 
    events during ongoing behaviour. Every experimental paradigm in the reviewed literature
    is a point in this space. The axes are interpretable variables defining the structure
    of the paradigm: 
    : how much capacity the task commits,
     how long the motor command stays open to revision, how deep a hierarchy the
     perturbing event's statistics demand, and how much of the task the event carries.
     One row of the workbook is one paradigm, so a paper running four experiments occupies
     four points. Two literatures that rarely cite each other occupy opposite corners;
     the argument of the thesis is about what lies between them.
     <span class="coord">thesis paradigm · __THESISCOORD__</span></p>
     <p><b>Clustering method:</b> {clustering_method}</p>
    <p><b>Number of clusters:</b> {n_clusters}</p>
</header>

<section>
  <p class="eyebrow">step 1 — the coordinates</p>
  <h2>Choose three axes and rotate</h2>
  <p class="lead">Colour is cluster, marker is scoring confidence. Hover any point for the
     reference, its coordinates and the note the score was justified with. The shaded volume
     is the region set in step 2; the three axes on screen are the three you pick here, so a
     constraint on a fourth axis is not carried and the box is drawn faint when that happens.</p>
  <div class="controls">
    <div class="ctl"><label>axis 1</label><select id="a1"></select></div>
    <div class="ctl"><label>axis 2</label><select id="a2"></select></div>
    <div class="ctl"><label>axis 3</label><select id="a3"></select></div>
    <div class="ctl"><label>colour</label><select id="colorby">
      <option value="cluster">cluster</option><option value="conf">confidence</option>
      <option value="account">dominant account</option><option value="year">year</option>
    </select></div>
    <div class="ctl"><label>corpus</label>
      <button class="seg" id="tlo" aria-pressed="false">drop lo</button>
      <button class="seg" id="tth" aria-pressed="false">hide thesis</button></div>
  </div>
  <div class="searchbar">
    <input type="search" id="q" placeholder="search __NPARA__ paradigms — citekey, title, topic, account, note">
    <span class="hits" id="qn"></span>
  </div>
  <div id="qlist" class="qlist"></div>
  <div class="plot"><div id="cube" style="height:540px;width:100%"></div></div>
  <div class="two">
    <div class="card motor"><h3>Motor control</h3>
      <p>Continuous, task-relevant control: high difficulty, long timescale, and an event
         that redefines the goal. Every perturbation is something the controller must use.</p></div>
    <div class="card sal"><h3>Environmental Surprise</h3>
      <p>Passive or discrete-response stimulation: low difficulty, short timescale, an event
         that carries no task information and is often explicitly to be ignored.</p></div>
  </div>
</section>

<section>
  <p class="eyebrow">step 2 — the claim, made breakable</p>
  <h2>Move the walls of the box yourself</h2>
  <p class="lead">The region is axis-aligned, so it can be interrogated directly. Drag the
     thresholds: the counter reports how many paradigms are inside on <i>all four</i>
     principal axes, and the funnel underneath shows where the corpus drops out. A count
     taken from a two-axis picture is a count of a shadow — this one is not.</p>
  <div class="controls">
    <div class="ctl"><label>motor difficulty x ≥ <span id="vx" class="coord"></span></label>
      <input type="range" id="bx" min="0" max="1" step="0.01" value="0.67"></div>
    <div class="ctl"><label>motor timescale y ≥ <span id="vy" class="coord"></span></label>
      <input type="range" id="by" min="0" max="1" step="0.01" value="0.57"></div>
    <div class="ctl"><label>surprise hierarchy z ≥ <span id="vz" class="coord"></span></label>
      <input type="range" id="bz" min="0" max="1" step="0.01" value="0"></div>
    <div class="ctl"><label>task relevance t ≤ <span id="vt" class="coord"></span></label>
      <input type="range" id="bt" min="0" max="1" step="0.01" value="0.17"></div>
    <div class="readout">paradigms inside<b id="count">0</b>
      <span id="ofn"></span></div>
  </div>
  <div class="chips" id="presets"></div>
  <div class="funnel" id="funnel"></div>
  <p class="note" id="which"></p>
</section>

<section>
  <p class="eyebrow">step 3 — emptiness as a low-density basin</p>
  <h2>The corpus as a field, not a scatter</h2>
  <p class="lead">Replace each paradigm by an isotropic Gaussian and the corpus becomes a
     density over the space; the coverage deficit is what is left of it. Because a marginal
     of a Gaussian mixture is a Gaussian mixture, the plane below is exact rather than
     histogrammed. Gaps are searched in four dimensions and then projected: open circles are
     <b>frontier</b> gaps, the smallest modification of an existing design that lands
     somewhere new, and stars are <b>island</b> gaps, which need a new paradigm. Both are
     recomputed when you move σ.</p>
  <div class="controls">
    <div class="ctl"><label>plane</label><select id="plane"></select></div>
    <div class="ctl"><label>bandwidth σ = <span id="vs" class="coord"></span></label>
      <input type="range" id="bs" min="0.05" max="0.20" step="0.005" value="0.09"></div>
    <div class="ctl"><label>show</label>
      <button class="seg" id="tgap" aria-pressed="true">gaps</button>
      <button class="seg" id="tpts" aria-pressed="true">paradigms</button></div>
    <div class="readout">reachability of the box<b id="reach">—</b>
      <span id="reachnote"></span></div>
  </div>
  <div class="plot"><div id="dens" style="height:460px;width:100%"></div></div>
  <p class="note">Reachability is measured from the centre of the box you set in step 2 to
     the nearest published paradigm, so it answers a different question from the count:
     not whether anyone has been there, but how far it is from somewhere they have.</p>
</section>

<section>
  <p class="eyebrow">step 4 — an empty region has to be empty in account space too</p>
  <h2>What the field would call the experiment you just specified</h2>
  <p class="lead">A region can be paper-empty and still map onto a well-covered part of
     theory, in which case filling it teaches nothing. The map f from a design to the
     distribution over computational accounts is estimated by kernel regression on the
     scored corpus, with the thesis rows held out. It is evaluated at the centre of your box,
     so it moves as you move the walls.</p>
  <div class="two">
    <div><p class="eyebrow">f at the centre of the box</p>
      <div class="plot"><div id="facc" style="height:280px;width:100%"></div></div></div>
    <div><p class="eyebrow">accounts by corpus mass</p>
      <div class="plot"><div id="push" style="height:280px;width:100%"></div></div></div>
  </div>
</section>

<section id="fitsec">
  <p class="eyebrow">step 5 — the labels, checked against the geometry</p>
  <h2>What the corpus splits into when nobody tells it the answer</h2>
  <p class="lead">The cluster on every point so far is a label assigned by hand. This
     section shows the partition <span class="coord" id="fitmeta"></span> found by
     clustering the coordinates alone, with the labels hidden, so the two can be
     compared point by point. Colour the same scatter both ways: where the two pictures
     differ, the label and the geometry disagree about that paradigm. Use
     <b>disagreements only</b>, or click a cell of the table, to list exactly which ones.</p>
  <div class="controls">
    <div class="ctl"><label>plane</label><select id="fplane"></select></div>
    <div class="ctl"><label>hand label</label><select id="fgiven"></select></div>
    <div class="ctl"><label>cluster found</label><select id="ffound"></select></div>
    <div class="ctl"><label>show</label>
      <button class="seg" id="fdis" aria-pressed="false">disagreements only</button></div>
    <div class="readout">rows where the two agree<b id="fagree">—</b>
      <span id="fstat"></span></div>
  </div>
  <div class="two">
    <div><p class="eyebrow">coloured by the cluster the code found</p>
      <div class="plot"><div id="ffoundplot" style="height:330px;width:100%"></div></div></div>
    <div><p class="eyebrow">the same points, coloured by the hand label</p>
      <div class="plot"><div id="fgivenplot" style="height:330px;width:100%"></div></div></div>
  </div>
  <div class="swatches" id="fswatch"></div>
  <p class="eyebrow" style="margin-top:26px">where each hand label went</p>
  <div class="plot"><div id="fflow" style="height:300px;width:100%"></div></div>
  <table class="cross" id="fcross"></table>
  <div id="flist" class="qlist"></div>
  <p class="note" id="fnote"></p>
</section>

<section>
  <p class="eyebrow">step 6 — what the corpus is and is not</p>
  <h2>Audit</h2>
  <table class="audit" id="audit"></table>
  <div class="flag"><b>Read before quoting a number from this page</b>
    Absence in a corpus is not absence in a literature: the smooth-pursuit and
    saccadic-inhibition work delivers transient task-irrelevant events during ongoing
    pursuit and is not yet in the database, so every statement here is a statement about
    this corpus. Scores marked <span class="coord">lo</span> were inferred rather than read
    from a Methods section — use the <span class="coord">drop lo</span> toggle before
    quoting any coverage claim.</div>
</section>

<footer>Generated from <span class="coord">__SOURCE__</span> ·
  σ = <span id="fsig">0.09</span> · reachability scale __SIGR__ ·
  weights hi __WHI__ / md __WMD__ / lo __WLO__ ·
  every figure and every number on this page comes from
  <span class="coord">paradigm_space.py</span></footer>
</div>

<script>
const D = __DATA__;
const PRIN = D.cfg.principal;
const COL = {}; Object.entries(D.clusters).forEach(([k,v]) => COL[k] = v.color);
const SYM = {hi:'circle', md:'square', lo:'diamond-open'};
let state = {dropLo:false, hideThesis:false, sigma:D.cfg.sigma,
             showGaps:true, showPts:true, plane:['x','z'],
             axes:['x','y','t'], colorby:'cluster', query:'',
             box:{x:0.67, y:0.57, z:0, t:0.17}};

/* the clustering the script ran with -k, or null if it was switched off.
   Same palette as figure 8, so the page and the figure name the clusters alike. */
const FIT = D.fit;
const CPAL = ['#1B3A5C','#E08A2E','#6B8E4E','#A03050','#5D4E8C','#2E8B8B','#8C6239','#B0447A'];
const AGREE_COL = {'agree':'#157F7F', 'disagree':'#C1425A', 'unclustered':'#C2CBCF'};
const AGREE_LAB = {'agree':'label and geometry agree', 'disagree':'they disagree',
                   'unclustered':'not clustered'};
D.pts.forEach(p => { p.found = (FIT && FIT.found[p.id] !== undefined) ? FIT.found[p.id] : null; });
state.fit = {plane:['x','t'], given:'all', found:'all', disagree:false};

const el = id => document.getElementById(id);
const fmt = v => (v===null||v===undefined) ? '—' : (+v).toFixed(2);
const cpal = i => CPAL[i % CPAL.length];
/* a point agrees when its hand label is the one the best one-to-one match assigns
   to the cluster it landed in; anything the clustering could not use is neither */
function agreeKey(p){
  if (!FIT || p.found === null) return 'unclustered';
  return FIT.mapping[p.found] === p.cluster ? 'agree' : 'disagree';
}
function rgba(hex, a){
  const v = parseInt(hex.slice(1), 16);
  return `rgba(${v>>16&255},${v>>8&255},${v&255},${a})`;
}

function pool(){
  return D.pts.filter(p => (!state.dropLo || p.conf !== 'lo')
                        && (!state.hideThesis || p.cluster !== 'thesis'));
}
/* The axes are ladders, so exact ties are the rule: eight configurations scored
   identically are one marker unless they are fanned apart. Deterministic spiral,
   always narrower than half a rung, applied for display only. */
function spread(pts, a, b, c, radius){
  radius = radius || 0.013;
  const key = {}, out = pts.map(p => ({x:p[a], y:p[b], z:p[c]}));
  pts.forEach((p,i) => {
    const k = [p[a],p[b],p[c]].map(v => (+v).toFixed(3)).join('|');
    (key[k] = key[k] || []).push(i);
  });
  const golden = Math.PI * (3 - Math.sqrt(5));
  Object.values(key).forEach(idx => {
    if (idx.length < 2) return;
    idx.forEach((i,j) => {
      const rad = radius * Math.sqrt((j+0.5)/idx.length) * Math.sqrt(idx.length);
      out[i].x += rad*Math.cos(j*golden);
      out[i].y += rad*Math.sin(j*golden);
      out[i].z += rad*Math.cos(j*golden + 1.1)*0.6;
    });
  });
  return out;
}
function matches(p){
  const q = state.query;
  if (!q) return true;
  return [p.key, p.title, p.topic, p.acc, p.note, p.cluster]
    .some(v => (v||'').toLowerCase().includes(q));
}
function full(){ return pool().filter(p => PRIN.every(a => p[a] !== null)); }
function inBox(p, b){ return p.x >= b.x && p.y >= b.y && p.z >= b.z && p.t <= b.t; }
function boxCentre(b){
  return {x:(b.x+1)/2, y:(b.y+1)/2, z:(b.z+1)/2, t:b.t/2};
}

/* ---------- the cube ---------- */
function cubeTraces(){
  const [a1,a2,a3] = state.axes;
  const pts = pool().filter(p => p[a1]!==null && p[a2]!==null && p[a3]!==null);
  const traces = [];
  const groups = {};
  pts.forEach(p => {
    let key = state.colorby === 'cluster' ? p.cluster
            : state.colorby === 'conf' ? p.conf
            : state.colorby === 'account' ? (p.acc || 'unassigned')
            : state.colorby === 'found' ? (p.found === null ? 'unclustered' : 'c'+p.found)
            : state.colorby === 'agree' ? agreeKey(p) : 'year';
    (groups[key] = groups[key] || []).push(p);
  });
  const palette = ['#157F7F','#C1425A','#4B2E83','#D2892A','#3E7CB1','#7A5EA8','#8A9A5B','#B05A3C'];
  let i = 0;
  for (const [key, arr] of Object.entries(groups)) {
    const colour = state.colorby === 'cluster' ? COL[key]
                 : state.colorby === 'found' ? (key === 'unclustered' ? AGREE_COL.unclustered
                                                                     : cpal(+key.slice(1)))
                 : state.colorby === 'agree' ? AGREE_COL[key]
                 : state.colorby === 'year' ? undefined : palette[i++ % palette.length];
    const xy = spread(arr, a1, a2, a3);
    traces.push({
      type:'scatter3d', mode:'markers',
      name: state.colorby === 'agree' ? AGREE_LAB[key]
          : (D.clusters[key] && state.colorby === 'cluster') ? D.clusters[key].label : key,
      x:xy.map(v=>v.x), y:xy.map(v=>v.y), z:xy.map(v=>v.z),
      text:arr.map(p=>`<b>${p.key}</b> (${p.year||'—'})<br>${p.title}<br>`
        + `<span style="font-family:monospace">` + PRIN.map(a=>`${a} ${fmt(p[a])}`).join(' · ')
        + `</span><br>${p.conf} · ${p.acc||'no account'}`
        + `<br>label ${p.cluster}` + (FIT ? ` · found ${p.found===null?'—':'c'+p.found}` : '')
        + (p.note?`<br><i>${p.note}</i>`:'')),
      hovertemplate:'%{text}<extra></extra>',
      marker:{ size:arr.map(p => (state.query && matches(p)) ? 11
                                : inBox(p, state.box) ? 8 : 4.5),
               opacity: 0.92,
               color: state.colorby === 'year' ? arr.map(p=>p.year) : colour,
               colorscale: state.colorby === 'year' ? 'Viridis' : undefined,
               showscale: state.colorby === 'year',
               symbol:arr.map(p=>SYM[p.conf]||'circle'),
               line:{width:arr.map(p => (state.query && matches(p)) ? 2.4 : 0.5),
                     color:arr.map(p => (state.query && matches(p)) ? '#101820' : 'white')} }
    });
  }
  const b = state.box, carried = PRIN.filter(a => state.axes.includes(a));
  const lim = {x:[b.x,1], y:[b.y,1], z:[b.z,1], t:[0,b.t]};
  const bx = lim[a1] || [0,1], by = lim[a2] || [0,1], bz = lim[a3] || [0,1];
  const dropped = PRIN.filter(a => !state.axes.includes(a) &&
        !(a==='z' && b.z<=0.001) && !(a==='t' && b.t>=0.999) &&
        !((a==='x'&&b.x<=0.001)||(a==='y'&&b.y<=0.001)));
  traces.push({
    type:'mesh3d', name:'region', hoverinfo:'skip', showlegend:false,
    x:[bx[0],bx[0],bx[1],bx[1],bx[0],bx[0],bx[1],bx[1]],
    y:[by[0],by[1],by[1],by[0],by[0],by[1],by[1],by[0]],
    z:[bz[0],bz[0],bz[0],bz[0],bz[1],bz[1],bz[1],bz[1]],
    i:[7,0,0,0,4,4,6,6,4,0,3,2], j:[3,4,1,2,5,6,5,2,0,1,6,3],
    k:[0,7,2,3,6,7,1,1,5,5,7,6],
    color:'#6A3D9A', opacity: dropped.length ? 0.05 : 0.13, flatshading:true
  });
  return {traces, dropped};
}
function drawCube(){
  const {traces, dropped} = cubeTraces();
  const [a1,a2,a3] = state.axes;
  const ax = n => ({title:{text:D.axes[n].label, font:{size:11}}, range:[-0.03,1.03],
                    tickvals:[0,0.5,1], gridcolor:'#E4E9EB', zeroline:false});
  Plotly.react('cube', traces, {
    margin:{l:0,r:0,t:6,b:0}, paper_bgcolor:'rgba(0,0,0,0)',
    scene:{xaxis:ax(a1), yaxis:ax(a2), zaxis:ax(a3), aspectmode:'cube',
           camera:{eye:{x:1.5,y:1.5,z:1.1}}},
    legend:{orientation:'h', y:-0.02, font:{size:11}},
    annotations: dropped.length ? [{text:'box drawn faint: '+dropped.join(', ')
      +' not on screen', showarrow:false, x:0, y:1, xref:'paper', yref:'paper',
      font:{size:11, color:'#6C7C85'}}] : []
  }, {displayModeBar:false, responsive:true});
}

/* ---------- step 2: box, funnel, list ---------- */
function refreshBox(){
  const b = state.box;
  ['x','y','z','t'].forEach(a => el('v'+a).textContent = b[a].toFixed(2));
  const F = full();
  const hit = F.filter(p => inBox(p, b));
  el('count').textContent = hit.length;
  el('ofn').textContent = 'of ' + F.length + ' scored on all four axes';
  el('which').innerHTML = hit.length
    ? hit.map(p=>`<span class="hit">${p.key}</span>`).join(' ')
    : '<span class="hit">no paradigm in the box</span>';

  const seq = [['all scored on x, y, z, t', F.length],
               ['x ≥ '+b.x.toFixed(2), F.filter(p=>p.x>=b.x).length],
               ['+ y ≥ '+b.y.toFixed(2), F.filter(p=>p.x>=b.x&&p.y>=b.y).length],
               ['+ z ≥ '+b.z.toFixed(2), F.filter(p=>p.x>=b.x&&p.y>=b.y&&p.z>=b.z).length],
               ['+ t ≤ '+b.t.toFixed(2), hit.length]];
  const max = Math.max(1, seq[0][1]);
  el('funnel').innerHTML = seq.map(([lab,n]) =>
    `<div class="frow"><span class="lab">${lab}</span>`
    + `<span><span class="bar" style="display:block;width:${100*n/max}%"></span></span>`
    + `<span class="n">${n}</span></div>`).join('');
  drawCube();
  drawAccounts();
  updateReach();
}

/* ---------- step 3: density, deficit, gaps ---------- */
function marginal(a, b, n, sigma){
  const P = full(), g = [], s2 = 2*sigma*sigma;
  const lin = Array.from({length:n}, (_,i)=>i/(n-1));
  let wsum = 0; P.forEach(p => wsum += p.w);
  const grid = Array.from({length:n}, ()=>new Float64Array(n));
  P.forEach(p => { for (let i=0;i<n;i++){ const da=(lin[i]-p[a])**2;
    for (let j=0;j<n;j++){ const db=(lin[j]-p[b])**2;
      grid[i][j] += p.w*Math.exp(-(da+db)/s2); } } });
  let mx = 0; grid.forEach(r=>r.forEach(v=>{ if(v>mx) mx=v; }));
  const def = grid.map(r=>Array.from(r, v => 1 - v/mx));
  return {lin, def};
}
function feasibleCell(u){
  return u.y <= u.x + 1/3 + 1e-9 && u.x <= u.y + 0.5 + 1e-9
      && !(u.z < 0.05 && u.t > 0.05);
}
function findGaps(sigma){
  const P = full(), n = D.cfg.grid, s2 = 2*sigma*sigma, sr2 = 2*D.cfg.sigmaReach**2;
  const lin = Array.from({length:n}, (_,i)=>i/(n-1));
  const cells = [];
  for (const x of lin) for (const y of lin) for (const z of lin) for (const t of lin) {
    const u = {x,y,z,t};
    if (!feasibleCell(u)) continue;
    let p = 0, dmin = Infinity;
    for (const q of P) {
      const d = (x-q.x)**2 + (y-q.y)**2 + (z-q.z)**2 + (t-q.t)**2;
      p += q.w*Math.exp(-d/s2); if (d < dmin) dmin = d;
    }
    cells.push({x,y,z,t,p,dist:Math.sqrt(dmin),reach:Math.exp(-dmin/sr2)});
  }
  let mx = 0; cells.forEach(c => { if (c.p > mx) mx = c.p; });
  const empty = cells.filter(c => 1 - c.p/mx >= D.cfg.emptyDeficit);
  const pick = (arr, k) => {
    const out = [];
    for (const c of arr) {
      if (out.some(o => Math.hypot(o.x-c.x,o.y-c.y,o.z-c.z,o.t-c.t) < 0.25)) continue;
      out.push(c); if (out.length === k) break;
    }
    return out;
  };
  return [...pick([...empty].sort((a,b)=>b.reach-a.reach), 2).map(c=>({...c,kind:'frontier'})),
          ...pick([...empty].sort((a,b)=>a.reach-b.reach), 2).map(c=>({...c,kind:'island'}))];
}
function drawDensity(){
  const [a,b] = state.plane, n = 70;
  const {lin, def} = marginal(a, b, n, state.sigma);
  const traces = [{ type:'contour', x:lin, y:lin, z:def[0].map((_,j)=>def.map(r=>r[j])),
    colorscale:'RdYlBu', reversescale:true, zmin:0, zmax:1,
    contours:{start:0, end:1, size:0.1}, colorbar:{title:{text:'coverage deficit',
    font:{size:11}}, thickness:11, len:0.85}, hoverinfo:'skip' }];
  if (state.showPts) {
    const P = full();
    traces.push({type:'scatter', mode:'markers', showlegend:false,
      x:P.map(p=>p[a]), y:P.map(p=>p[b]), text:P.map(p=>p.key),
      hovertemplate:'%{text}<extra></extra>',
      marker:{size:5, color:P.map(p=>COL[p.cluster]), line:{width:0.8, color:'white'}}});
  }
  if (state.showGaps) {
    const g = findGaps(state.sigma);
    ['frontier','island'].forEach(kind => {
      const s = g.filter(v=>v.kind===kind);
      traces.push({type:'scatter', mode:'markers', name:kind+' gap',
        x:s.map(v=>v[a]), y:s.map(v=>v[b]),
        text:s.map(v=>`${kind} gap<br>x ${fmt(v.x)} · y ${fmt(v.y)} · z ${fmt(v.z)} · t ${fmt(v.t)}`
          +`<br>reachability ${v.reach.toFixed(3)}`),
        hovertemplate:'%{text}<extra></extra>',
        marker:{symbol: kind==='frontier'?'circle-open':'star', size: kind==='frontier'?13:15,
                color:'#3B2E80', line:{width:2, color:'#3B2E80'}}});
    });
  }
  Plotly.react('dens', traces, {
    margin:{l:52,r:10,t:10,b:46}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'white',
    xaxis:{title:{text:D.axes[a].label, font:{size:12}}, range:[0,1], constrain:'domain'},
    yaxis:{title:{text:D.axes[b].label, font:{size:12}}, range:[0,1],
           scaleanchor:'x', scaleratio:1},
    legend:{orientation:'h', y:1.06, x:0, font:{size:11}}
  }, {displayModeBar:false, responsive:true});
}
function updateReach(){
  const c = boxCentre(state.box);
  const P = full().filter(p => p.cluster !== 'thesis');
  let best = Infinity, who = '—';
  P.forEach(p => { const d = (c.x-p.x)**2+(c.y-p.y)**2+(c.z-p.z)**2+(c.t-p.t)**2;
    if (d < best) { best = d; who = p.key; } });
  const reach = Math.exp(-best/(2*D.cfg.sigmaReach**2));
  el('reach').textContent = reach.toFixed(3);
  el('reachnote').textContent = 'nearest published: ' + who
    + ' at ' + Math.sqrt(best).toFixed(2);
}

/* ---------- step 4: accounts ---------- */
function drawAccounts(){
  const c = boxCentre(state.box), s2 = 2*state.sigma*state.sigma;
  const P = full().filter(p => p.acc && p.cluster !== 'thesis');
  const v = {}; D.accounts.forEach(a => v[a] = 0);
  let tot = 0;
  P.forEach(p => { const d = (c.x-p.x)**2+(c.y-p.y)**2+(c.z-p.z)**2+(c.t-p.t)**2;
    const k = p.w*Math.exp(-d/s2); if (v[p.acc] !== undefined) v[p.acc] += k; tot += k; });
  const keys = D.accounts.filter(a => tot > 0 && v[a]/tot > 0.004);
  Plotly.react('facc', [{type:'bar', orientation:'h', x:keys.map(a=>v[a]/tot),
      y:keys, marker:{color:'#7A5EA8'}, hovertemplate:'%{y}: %{x:.2f}<extra></extra>'}], {
    margin:{l:46,r:14,t:8,b:34}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'white',
    xaxis:{title:{text:'P(account | design at box centre)', font:{size:11}}, range:[0,1]}
  }, {displayModeBar:false, responsive:true});

  const mass = {}; D.accounts.forEach(a => mass[a] = 0); let mt = 0;
  pool().forEach(p => { if (p.acc && mass[p.acc] !== undefined) { mass[p.acc] += p.w; mt += p.w; } });
  const mk = D.accounts.filter(a => mt > 0 && mass[a]/mt > 0.001);
  Plotly.react('push', [{type:'bar', x:mk, y:mk.map(a=>mass[a]/mt),
      marker:{color:'#7A5EA8'}, hovertemplate:'%{x}: %{y:.2f}<extra></extra>'}], {
    margin:{l:46,r:14,t:8,b:34}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'white',
    yaxis:{title:{text:'share of corpus mass', font:{size:11}}}
  }, {displayModeBar:false, responsive:true});
}

/* ---------- step 5: found clusters against the hand labels ----------
   Everything here is drawn from the partition the script already computed, so the
   drop-lo and hide-thesis toggles above deliberately do not touch it: those change
   which rows are displayed, and changing which rows are clustered would be a
   different analysis with different centroids. */
function fitPts(){ return FIT ? D.pts.filter(p => p.found !== null) : []; }
function fitSelected(p){
  const f = state.fit;
  if (f.given !== 'all' && p.cluster !== f.given) return false;
  if (f.found !== 'all' && p.found !== +f.found) return false;
  if (f.disagree && agreeKey(p) !== 'disagree') return false;
  return true;
}
function fitScatter(div, colourOf, title){
  const [a,b] = state.fit.plane;
  const P = fitPts().filter(p => p[a] !== null && p[b] !== null);
  const on = P.filter(fitSelected), off = P.filter(p => !fitSelected(p));
  const xy = arr => spread(arr, a, b, a);
  const hover = p => `<b>${p.key}</b><br>${p.title}<br>`
    + `<span style="font-family:monospace">${a} ${fmt(p[a])} · ${b} ${fmt(p[b])}</span>`
    + `<br>label ${p.cluster} · found c${p.found}`
    + `<br>${agreeKey(p) === 'agree' ? 'agrees with the label' : 'differs from the label'}`;
  const traces = [];
  if (off.length) {
    const o = xy(off);
    traces.push({type:'scatter', mode:'markers', showlegend:false, hoverinfo:'skip',
      x:o.map(v=>v.x), y:o.map(v=>v.y),
      marker:{size:5, color:'#DCE3E6', line:{width:0.5, color:'white'}}});
  }
  const o = xy(on);
  traces.push({type:'scatter', mode:'markers', showlegend:false,
    x:o.map(v=>v.x), y:o.map(v=>v.y), text:on.map(hover),
    hovertemplate:'%{text}<extra></extra>',
    marker:{size:8, color:on.map(colourOf), line:{width:0.8, color:'white'}}});
  Plotly.react(div, traces, {
    margin:{l:46,r:12,t:8,b:42}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'white',
    xaxis:{title:{text:D.axes[a].label, font:{size:11}}, range:[-0.05,1.05]},
    yaxis:{title:{text:D.axes[b].label, font:{size:11}}, range:[-0.05,1.05],
           scaleanchor:'x', scaleratio:1}
  }, {displayModeBar:false, responsive:true});
}
function drawFlow(){
  const gn = FIT.givenNames, fn = FIT.foundNames, M = FIT.contingency;
  const src = [], tgt = [], val = [], lc = [];
  gn.forEach((g,j) => fn.forEach((f,i) => {
    if (!M[i][j]) return;
    src.push(j); tgt.push(gn.length + i); val.push(M[i][j]);
    lc.push(rgba(COL[g] || '#8A9AA2', state.fit.given === 'all' || state.fit.given === g ? 0.42 : 0.08));
  }));
  Plotly.react('fflow', [{
    type:'sankey', orientation:'h', arrangement:'snap',
    node:{label: gn.map(g => (D.clusters[g] ? D.clusters[g].label : g)).concat(fn),
          color: gn.map(g => COL[g] || '#8A9AA2').concat(fn.map((_,i)=>cpal(i))),
          pad:16, thickness:14, line:{width:0}},
    link:{source:src, target:tgt, value:val, color:lc},
    textfont:{size:11}
  }], {margin:{l:6,r:6,t:6,b:6}, paper_bgcolor:'rgba(0,0,0,0)', font:{size:11}},
     {displayModeBar:false, responsive:true});
}
function drawCross(){
  const gn = FIT.givenNames, fn = FIT.foundNames, M = FIT.contingency, f = state.fit;
  let head = '<tr><th></th>' + gn.map(g =>
    `<th>${D.clusters[g] ? D.clusters[g].label : g}</th>`).join('') + '<th class="tot">all</th></tr>';
  const body = fn.map((name,i) => {
    const rowTot = M[i].reduce((a,b)=>a+b, 0);
    return `<tr><th class="rowhead" style="color:${cpal(i)}">${name}</th>`
      + gn.map((g,j) => {
          const on = (f.found === String(i) && f.given === g);
          const cls = (FIT.mapping[i] === g ? 'matched ' : '') + (M[i][j] ? '' : 'zero');
          return `<td class="${cls}"><button data-i="${i}" data-g="${g}" `
               + `data-on="${on}">${M[i][j]}</button></td>`;
        }).join('')
      + `<td class="tot">${rowTot}</td></tr>`;
  }).join('');
  const colTot = '<tr><th class="rowhead tot">all</th>'
    + gn.map((_,j) => `<td class="tot">${fn.reduce((a,_,i)=>a+M[i][j],0)}</td>`).join('')
    + `<td class="tot">${M.flat().reduce((a,b)=>a+b,0)}</td></tr>`;
  el('fcross').innerHTML = head + body + colTot;
  el('fcross').querySelectorAll('button').forEach(btn => btn.onclick = () => {
    const same = btn.dataset.on === 'true';
    state.fit.given = same ? 'all' : btn.dataset.g;
    state.fit.found = same ? 'all' : btn.dataset.i;
    el('fgiven').value = state.fit.given;
    el('ffound').value = state.fit.found;
    refreshFit();
  });
}
function refreshFit(){
  if (!FIT) return;
  fitScatter('ffoundplot', p => cpal(p.found));
  fitScatter('fgivenplot', p => COL[p.cluster] || '#8A9AA2');
  drawFlow();
  drawCross();

  const P = fitPts(), sel = P.filter(fitSelected);
  const nAgree = P.filter(p => agreeKey(p) === 'agree').length;
  el('fagree').textContent = (100*nAgree/Math.max(1,P.length)).toFixed(0) + '%';
  const p = FIT.pAri === null ? '' :
    (FIT.pAri < 0.001 ? ' (p < 0.001)' : ` (p = ${FIT.pAri.toFixed(3)})`);
  el('fstat').textContent = `${nAgree} of ${P.length} · ARI ${fmt(FIT.ari)}${p} · AMI ${fmt(FIT.ami)}`;

  el('flist').innerHTML = sel.slice(0, 80).map(q => {
    const ok = agreeKey(q) === 'agree';
    return `<div class="qrow fitrow" data-k="${q.id}">`
      + `<span class="k"><span class="dot" style="background:${COL[q.cluster]||'#8A9AA2'}"></span>${q.key}</span>`
      + `<span class="t">${q.title || ''}</span>`
      + `<span class="c">${q.cluster} → <span style="color:${cpal(q.found)}">c${q.found}</span></span>`
      + `<span class="ok" style="color:${ok?'#157F7F':'#C1425A'}">${ok?'✓':'✗'}</span></div>`;
  }).join('');
  el('flist').querySelectorAll('.qrow').forEach(rowEl => rowEl.onclick = () => {
    const q = D.pts.find(v => v.id === rowEl.dataset.k);
    if (!q) return;
    state.query = (q.key || '').toLowerCase();
    el('q').value = q.key; el('q').dispatchEvent(new Event('input'));
    document.getElementById('cube').scrollIntoView({behavior:'smooth', block:'center'});
  });
  el('fnote').textContent = sel.length
    ? `${sel.length} paradigm${sel.length===1?'':'s'} selected`
      + (sel.length > 80 ? ', first 80 listed' : '')
      + '. Click a row to find it in the cube above.'
    : 'Nothing matches this combination.';
}
function buildFitControls(){
  if (!FIT) { el('fitsec').style.display = 'none'; return; }
  el('fitmeta').textContent = `k = ${FIT.k} · ${FIT.method} · on ${FIT.axes.join(', ')}`;
  ['found','agree'].forEach(v => {
    const o = document.createElement('option');
    o.value = v;
    o.textContent = v === 'found' ? 'cluster found by the code' : 'label vs found cluster';
    el('colorby').appendChild(o);
  });
  const planes = [];
  for (let i=0;i<PRIN.length;i++) for (let j=i+1;j<PRIN.length;j++) planes.push([PRIN[i],PRIN[j]]);
  el('fplane').innerHTML = planes.map(([a,b]) =>
    `<option value="${a},${b}">${a} vs ${b}</option>`).join('');
  el('fplane').value = state.fit.plane.join(',');
  el('fplane').onchange = () => { state.fit.plane = el('fplane').value.split(','); refreshFit(); };

  el('fgiven').innerHTML = '<option value="all">all</option>' + FIT.givenNames.map(g =>
    `<option value="${g}">${D.clusters[g] ? D.clusters[g].label : g}</option>`).join('');
  el('ffound').innerHTML = '<option value="all">all</option>' + FIT.foundNames.map((n,i) =>
    `<option value="${i}">${n}${FIT.mapping[i] ? ' — matched to ' + FIT.mapping[i] : ''}</option>`).join('');
  el('fgiven').onchange = () => { state.fit.given = el('fgiven').value; refreshFit(); };
  el('ffound').onchange = () => { state.fit.found = el('ffound').value; refreshFit(); };
  el('fdis').onclick = () => {
    state.fit.disagree = !state.fit.disagree;
    el('fdis').setAttribute('aria-pressed', state.fit.disagree);
    refreshFit();
  };
  el('fswatch').innerHTML = FIT.foundNames.map((n,i) =>
      `<span class="s"><span class="dot" style="background:${cpal(i)}"></span>${n}`
      + `${FIT.mapping[i] ? ' → ' + FIT.mapping[i] : ''}</span>`).join('')
    + FIT.givenNames.map(g =>
      `<span class="s"><span class="dot" style="background:${COL[g]||'#8A9AA2'}"></span>`
      + `${D.clusters[g] ? D.clusters[g].label : g}</span>`).join('');
}

/* ---------- wiring ---------- */
function buildControls(){
  const opts = Object.keys(D.axes).map(a =>
    `<option value="${a}">${a} — ${D.axes[a].label}</option>`).join('');
  ['a1','a2','a3'].forEach((id,i) => { el(id).innerHTML = opts;
    el(id).value = state.axes[i];
    el(id).onchange = () => { state.axes[i] = el(id).value; drawCube(); }; });
  const planes = [];
  for (let i=0;i<PRIN.length;i++) for (let j=i+1;j<PRIN.length;j++) planes.push([PRIN[i],PRIN[j]]);
  el('plane').innerHTML = planes.map(([a,b]) =>
    `<option value="${a},${b}">${a} vs ${b}</option>`).join('');
  el('plane').value = state.plane.join(',');
  el('plane').onchange = () => { state.plane = el('plane').value.split(','); drawDensity(); };
  el('colorby').onchange = () => { state.colorby = el('colorby').value; drawCube(); };

  el('presets').innerHTML = Object.entries(D.regions).map(([k,r]) =>
    `<button class="seg" data-r="${k}">${k} · ${r.title}</button>`).join('')
    + '<button class="seg" data-r="reset">reset</button>';
  el('presets').querySelectorAll('button').forEach(btn => btn.onclick = () => {
    const k = btn.dataset.r;
    if (k === 'reset') state.box = {x:0, y:0, z:0, t:1};
    else {
      state.box = {x:0, y:0, z:0, t:1};
      D.regions[k].constraints.forEach(([a,op,v]) => state.box[a] = v);
    }
    ['x','y','z','t'].forEach(a => el('b'+a).value = state.box[a]);
    refreshBox();
  });

  ['x','y','z','t'].forEach(a => el('b'+a).addEventListener('input', () => {
    state.box[a] = +el('b'+a).value; refreshBox(); }));
  el('bs').addEventListener('input', () => {
    state.sigma = +el('bs').value; el('vs').textContent = state.sigma.toFixed(3);
    el('fsig').textContent = state.sigma.toFixed(3); });
  el('bs').addEventListener('change', () => { drawDensity(); drawAccounts(); });
  el('vs').textContent = state.sigma.toFixed(3);

  const toggle = (id, key, after) => el(id).onclick = () => {
    state[key] = !state[key];
    el(id).setAttribute('aria-pressed', state[key]);
    after();
  };
  toggle('tlo','dropLo', () => { refreshBox(); drawDensity(); });
  toggle('tth','hideThesis', () => { refreshBox(); drawDensity(); });
  el('tgap').onclick = () => { state.showGaps = !state.showGaps;
    el('tgap').setAttribute('aria-pressed', state.showGaps); drawDensity(); };
  el('tpts').onclick = () => { state.showPts = !state.showPts;
    el('tpts').setAttribute('aria-pressed', state.showPts); drawDensity(); };

  const q = el('q');
  const runSearch = () => {
    state.query = q.value.trim().toLowerCase();
    const hits = state.query ? pool().filter(matches) : [];
    el('qn').textContent = state.query
      ? `${hits.length} of ${pool().length} match` : '';
    el('qlist').innerHTML = hits.slice(0, 60).map(p => {
      const col = COL[p.cluster] || '#6C7C85';
      const co = PRIN.map(a => `${a} ${fmt(p[a])}`).join(' · ');
      return `<div class="qrow" data-k="${p.id}">`
        + `<span class="k"><span class="dot" style="background:${col}"></span>${p.key}</span>`
        + `<span class="t">${p.title || ''}</span>`
        + `<span class="c">${co} · ${p.conf}</span></div>`;
    }).join('');
    el('qlist').querySelectorAll('.qrow').forEach(rowEl => rowEl.onclick = () => {
      const p = D.pts.find(v => v.id === rowEl.dataset.k);
      if (!p) return;
      // walk the box out to this paradigm, so you can see which constraint excludes it
      ['x','y','z'].forEach(a => { if (p[a] !== null) state.box[a] = Math.min(state.box[a], p[a]); });
      if (p.t !== null) state.box.t = Math.max(state.box.t, p.t);
      ['x','y','z','t'].forEach(a => el('b'+a).value = state.box[a]);
      refreshBox();
    });
    drawCube();
  };
  q.addEventListener('input', runSearch);

  const rows = D.audit.dispositions.map(([k,v]) =>
    `<tr><td>${k}</td><td class="num">${v}</td></tr>`).join('');
  el('audit').innerHTML = '<tr><th>disposition</th><th>records</th></tr>' + rows
    + `<tr><td>scored on all four principal axes</td><td class="num">${full().length}</td></tr>`
    + `<tr><td>scores off the rung lattice (rule R7)</td>`
    + `<td class="num">${D.audit.offLattice.length}</td></tr>`;
}
buildControls();
buildFitControls();
refreshBox();
drawDensity();
refreshFit();
</script>
</body></html>
"""


def write_html(path, df, ladders, cfg, gaps, raw, source, inline=None, clus=None):
    js, offline = plotly_bundle()
    if inline is False:
        js, offline = ('<script src="https://cdn.plot.ly/plotly-3.0.0.min.js" '
                       'charset="utf-8"></script>', False)
    payload = html_payload(df, ladders, cfg, gaps, raw, clus)
    thesis = df[df["thesis"]]
    coord = " · ".join(
        f"{a} {thesis[a].mean():.2f}" for a in PRINCIPAL
    ) if len(thesis) else "no thesis rows in the workbook"
    page = (HTML
            .replace("__PLOTLY__", js)
            .replace("__DATA__", json.dumps(payload, allow_nan=False))
            .replace("__NPARA__", str(len(df)))
            .replace("__NRECORDS__", str(len(raw)))
            .replace("__THESISCOORD__", coord)
            .replace("__SOURCE__", Path(source).name)
            .replace("__SIGR__", f"{cfg.sigma_reach:g}")
            .replace("__WHI__", f"{WEIGHT['hi']:g}")
            .replace("__WMD__", f"{WEIGHT['md']:g}")
            .replace("__WLO__", f"{WEIGHT['lo']:g}"))
    Path(path).write_text(page, encoding="utf-8")
    return offline


# ---------------------------------------------------------------------------
# 6. report and captions
# ---------------------------------------------------------------------------

def sensitivity(df, cfg, sigmas=(0.06, 0.09, 0.12, 0.15)):
    rows = []
    for s in sigmas:
        c = Config(sigma=s, sigma_reach=cfg.sigma_reach, grid=cfg.grid, axes=cfg.axes)
        g, _ = gap_search(df, c, k=1)
        front = g[g["kind"] == "frontier"]
        row = dict(sigma=s,
                   G1=len(region_members(df, "G1")),
                   G2=len(region_members(df, "G2")))
        if len(front):
            row.update({f"frontier_{a}": round(front.iloc[0][a], 2) for a in cfg.axes})
        rows.append(row)
    return pd.DataFrame(rows)


def report(df, raw, cfg, gaps, counts, mean_disp, f_thesis, n_eff, corr, path, clus=None):
    sub = df.dropna(subset=PRINCIPAL)
    P, w, pub = matrix(df[~df["thesis"]], cfg.axes)
    lines = []
    add = lines.append
    add("paradigm space — every number quoted in the prose")
    add("=" * 64)
    add(f"records in the workbook            {len(raw)}")
    add(f"scorable paradigms                 {len(df)}")
    add(f"scored on all four principal axes  {len(sub)}")
    add(f"confidence hi/md/lo                "
        f"{(df.confidence == 'hi').sum()}/{(df.confidence == 'md').sum()}"
        f"/{(df.confidence == 'lo').sum()}")
    add(f"clusters                           "
        + ", ".join(f"{k}:{int((df.cluster == k).sum())}" for k in CLUSTERS))
    add("")
    add("regions")
    for name, spec in REGIONS.items():
        cons = " ∧ ".join(f"{a} {'≥' if o == '>=' else '≤'} {v:g}"
                          for a, o, v in spec["constraints"])
        add(f"  {name}: {cons}")
        add(f"    occupancy {counts[name]} of {len(sub)}")
        for lab, n in funnel(sub, spec["constraints"]):
            add(f"      {lab:<22} {n}")
        m = region_members(df, name)
        if len(m):
            add("      members: " + ", ".join(sorted(set(m['paradigm_id']))))
        c = centroid(name, cfg.axes)
        rc, dist, j = reachability(c, P, cfg.sigma_reach)
        add(f"      centroid reachability {rc[0]:.3f} at distance {dist[0]:.3f} "
            f"from {pub.iloc[j[0]]['paradigm_id']}")
    add("")
    thesis = df[df["thesis"]].dropna(subset=cfg.axes)
    if len(thesis):
        u = thesis[cfg.axes].mean().to_numpy()
        rc, dist, j = reachability(u, P, cfg.sigma_reach)
        add(f"thesis paradigm at ({', '.join(f'{v:.2f}' for v in u)})")
        add(f"  reachability against the published corpus {rc[0]:.3f} "
            f"at distance {dist[0]:.3f}")
        add(f"  nearest published paradigm {pub.iloc[j[0]]['paradigm_id']} "
            f"({pub.iloc[j[0]]['title'][:50]})")
        add("  f(u) = " + ", ".join(f"P({k})={v:.2f}"
                                    for k, v in f_thesis.items() if v > 0.005))
        add(f"  the kernel regression rests on an effective {n_eff:.1f} rows")
        add("  per configuration (the mean above hides the spread):")
        rc, dist, j = reachability(thesis[cfg.axes].to_numpy(float), P, cfg.sigma_reach)
        for pid, r_, d_, k in zip(thesis["paradigm_id"], rc, dist, j):
            add(f"    {pid:<18} reach {r_:.3f} at {d_:.3f} "
                f"from {pub.iloc[k]['paradigm_id']}")
    add("")
    add("gaps (searched in 4D, separately among the uncovered cells)")
    for _, g in gaps.iterrows():
        add(f"  {g['kind']:<9} ({', '.join(f'{g[a]:.2f}' for a in cfg.axes)})"
            f"  reach {g['reach']:.3f}")
    add("")
    add(f"mean motor → non-motor displacement {mean_disp:.2f}")
    add("axis collinearity above 0.5:")
    for i, a in enumerate(ALL_AXES):
        for b in ALL_AXES[i + 1:]:
            v = corr.loc[a, b]
            if np.isfinite(v) and abs(v) >= 0.5:
                add(f"  corr({a}, {b}) = {v:+.2f}")
    coord = sub[PRINCIPAL].round(2).astype(str).agg("|".join, axis=1)
    per_topic = sub.assign(c=coord).groupby("topic")["c"].nunique()
    add("")
    add(f"label determinism: {coord.nunique()} distinct coordinates over {len(sub)} rows; "
        f"{int((per_topic == 1).sum())} of {len(per_topic)} topics at a single point")
    off = df[df["off_lattice"]]
    if len(off):
        add(f"off-lattice scores (rule R7): {', '.join(off['paradigm_id'])}")
    dup = df[df.duplicated("citekey", keep=False) & df["citekey"].astype(bool)]
    if len(dup):
        add("papers occupying more than one row (expected where a paper runs several "
            "paradigms; check the rest):")
        for k, n in dup["citekey"].value_counts().items():
            add(f"  {k}: {n}")
    if clus is not None:
        add("")
        add(f"unsupervised clustering ({clus['method']}, k = {clus['k']}, on "
            f"{', '.join(clus['axes'])}, n = {len(clus['sub'])})")
        add(f"  ARI {clus['ari']:.3f} (permutation p = {clus['p_ari']:.4f}), "
            f"AMI {clus['ami']:.3f}, best one-to-one agreement {clus['agreement']:.1%}")
        add(f"  silhouette {clus['silhouette']:.3f}, bootstrap stability "
            f"{clus['stability']:.3f}")
        for j, c in enumerate(clus["centers"]):
            n = int((clus["labels"] == j).sum())
            add(f"  c{j} (n = {n}): " +
                ", ".join(f"{a} {v:.2f}" for a, v in zip(clus["axes"], c)))
        M = clus["contingency"]
        add("  " + "found".ljust(8) + "".join(g[:9].rjust(10) for g in clus["given_names"]))
        for i, row_ in enumerate(M):
            add("  " + f"c{i}".ljust(8) + "".join(str(v).rjust(10) for v in row_))
        add("  k sweep:")
        add("    " + clus["curve"].round(3).to_string(index=False).replace("\n", "\n    "))
    add("")
    add("sensitivity to the bandwidth")
    add(sensitivity(df, cfg).to_string(index=False))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def _tex(md):
    """Minimal markdown -> LaTeX. The prose below is written to stay inside this."""
    out = []
    for line in md.split("\n"):
        if line.startswith("### "):
            out.append(r"\subsubsection{" + line[4:] + "}")
        elif line.startswith("## "):
            out.append(r"\subsection{" + line[3:] + "}")
        elif line.startswith("# "):
            out.append(r"\section{" + line[2:] + "}")
        elif line.startswith("- "):
            out.append(r"\item " + line[2:])
        else:
            out.append(line)
    tex = "\n".join(out)
    # wrap runs of \item in itemize
    tex = re.sub(r"((?:^\\item .*\n?)+)", lambda m: "\\begin{itemize}\n" + m.group(1) +
                 "\\end{itemize}\n", tex, flags=re.M)
    tex = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", tex)
    tex = re.sub(r"`(.+?)`", r"\\texttt{\1}", tex)
    tex = tex.replace("%", r"\%").replace("&", r"\&")
    tex = tex.replace("≥", r"$\geq$").replace("≤", r"$\leq$").replace("→", r"$\to$")
    tex = tex.replace("·", r"$\cdot$").replace("—", "---").replace("σ", r"$\sigma$")
    tex = re.sub(r"(?<![\\{])_", r"\\_", tex)
    return tex


def write_results(path_md, path_tex, df, raw, cfg, gaps, counts, disp, f_thesis,
                  n_eff, corr, ladders, source, clus=None):
    """A results section with every number substituted, in markdown and in LaTeX.

    The point of generating the prose rather than typing it is that re-running after
    an edit to the workbook cannot leave the text disagreeing with the figures.
    """
    sub = df.dropna(subset=PRINCIPAL)
    P, w, pub = matrix(df[~df["thesis"]], cfg.axes)
    n_paper = df["citekey"].replace("", np.nan).nunique()

    def fun(name):
        return " → ".join(str(n) for _, n in funnel(sub, REGIONS[name]["constraints"]))

    thesis = df[df["thesis"]].dropna(subset=cfg.axes)
    tline = ""
    if len(thesis):
        rc, dist, j = reachability(thesis[cfg.axes].to_numpy(float), P, cfg.sigma_reach)
        u = thesis[cfg.axes].mean().to_numpy()
        rcm, distm, jm = reachability(u, P, cfg.sigma_reach)
        span = (f"is {rc.min():.2f}" if rc.max() - rc.min() < 0.005
                else f"runs from {rc.min():.2f} to {rc.max():.2f}")
        tline = (f"The {len(thesis)} configurations of the thesis experiment sit at "
                 f"({', '.join(f'{v:.2f}' for v in u)}) on average. Measured one "
                 f"configuration at a time against the published corpus, reachability "
                 f"{span}, the nearest published neighbour "
                 f"in every case being {pub.iloc[j[0]]['citekey']}; measured as a single "
                 f"averaged point it is {rcm[0]:.2f} at a distance of {distm[0]:.2f}. "
                 f"Either way this is a frontier, not an island: a short and natural step "
                 f"from a paradigm the field already runs.")

    cent = {n: reachability(centroid(n, cfg.axes), P, cfg.sigma_reach) for n in REGIONS}

    def verdict(name):
        """Interpretation has to track the count, or re-running silently lies."""
        n = counts[name]
        if n == 0:
            return ("is empty on the present axes. Emptiness is a claim the corpus can "
                    "refute, and on this pass it does not")
        if n <= 3:
            return (f"holds {n} paradigm" + ("s" if n > 1 else "") +
                    ", which is thin enough that the region stands or falls on how those "
                    "rows were calibrated")
        return (f"is occupied: {n} paradigms fall inside it, so it is a frontier rather "
                "than an island")
    fr = gaps[gaps["kind"] == "frontier"]
    isl = gaps[gaps["kind"] == "island"]
    g_txt = "; ".join(
        f"({', '.join(f'{g[a]:.2f}' for a in cfg.axes)}) at reachability {g['reach']:.2f}"
        for _, g in fr.iterrows())
    i_txt = "; ".join(
        f"({', '.join(f'{g[a]:.2f}' for a in cfg.axes)})" for _, g in isl.iterrows())

    hi_corr = [(a, b, corr.loc[a, b]) for i, a in enumerate(ALL_AXES)
               for b in ALL_AXES[i + 1:]
               if np.isfinite(corr.loc[a, b]) and abs(corr.loc[a, b]) >= 0.5]
    coord = sub[PRINCIPAL].round(2).astype(str).agg("|".join, axis=1)
    per_topic = sub.assign(c=coord).groupby("topic")["c"].nunique()
    acc = ", ".join(f"P({k}) = {v:.2f}" for k, v in
                    sorted(f_thesis.items(), key=lambda kv: -kv[1]) if v > 0.02)
    push = pushforward(df)
    top_push = ", ".join(f"{k} {v:.2f}" for k, v in
                         sorted(push.items(), key=lambda kv: -kv[1])[:4])

    if clus is None:
        clus_txt = ("This run was made with clustering switched off; pass `-k 2` to "
                    "reproduce the check.")
    else:
        M, names = clus["contingency"], list(clus["given_names"])
        split = []
        for j, g in enumerate(names):
            col = M[:, j]
            if col.sum() and col.max() / col.sum() < 0.8:
                where = ", ".join(f"{v} in c{i}" for i, v in enumerate(col) if v)
                split.append(f"**{g}** is divided ({where})")
        rows = []
        for j, c in enumerate(clus["centers"]):
            n = int((clus["labels"] == j).sum())
            rows.append(f"c{j} (n = {n}) at " +
                        ", ".join(f"{a} = {v:.2f}" for a, v in zip(clus["axes"], c)))
        pstr = ("p < 0.001" if clus["p_ari"] < 0.001 else f"p = {clus['p_ari']:.3f}")
        strength = ("strong" if clus["ari"] > 0.6 else
                    "partial" if clus["ari"] > 0.3 else "weak")
        best_k = int(clus["curve"].loc[clus["curve"]["silhouette"].idxmax(), "k"])
        k_note = ("which is also the $k$ the silhouette prefers"
                  if best_k == clus["k"] else
                  f"though the silhouette peaks at k = {best_k} rather than "
                  f"{clus['k']}, so the two-literature reading is a choice about how "
                  f"coarsely to look rather than something the data insists on")
        clus_txt = (
            f"Clustering the {len(clus['sub'])} fully scored paradigms on "
            f"{', '.join(clus['axes'])} with weighted {clus['method']}, blind to the labels "
            f"and with each row weighted by its scoring confidence, gives k = {clus['k']} "
            f"groups: {'; '.join(rows)} (`fig8_clusters`). Agreement with the hand labels is "
            f"{strength}: the adjusted Rand index is {clus['ari']:.2f} "
            f"(permutation {pstr}), the adjusted mutual information "
            f"{clus['ami']:.2f}, and the best one-to-one match puts "
            f"{clus['agreement']:.0%} of rows in the corresponding group. The partition is "
            f"{'stable' if clus['stability'] > 0.75 else 'not especially stable'} under "
            f"resampling ({clus['stability']:.2f} mean pairwise co-assignment) with a mean "
            f"silhouette of {clus['silhouette']:.2f}, {k_note}. "
            + (("Read against the ladders, the split the corpus makes is "
                "not the split the labels make: " + "; ".join(split) + ". ")
               if split else "Every hand label maps cleanly onto one discovered group. ")
            + "The distinction to keep in view is which axes carry the separation — a "
              "partition that differs between groups on task difficulty but not on task "
              "relevance is telling you the corpus is organised by engagement rather than "
              "by what the event means, whatever the labels are called.")

    md = f"""# The paradigm space: results

Generated from `{Path(source).name}` by `paradigm_space.py`. Every number below is
computed at build time, so the prose cannot drift from the figures.

## The corpus

The database holds {len(raw)} records. Applying the decision rules leaves
**{len(df)} scorable paradigms** drawn from {n_paper} distinct citation keys — one row is one
experimental design, so a paper running several experiments contributes several points —
of which **{len(sub)} carry a value on all four principal axes**. Confidence under rule R8 is
{(df.confidence == 'hi').sum()} hi, {(df.confidence == 'md').sum()} md and
{(df.confidence == 'lo').sum()} lo; re-running every occupancy claim without the inferred
rows is what rule R8 asks for, and the `--drop-lo` flag does it.

Figure `fig0_space` states what the coordinates mean before any occupancy claim is made: the
rung ladder of each principal axis, how many paradigms sit on each rung, and the slab that
each candidate region cuts out of it. Read in these coordinates the corpus separates into
the two literatures the review is about: continuous, task-relevant control at high {SHORT['x']},
high {SHORT['y']} and {SHORT['t']} → 1, and passive or discrete-response stimulation at low
{SHORT['x']}, low {SHORT['y']} and {SHORT['t']} → 0 (`fig1_projections`). Splitting the task
axes in two was not cosmetic: the displacement between a paradigm's motor position and its
non-motor position averages **{disp:.2f}**, so the second pair carries information the first
does not (`fig4_task_axes`).

## Region occupancy

A region is a conjunction of constraints on three axes, so no two-dimensional picture can
settle whether a paradigm is inside it: a box drawn on a plane is the region's shadow, and a
point inside the shadow may be excluded by the constraint that plane does not carry. That
failure mode is what made an earlier version of this analysis read as emptier than the data.
Occupancy is therefore reported as a funnel, and drawn in `fig2_regions` with the off-plane
constraints applied, and in `fig3_cube` on the axis triple that carries every constraint.

**G1** — engaging control ({SHORT['x']} ≥ 0.67), a command still open to revision
({SHORT['y']} ≥ 0.57), and an event carrying no task information ({SHORT['t']} ≤ 0.17) —
{verdict('G1')}. The funnel runs {fun('G1')}: **{counts['G1']} of {len(sub)}** paradigms
survive. The centroid of the box has a reachability of {cent['G1'][0][0]:.2f} against the
published corpus.

**G2** — the same, with a surprise horizon deeper than a fixed-probability deviant
({SHORT['z']} ≥ 0.5) — {verdict('G2')}. The funnel runs {fun('G2')}: **{counts['G2']} of
{len(sub)}**. Its centroid has a reachability of {cent['G2'][0][0]:.2f}, at a distance of
{cent['G2'][1][0]:.2f} from its nearest published neighbour. """ + \
("""Where a design in that region is task-relevant it exists; the moment the event is made
irrelevant, it disappears. The geometry recovers, from the corpus rather than from assertion,
the claim that the statistical horizon of the unexpected event is never manipulated in the
presence of ongoing control — which is exactly the design feature on which the
precision-gating and global-suppression accounts differ.""" if counts["G2"] == 0 else
"""The region is no longer empty, so the claim that the statistical horizon of the unexpected
event is never manipulated in the presence of ongoing control no longer follows from the
geometry and should be restated against the rows that now occupy it.""") + f"""

{tline}

## Where the corpus is not

Smoothing the corpus into a density over the space and searching separately among the cells
it does not cover distinguishes two classes of absence (`fig5_gaps`). The most reachable
uncovered designs — the smallest modification of existing work that lands somewhere new —
sit at {g_txt}. The least reachable feasible empty cells sit at {i_txt}, with reachability
below 0.01. Gaps are found in four dimensions and then projected, so a marker on any one
plane is a real four-dimensional cell rather than an artefact of that projection.

## Account space

A region can be paper-empty and still map onto a well-covered part of theory, in which case
filling it teaches nothing. Estimated by kernel regression over the scored corpus with the
thesis rows held out, the map from a design to a distribution over computational accounts
returns {acc} at the thesis paradigm (`fig6_accounts`). That estimate rests on an effective
{n_eff:.1f} rows, which is the number to quote alongside it. Across the corpus as a whole the
mass sits on {top_push}.

## Do the two literatures exist, or were they assumed?

The cluster label on each row was assigned by the reviewer. Whether the corpus separates
that way is a different question, and the geometry can answer it without being told the
answer. {clus_txt}

## What the representation costs

- **The axes are a choice.** The space is a projection, selected because it makes one
  distinction visible, and the apparent emptiness of a region is always relative to it.
  {len(hi_corr)} axis pairs remain correlated at 0.5 or above""" + \
(": " + ", ".join(f"{a}–{b} at {v:+.2f}" for a, b, v in hi_corr) if hi_corr else "") + f""".
- **The coordinates are not yet independent of the topic label.** The {len(sub)} scored rows
  occupy {coord.nunique()} distinct coordinates, and {int((per_topic == 1).sum())} of
  {len(per_topic)} topics have all their members at a single point (`fig7_audit`, panel C).
- **Absence in a corpus is not absence in a literature.** The smooth-pursuit and
  saccadic-inhibition work delivers transient task-irrelevant events during ongoing pursuit
  and is not yet ingested. Until it is, every statement here is a statement about this corpus.
- **The smoothing is reported, not fitted.** The bandwidth σ = {cfg.sigma:g} is a choice; the
  sensitivity table in `scoring_report.txt` re-runs the occupancy and the frontier gaps across
  σ in [0.06, 0.15].
"""

    # two tables of the numbers the prose quotes, so a reader can check them
    rows = []
    for name, spec in REGIONS.items():
        steps = funnel(sub, spec["constraints"])
        cons = ", ".join(f"${k} \\{'geq' if op == '>=' else 'leq'} {v:g}$"
                         for k, op, v in spec["constraints"])
        rows.append(f"{name} & {cons} & " +
                    " & ".join(str(n) for _, n in steps) +
                    f" & {cent[name][0][0]:.2f} \\\\")
    tbl_regions = "\n".join([
        r"\begin{table}[tbp]", r"  \centering", r"  \small",
        r"  \begin{tabular}{llrrrrr}", r"    \hline",
        r"    Region & Constraints & all & +1 & +2 & +3 & reach \\", r"    \hline",
        *[f"    {r}" for r in rows], r"    \hline", r"  \end{tabular}",
        r"  \caption{Region occupancy as a constraint funnel. Each column applies one more "
        r"constraint, in the order listed; the last count is the occupancy of the region. "
        r"\emph{reach} is the reachability of the region centroid against the published "
        r"corpus.}", r"  \label{tab:regions}", r"\end{table}"])

    grows = [f"{g['kind']} & " + " & ".join(f"{g[a]:.2f}" for a in cfg.axes) +
             f" & {g['reach']:.3f} \\\\" for _, g in gaps.iterrows()]
    tbl_gaps = "\n".join([
        r"\begin{table}[tbp]", r"  \centering", r"  \small",
        r"  \begin{tabular}{lrrrrr}", r"    \hline",
        r"    Class & $" + "$ & $".join(cfg.axes) + r"$ & reachability \\", r"    \hline",
        *[f"    {r}" for r in grows], r"    \hline", r"  \end{tabular}",
        r"  \caption{The two classes of absence, searched separately among the cells the "
        r"corpus does not cover. A frontier gap is the smallest modification of an existing "
        r"design that lands somewhere new; an island gap needs a new paradigm.}",
        r"  \label{tab:gaps}", r"\end{table}"])

    Path(path_md).write_text(md, encoding="utf-8")
    body = _tex(md)
    figs = []
    for name, (short, long) in CAPTIONS.items():
        text = long
        for k, v in dict(n=len(sub), disp=f"{disp:.2f}", records=len(raw),
                         funnel="; ".join(f"{n}: {fun(n)}" for n in REGIONS)).items():
            text = text.replace("{" + k + "}", str(v))
        figs += [r"\begin{figure}[tbp]", r"  \centering",
                 rf"  \includegraphics[width=\textwidth]{{{name}.pdf}}",
                 rf"  \caption[{short}]{{\textbf{{{short}.}} {text}}}",
                 rf"  \label{{fig:{name}}}", r"\end{figure}", ""]
    fragment = ("% generated by paradigm_space.py -- do not edit, re-run instead\n"
                "% requires \\usepackage{graphicx}; \\graphicspath{{figs/}} assumed\n\n"
                + body + "\n\n" + tbl_regions + "\n\n" + tbl_gaps + "\n\n"
                + "\n".join(figs))
    Path(path_tex).write_text(fragment, encoding="utf-8")
    return fragment


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\IfFileExists{lmodern.sty}{\usepackage{lmodern}}{}
\usepackage[margin=2.6cm]{geometry}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\IfFileExists{booktabs.sty}{\usepackage{booktabs}}{}
\IfFileExists{hyperref.sty}{\usepackage[hidelinks]{hyperref}}{}
\IfFileExists{caption.sty}{\usepackage{caption}
\captionsetup{font=small,labelfont=bf}}{}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
\graphicspath{{./}}
\title{%(title)s}
\author{Generated by \texttt{paradigm\_space.py} from \texttt{%(source)s}}
\date{%(date)s}
\begin{document}
\maketitle
"""


def write_report(path, fragment, source, title="The experimental paradigm space"):
    """A standalone document: same body, plus a preamble so it compiles on its own.

    The fragment is what goes into the thesis; this is what to look at while iterating,
    since it can be built without touching the thesis project.
    """
    from datetime import date
    head = PREAMBLE % dict(title=title, source=Path(source).name.replace("_", r"\_"),
                           date=date.today().isoformat())
    body = "\n".join(l for l in fragment.split("\n") if not l.startswith("%"))
    Path(path).write_text(head + body + "\n\\end{document}\n", encoding="utf-8")


def compile_pdf(tex_path):
    """Build the standalone report if a LaTeX toolchain is on PATH. Optional by design."""
    import shutil
    import subprocess
    exe = shutil.which("pdflatex")
    if not exe:
        return None
    tex = Path(tex_path)
    for _ in range(2):                       # twice, so refs and captions settle
        r = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error",
                            tex.name], cwd=tex.parent, capture_output=True, text=True)
    pdf = tex.with_suffix(".pdf")
    if not pdf.exists():
        tail = "\n".join(r.stdout.strip().split("\n")[-12:])
        print(f"pdflatex failed; the .tex is still usable:\n{tail}")
        return None
    for ext in (".aux", ".log", ".out"):
        tex.with_suffix(ext).unlink(missing_ok=True)
    return pdf


CAPTIONS = {
    "fig0_space": (
        "The design space",
        "A the rung ladder of each principal axis, with the number of paradigms sitting on "
        "each rung drawn as a bar and the slab each candidate region cuts shaded. B the "
        "corpus in three dimensions, with the same points cast onto the three walls as grey "
        "marginals and the candidate regions drawn as volumes. Every coordinate is a feature "
        "that can be read off a Methods section rather than inferred from a result."),
    "fig1_projections": (
        "The corpus in the six principal planes",
        "Colour is cluster, marker shape is scoring confidence (rule R8). Coincident "
        "paradigms are fanned onto a small spiral, always narrower than half a rung, so "
        "that a rung holding {n} paradigms does not look like a rung holding one. No "
        "candidate region is drawn here: a box on a plane is a slab in the space, and a "
        "point inside its shadow need not be inside the region — that is the failure mode "
        "which made the earlier version of this figure read as emptier than the data."),
    "fig2_regions": (
        "The candidate regions, tested",
        "One row per region. In the two scatter panels only the paradigms satisfying the "
        "constraints on the axes {\\em not} drawn are shown in colour; the rest are open grey "
        "markers. What lies inside the box on screen therefore lies inside the region in the "
        "space. The bars on the right add the constraints one at a time: {funnel}."),
    "fig3_cube": (
        "The paradigm cube, in the two triples that carry the constraints",
        "A $(x,y,t)$ carries every constraint of $G_1$ and B $(x,z,t)$ every constraint of "
        "$G_2$, so in each panel the shaded volume is the region rather than its shadow and "
        "a marker inside it is a member. Ringed markers are the members."),
    "fig4_task_axes": (
        "Splitting the task axes in two",
        "A the motor plane $(x,y)$, B the non-motor plane $(x_1,y_1)$, C the displacement "
        "between them for each paradigm. If the non-motor pair were redundant with the motor "
        "pair, C would be a field of near-zero arrows; the mean displacement is {disp}."),
    "fig5_gaps": (
        "Coverage deficit over the principal axes",
        "Smoothing the corpus into a density over $\\mathcal H$ and searching separately "
        "among the cells it does not cover distinguishes two classes of absence: designs "
        "that are empty but a short walk from existing work (frontier) and designs that are "
        "empty and isolated (island). Gaps are found in four dimensions and projected here."),
    "fig6_accounts": (
        "Mapping the corpus into computational-account space",
        "A the $h$ layer as a block-structured heat map. B the pushforward $f_\\#\\mu$: which "
        "accounts the corpus actually spends its mass on. C $f$ evaluated at the thesis "
        "paradigm by kernel regression over the scored corpus, with the thesis rows held out."),
    "fig7_audit": (
        "Audit of the scoring pass",
        "A what happened to each of the {records} records. B the confidence distribution over "
        "scored paradigms (rule R8). C how far the coordinates are still a function of the "
        "Topic label: a topic on the diagonal has one coordinate per paradigm, a topic at "
        "$y=1$ has all its members at a single point. D axis collinearity."),
    "fig8_clusters": (
        "Clustering the corpus without its labels",
        "A silhouette and bootstrap stability across $k$; the vertical line is the $k$ the "
        "run was asked for. B the partition the data produces, with centroids marked; C the "
        "same points coloured by the label assigned by hand. Anywhere the two disagree is a "
        "paradigm the label puts in one literature and the geometry puts in the other. D the "
        "contingency table with chance-corrected agreement. E what each discovered cluster "
        "is, read off the ladders."),
    "figA1_ladders": (
        "Occupancy of every rung of every ladder",
        "Bars are counts at the admissible values; ticks below each axis are individual "
        "paradigms, coloured by cluster. Drawing counts on the lattice rather than a smooth "
        "density avoids inventing mass between rungs, where no design can sit."),
    "figA2_year": (
        "What the field has been able to put on the table over time",
        "Motor task difficulty $x$ (left) and surprise hierarchy $z$ (right) against "
        "publication year, with a rolling mean. The rise in $x$ tracks the move from "
        "stimulation and discrete-trial designs to continuous control; $z$ shows no "
        "comparable trend, which is the asymmetry this review is about."),
}


def write_captions(path, subs):
    out = ["% generated by paradigm_space.py — numbers are filled in from the corpus",
           "% \\graphicspath{{figs/}} and \\usepackage{graphicx} assumed", ""]
    for name, (short, long) in CAPTIONS.items():
        text = long
        for k, v in subs.items():
            text = text.replace("{" + k + "}", str(v))
        out += [r"\begin{figure}[tbp]", r"  \centering",
                rf"  \includegraphics[width=\textwidth]{{{name}.pdf}}",
                rf"  \caption[{short}]{{\textbf{{{short}.}} {text}}}",
                rf"  \label{{fig:{name}}}", r"\end{figure}", ""]
    Path(path).write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Figures and interactive map of the experimental paradigm space.")
    ap.add_argument("workbook", help="literature database .xlsx")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    ap.add_argument("--sheet", default="Articles")
    ap.add_argument("--criteria", default="Axis Fla criteria")
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--sigma-reach", type=float, default=SIGMA_REACH)
    ap.add_argument("--grid", type=int, default=GRID)
    ap.add_argument("--drop-lo", action="store_true",
                    help="rule R8: re-run every coverage claim without inferred scores")
    ap.add_argument("--per-event", action="store_true",
                    help="expand the second and third event blocks into their own points")
    ap.add_argument("-k", "--clusters", type=int, default=2,
                    help="number of clusters for the unsupervised check (0 to skip)")
    ap.add_argument("--cluster-method", choices=["kmeans", "ward"], default="kmeans")
    ap.add_argument("--cluster-axes", default=None,
                    help="comma-separated axes to cluster on, e.g. x,y,z,t")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--no-pdf", action="store_true",
                    help="write report.tex but do not run pdflatex on it")
    ap.add_argument("--cdn", action="store_true",
                    help="load plotly from the CDN instead of inlining it")
    args = ap.parse_args(argv)

    cfg = Config(sigma=args.sigma, sigma_reach=args.sigma_reach, grid=args.grid,
                 drop_lo=args.drop_lo, per_event=args.per_event)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(args.workbook, sheet_name=args.sheet)
    ladders = load_ladders(args.workbook, args.criteria)
    df = load_corpus(args.workbook, cfg, args.sheet)
    if not len(df):
        raise SystemExit("no rows with scorable = yes")
    df.to_csv(out / "paradigm_scores.csv", index=False)

    use_style()
    fig_space(df, ladders, out)
    n_full = fig_projections(df, ladders, out)
    counts = fig_regions(df, ladders, out)
    fig_cube(df, ladders, out)
    disp = fig_task_axes(df, ladders, out)
    gaps, _ = gap_search(df, cfg)
    fig_gaps(df, cfg, ladders, gaps, out)
    thesis = df[df["thesis"]].dropna(subset=cfg.axes)
    point = thesis[cfg.axes].mean().to_numpy() if len(thesis) else centroid("G1", cfg.axes)
    f_thesis, n_eff = fig_accounts(df, point, out)
    corr = fig_audit(raw, df, out)
    clus = None
    if args.clusters and args.clusters >= 2:
        cax = args.cluster_axes.split(",") if args.cluster_axes else cfg.axes
        clus = cluster_corpus(df, cfg, k=args.clusters, method=args.cluster_method,
                              axes=[a.strip() for a in cax], 
                              out = out )
        fig_clusters(clus, ladders, out)
    fig_ladders(df, ladders, out)
    fig_year(df, ladders, out)

    txt = report(df, raw, cfg, gaps, counts, disp, f_thesis, n_eff,
                 corr, out / "scoring_report.txt", clus)
    fragment = write_results(out / "results.md", out / "results.tex", df, raw, cfg, gaps,
                             counts, disp, f_thesis, n_eff, corr, ladders, args.workbook,
                             clus)
    write_report(out / "report.tex", fragment, args.workbook)
    if not args.no_pdf:
        pdf = compile_pdf(out / "report.tex")
        if pdf:
            print(f"{pdf.name} compiled ({pdf.stat().st_size // 1024} kB)")

    if not args.no_html:
        offline = write_html(out / "paradigm_space.html", df, ladders, cfg, gaps, raw,
                             args.workbook, inline=False if args.cdn else None,
                             clus=clus 
                             )#method=args.cluster_method, n_clusters=args.clusters)
        note = "plotly inlined, works offline" if offline else "plotly loaded from the CDN"
        print(f"paradigm_space.html written ({note})")

    print(txt)
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()