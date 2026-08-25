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

# In the call of the function we can also modify other parameters:
#   --cluster-method ward: conducts the clustering with waed method instead of kmeans (default)
#   --thesis include: includes the entries from the thesis 
#       we can also select {auto,holdout,include,drop},
#       default auto. It detects whether the workbook has thesis rows and, 
#       if it does, holds them out of every estimate — density, region counts, 
#       gap search, clustering, diagnostics, account field — while still drawing 
#       them as purple stars. So the same command does the right thing before you 
#       add them and after. 
#       holdout forces it, include treats them as ordinary corpus, drop discards them entirely. 
#   --axis x,y,z,t,s,x1,y1 determines the axes used for the clsutering
#   --feasible applies the three structural constraints, restricting the gap search to
#       the assumed feasible region (see figs_0824). OFF by default: the corpus was
#       tested against those constraints and five published paradigms sit inside the
#       region they call impossible (sperry1950neural, eliades2008neural,
#       itti2009bayesian break y <= x + 1/3; landy2012dynamic, wolpert1995internal
#       break x <= y + 1/2). See fig11_feasible and constraint_audit.csv.
#       --no-feasible still works and is now a no-op restating the default.

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
import math
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

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
# The structural constraints are OFF by default. They were on until the corpus was
# tested against them: five published paradigms sit inside the region they call
# impossible (constraint_audit, fig11_feasible). Set this to True, or pass --feasible,
# if a revised set of constraints survives that test.
USE_FEASIBLE = False

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

# Two display switches, both defaulting off.
#
# Confidence is a property of the coding rather than of the paradigm, and letting it
# drive marker shape made every scatter carry a second variable the reader had to
# decode before seeing the first. It stays where it belongs: in the row weights, in
# the audit figure, and in the report.
#
# Region names and occupancy counts are stated in the funnel, the report and the
# prose, so annotating every panel with them repeats one number in five places and
# crowds the points it is counting.
MARK_CONFIDENCE = False
LABEL_REGIONS = False

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
ACCOUNT_LABEL = {
    "PC": "predictive coding", "AIF": "active inference",
    "OFC": "optimal feedback control", "SDT": "statistical decision theory",
    "DDM": "drift diffusion", "AC": "affordance competition",
    "SAL": "salience detection", "RL": "reinforcement learning",
}
# one colour per account, used wherever the account field is drawn categorically.
# Chosen so the two accounts that own the dense clusters (OFC, SAL) carry the
# cluster colours of figure 1, and the two the thesis argues for (PC, AIF) are
# neighbours in hue.
ACCOUNT_COLOR = {
    "PC": "#3E6BB0", "AIF": "#6A3D9A", "OFC": "#157F7F", "SDT": "#8C6239",
    "DDM": "#B0447A", "AC": "#D2892A", "SAL": "#C1425A", "RL": "#6B8E4E",
}
# the account field is only meaningful where enough rows support it; below this
# effective count the cell is drawn as unsupported rather than as a confident zero
ACCOUNT_MIN_NEFF = 2.0
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


def feasible(u, axes, enabled=None):
    """The structural constraints of appendix A.1, evaluated on a grid.

    These encode combinations of coordinates that no experiment can realise, not
    combinations nobody has tried. A motor command cannot stay open to revision over a
    long window when there is no motor task to revise, so y > x + 1/3 is impossible
    rather than unexplored, and reporting it as a gap would be reporting an artefact of
    the parametrisation. The constraints are a modelling assumption like any other and
    are worth testing: USE_FEASIBLE (--no-feasible) turns them off, and the search then
    runs on the whole cube, which is the honest way to see how much of the answer the
    mask is responsible for.
    """
    u = np.atleast_2d(np.asarray(u, float))
    if not (USE_FEASIBLE if enabled is None else enabled):
        return np.ones(len(u), bool)
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


# ---------------------------------------------------------------------------
# 2b. empty space, measured geometrically
# ---------------------------------------------------------------------------
#
# gap_search above answers "where is the smoothed density low, and is that place a
# short walk from existing work". That is a statement about a kernel estimate, and it
# inherits the kernel's bandwidth: a gap found at sigma = 0.09 is a gap at that scale.
# It is already a joint calculation — the grid is the full 4D product and the kernel
# uses the 4D distance — but nothing in its output tells the reader how large the hole
# is, and a bandwidth-dependent point estimate is awkward to defend as a design
# specification.
#
# The functions below take the other route: no kernel, no accounts, no smoothing. They
# treat the corpus as a finite point set in the feasible part of the unit cube and ask
# a purely geometric question — where are the largest regions containing no published
# paradigm at all, and how large are they. A ball gives the depth of a hole in one
# number; a box gives it as an interval per axis, which is what a design specification
# actually looks like.
#
# The feasibility mask matters more here than anywhere else in the pipeline and is the
# answer to the most common question about these maps. Large parts of the unit cube are
# not under-studied but structurally impossible: a motor command cannot stay open to
# revision over a long window when there is no motor task to revise, so everything with
# y > x + 1/3 is excluded before any search runs. Low motor difficulty with a long motor
# timescale is empty in the corpus, and correctly never reported as a gap.

CONSTRAINTS = [
    ("y <= x + 1/3", lambda P, i: P[:, i["y"]] <= P[:, i["x"]] + 1 / 3 + 1e-9,
     "a command cannot stay open to revision with no motor task"),
    ("x <= y + 1/2", lambda P, i: P[:, i["x"]] <= P[:, i["y"]] + 0.5 + 1e-9,
     "a demanding task cannot be over before it starts"),
    ("not (z < .05 and t > .05)",
     lambda P, i: ~((P[:, i["z"]] < 0.05) & (P[:, i["t"]] > 0.05)),
     "an event with no structure cannot define the task"),
]


def constraint_audit(df, axes):
    """Test each structural constraint against the corpus that is supposed to obey it.

    A constraint that claims impossibility is refutable, and the corpus is the thing
    that refutes it: any published paradigm sitting in the excluded region is a design
    the constraint says cannot exist. Run before trusting the mask, because an
    unfalsified assumption and an untested one look identical in the output.
    """
    sub = df.dropna(subset=axes)
    P = sub[axes].to_numpy(float)
    i = {a: k for k, a in enumerate(axes)}
    rows, offenders = [], {}
    for name, test, why in CONSTRAINTS:
        try:
            ok = test(P, i)
        except KeyError:
            continue
        bad = sub[~ok]
        rows.append(dict(constraint=name, rationale=why, violations=int((~ok).sum()),
                         share=float((~ok).mean())))
        offenders[name] = bad
    return pd.DataFrame(rows), offenders


def _grid_cells(axes, grid, feasible_only=True):
    lin = [np.linspace(0, 1, grid)] * len(axes)
    cells = np.stack(np.meshgrid(*lin, indexing="ij"), -1).reshape(-1, len(axes))
    return cells[feasible(cells, axes)] if feasible_only else cells


def empty_balls(P, axes, k=4, grid=25, refine=3, min_sep=0.20):
    """The largest empty balls in the feasible set: centre, radius, nearest paradigm.

    The centre of the largest empty ball is the point of the design space furthest
    from anything anyone has run — the deepest hole rather than the lowest smoothed
    density — and its radius says how deep in the units of the axes themselves. Found
    on a grid and then refined locally, which is enough at this resolution: the exact
    solution is a vertex of the farthest-point Voronoi diagram, and in four dimensions
    the exact construction buys nothing that a refined grid does not.

    Successive balls are required to be `min_sep` apart so that the list describes
    several holes rather than several points in the same one.
    """
    cells = _grid_cells(axes, grid)
    tree = cKDTree(P)
    d, _ = tree.query(cells, k=1)
    out, chosen = [], []
    step = 1.0 / (grid - 1)
    for _ in range(k):
        order = np.argsort(-d)
        pick = None
        for i in order:
            u = cells[i]
            if all(np.linalg.norm(u - c) >= min_sep for c in chosen):
                pick = i
                break
        if pick is None:
            break
        u = cells[pick].copy()
        r = d[pick]
        # local refinement: walk the centre uphill on the distance-to-nearest field,
        # halving the step, so the radius is not quantised to the grid
        h = step / 2
        for _ in range(refine):
            improved = True
            while improved:
                improved = False
                for j in range(len(axes)):
                    for s in (+1, -1):
                        v = u.copy()
                        v[j] = np.clip(v[j] + s * h, 0, 1)
                        if not feasible(v[None, :], axes)[0]:
                            continue
                        rv = np.sqrt(((v - P) ** 2).sum(1)).min()
                        if rv > r + 1e-12:
                            u, r, improved = v, rv, True
            h /= 2
        j = int(np.argmin(((u - P) ** 2).sum(1)))
        chosen.append(u)
        out.append(dict(radius=float(r), nearest=j,
                        **{a: float(v) for a, v in zip(axes, u)}))
        d = np.minimum(d, np.sqrt(((cells - u) ** 2).sum(1)))   # do not re-find it
    return pd.DataFrame(out)


def _box_feasible(lo, hi, axes, n=3):
    grids = [np.linspace(l, h, n) for l, h in zip(lo, hi)]
    pts = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, len(axes))
    return bool(feasible(pts, axes).all())


def maximal_box(u, P, axes, step=0.02, margin=1e-6):
    """Grow an axis-aligned empty box around u until nothing can grow further.

    Reported alongside the ball because a radius is not a protocol. A box reads
    directly as a specification — this range of motor difficulty, that range of task
    relevance — and its per-axis widths say which coordinate the hole is actually wide
    in, which a single radius cannot. Growth is greedy on volume and stops at the
    first paradigm or the edge of the feasible set, so the result is maximal (no face
    can move) though not necessarily maximum (no larger box exists elsewhere).
    """
    lo, hi = u.copy(), u.copy()

    def empty(lo_, hi_):
        return not np.all((P >= lo_ - margin) & (P <= hi_ + margin), axis=1).any()

    grew = True
    while grew:
        grew = False
        best = None
        for j in range(len(axes)):
            for s in (+1, -1):
                lo2, hi2 = lo.copy(), hi.copy()
                if s > 0:
                    hi2[j] = min(hi2[j] + step, 1.0)
                    if hi2[j] <= hi[j]:
                        continue
                else:
                    lo2[j] = max(lo2[j] - step, 0.0)
                    if lo2[j] >= lo[j]:
                        continue
                if not empty(lo2, hi2) or not _box_feasible(lo2, hi2, axes):
                    continue
                gain = np.prod(np.maximum(hi2 - lo2, 1e-9))
                if best is None or gain > best[0]:
                    best = (gain, lo2, hi2)
        if best is not None:
            _, lo, hi = best
            grew = True
    return lo, hi


def empty_regions(df, cfg, k=4, grid=25, step=0.02, n_null=200, seed=0):
    """Largest empty balls, their maximal boxes, and whether they are larger than chance.

    The null answers the question a reviewer asks of any hole: 104 points scattered
    at random in a four-dimensional feasible set leave holes too. Uniform samples of
    the same size on the same feasible set are drawn, and the radius of the largest
    empty ball is recorded for each, giving the distribution the observed radius is
    read against.
    """
    axes = cfg.axes
    P, w, sub = matrix(df, axes)
    balls = empty_balls(P, axes, k=k, grid=grid)
    rows = []
    for _, b in balls.iterrows():
        u = np.array([b[a] for a in axes])
        lo, hi = maximal_box(u, P, axes, step=step)
        r = dict(radius=b["radius"],
                 **{a: float(v) for a, v in zip(axes, u)},
                 volume=float(np.prod(hi - lo)),
                 nearest=str(sub.iloc[int(b["nearest"])]["citekey"]
                             or sub.iloc[int(b["nearest"])]["paradigm_id"]),
                 **{f"{a}_lo": float(l) for a, l in zip(axes, lo)},
                 **{f"{a}_hi": float(h) for a, h in zip(axes, hi)})
        r["in_region"] = "+".join(
            n for n, spec in REGIONS.items()
            if all((u[axes.index(kk)] >= v) if op == ">=" else (u[axes.index(kk)] <= v)
                   for kk, op, v in spec["constraints"])) or "—"
        rows.append(r)
    tab = pd.DataFrame(rows)

    rng = np.random.default_rng(seed)
    cells = _grid_cells(axes, grid)
    null = []
    obs0 = float(balls["radius"].max()) if len(balls) else float("nan")
    if n_null <= 0:                       # comparison run: the table is all that is wanted
        return dict(table=tab, null=np.array([]), radius=obs0, p=float("nan"),
                    null_mean=float("nan"),
                    feasible_fraction=float(feasible(
                        _grid_cells(axes, grid, feasible_only=False), axes).mean()))
    for _ in range(n_null):
        idx = rng.choice(len(cells), len(P), replace=True)
        Q = cells[idx] + rng.uniform(-0.5, 0.5, (len(P), len(axes))) / (grid - 1)
        Q = np.clip(Q, 0, 1)
        dn, _ = cKDTree(Q).query(cells, k=1)
        null.append(float(dn.max()))
    null = np.array(null)
    obs = float(tab["radius"].max()) if len(tab) else float("nan")
    return dict(table=tab, null=null, radius=obs,
                p=float((null >= obs).sum() + 1) / (n_null + 1),
                null_mean=float(null.mean()),
                feasible_fraction=float(feasible(
                    _grid_cells(axes, grid, feasible_only=False), axes).mean()))


# ---------------------------------------------------------------------------
# 2c. empty space, method 3: joint low-density regions
# ---------------------------------------------------------------------------
#
# Methods 1 and 2 both return points, or a box grown around a point. Neither describes
# the *shape* of an empty region, and neither uses the smoothed density jointly: method
# 1 thresholds it cell by cell and then picks two extreme cells, method 2 ignores it
# entirely and works from nearest-neighbour distance.
#
# This method takes the joint density as the object of interest. It evaluates rho(u)
# over the full four-dimensional grid, keeps the cells below a quantile of it, and
# groups those cells into connected components. A component is a low-density *region*
# with a volume, a shape and a boundary, rather than a point; the largest box that
# fits inside it is then reported so the region can still be read as a specification.
#
# The difference from method 2 is not joint versus sequential — method 2 is already
# joint, its emptiness test being a conjunction over all four coordinates at once — but
# smoothed versus exact, and region versus point. Method 2 asks how far a design can be
# from any published paradigm. Method 3 asks how large a contiguous stretch of the space
# the corpus leaves thinly covered. A hole can be deep and narrow (method 2 finds it,
# method 3 may not resolve it) or shallow and vast (the reverse).

def _neighbours(idx, shape):
    for d in range(len(shape)):
        for step in (-1, 1):
            j = list(idx)
            j[d] += step
            if 0 <= j[d] < shape[d]:
                yield tuple(j)


def _components(mask):
    """Connected components of a boolean grid, face-connectivity in n dimensions."""
    lab = np.zeros(mask.shape, int)
    cur = 0
    for start in zip(*np.nonzero(mask)):
        if lab[start]:
            continue
        cur += 1
        stack = [start]
        lab[start] = cur
        while stack:
            u = stack.pop()
            for v in _neighbours(u, mask.shape):
                if mask[v] and not lab[v]:
                    lab[v] = cur
                    stack.append(v)
    return lab, cur


def _inscribed_box(mask, seed):
    """The largest axis-aligned box inside `mask` containing `seed`, grown greedily."""
    lo = list(seed)
    hi = list(seed)
    grew = True
    while grew:
        grew = False
        best = None
        for d in range(mask.ndim):
            for step in (-1, 1):
                lo2, hi2 = lo.copy(), hi.copy()
                if step > 0:
                    if hi2[d] + 1 >= mask.shape[d]:
                        continue
                    hi2[d] += 1
                else:
                    if lo2[d] - 1 < 0:
                        continue
                    lo2[d] -= 1
                sl = tuple(slice(a, b + 1) for a, b in zip(lo2, hi2))
                if not mask[sl].all():
                    continue
                gain = np.prod([b - a + 1 for a, b in zip(lo2, hi2)])
                if best is None or gain > best[0]:
                    best = (gain, lo2, hi2)
        if best is not None:
            _, lo, hi = best
            grew = True
    return lo, hi


def low_density_regions(df, cfg, grid=17, quantile=0.02, k=3, min_cells=3):
    """Connected components of the joint low-density set, with an inscribed box each.

    The threshold is a quantile of the density over the searchable set rather than an
    absolute value, so the method reports the emptiest quarter of the space whatever
    the corpus size, and the reader can move it. Components smaller than `min_cells`
    are dropped as grid noise.
    """
    axes = cfg.axes
    P, w, sub = matrix(df, axes)
    lin = np.linspace(0, 1, grid)
    dens = density(P, w, tuple(range(len(axes))), (lin,) * len(axes), cfg.sigma)
    dens = dens / dens.max()
    cells = np.stack(np.meshgrid(*([lin] * len(axes)), indexing="ij"), -1)
    ok = feasible(cells.reshape(-1, len(axes)), axes).reshape(dens.shape)
    thr = float(np.quantile(dens[ok], quantile))
    low = (dens <= thr) & ok
    lab, n = _components(low)
    cell_vol = (1.0 / (grid - 1)) ** len(axes)
    rows = []
    for c in range(1, n + 1):
        m = lab == c
        size = int(m.sum())
        if size < min_cells:
            continue
        pts = np.stack(np.nonzero(m), -1)
        centre = cells[m].mean(0)
        deep = tuple(pts[np.argmin(dens[m])])
        lo_i, hi_i = _inscribed_box(m, deep)
        lo = np.array([lin[i] for i in lo_i])
        hi = np.array([lin[i] for i in hi_i])
        inside = int(satisfies(sub, [(a, ">=", float(l)) for a, l in zip(axes, lo)]
                               + [(a, "<=", float(h)) for a, h in zip(axes, hi)]).sum())
        rows.append(dict(
            component=c, cells=size, volume=size * cell_vol,
            min_density=float(dens[m].min()), mean_density=float(dens[m].mean()),
            occupancy=inside,
            **{a: float(v) for a, v in zip(axes, centre)},
            **{f"{a}_lo": float(l) for a, l in zip(axes, lo)},
            **{f"{a}_hi": float(h) for a, h in zip(axes, hi)},
            box_volume=float(np.prod(hi - lo))))
    tab = pd.DataFrame(rows).sort_values("volume", ascending=False).head(k)

    # how many distinct low-density regions there are is a function of the level, so
    # the level is not left as an unexamined choice: the sweep below is reported with
    # the result and drawn in the figure. Too high a quantile and the whole empty part
    # of the cube is one component; too low and it fragments into grid noise.
    sweep = []
    for q in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20):
        t2 = float(np.quantile(dens[ok], q))
        l2, n2 = _components((dens <= t2) & ok)
        big = [int((l2 == c).sum()) for c in range(1, n2 + 1)]
        sweep.append(dict(quantile=q, components=int(sum(b >= min_cells for b in big)),
                          largest=max(big) if big else 0,
                          cells=int(((dens <= t2) & ok).sum())))
    return dict(table=tab.reset_index(drop=True), threshold=thr, grid=grid,
                quantile=quantile, labels=lab, density=dens, lin=lin,
                n_components=n, sweep=pd.DataFrame(sweep),
                low_fraction=float(low.sum() / max(int(ok.sum()), 1)))


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


def pushforward(df, mask=None):
    """Which accounts the corpus actually spends its mass on."""
    sub = df if mask is None else df[mask]
    Y, keep = account_matrix(sub)
    sub, Y = sub[keep], Y[keep]
    if not len(sub):
        return {a: 0.0 for a in ACCOUNTS}
    w = sub["w"].to_numpy(float)
    m = (w[:, None] * Y).sum(0)
    m = m / m.sum() if m.sum() else m
    return {a: float(v) for a, v in zip(ACCOUNTS, m)}


# --- the account field over the design space ------------------------------
#
# account_field above answers "what would the field call this one design". The
# functions below answer the prior question the review actually needs: how the
# accounts are laid out over the space as a whole, so that an empty region can be
# checked for emptiness in C as well as in H. The estimator is the same
# Nadaraya-Watson regression, evaluated on a grid rather than at a point, and it is
# reported with its effective support: a cell whose estimate rests on one row is a
# statement about that row, not about the field.

def account_rows(df, exclude_thesis=True):
    """Rows that carry an account, with their normalised distributions."""
    sub = df if not exclude_thesis else df[~df["thesis"]]
    Y, keep = account_matrix(sub)
    return sub[keep], Y[keep]


def account_plane(df, a, b, n=61, sigma=SIGMA, exclude_thesis=True):
    """f(u) on the (a, b) plane, marginalising the remaining axes over the corpus.

    Conditioning the kernel on the two plotted coordinates only is the account-space
    analogue of the exact density marginal used for the coverage deficit: the value
    at a cell is the corpus-weighted expectation of the account distribution given
    those two coordinates, not a slice through a fixed value of the others.

    Returns the grid, F of shape (n, n, |ACCOUNTS|), and the effective number of
    rows behind every cell.
    """
    lin = np.linspace(0, 1, n)
    sub, Y = account_rows(df.dropna(subset=[a, b]), exclude_thesis)
    if not len(sub):
        return lin, np.zeros((n, n, len(ACCOUNTS))), np.zeros((n, n))
    P = sub[[a, b]].to_numpy(float)
    w = sub["w"].to_numpy(float)
    GA, GB = np.meshgrid(lin, lin, indexing="ij")
    U = np.stack([GA.ravel(), GB.ravel()], axis=1)
    d2 = ((U[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    K = w * np.exp(-d2 / (2 * sigma ** 2))
    s = K.sum(1)
    ok = s > 1e-12
    F = np.zeros((len(U), len(ACCOUNTS)))
    F[ok] = (K[ok] @ Y) / s[ok, None]
    n_eff = np.zeros(len(U))
    n_eff[ok] = s[ok] ** 2 / (K[ok] ** 2).sum(1)
    return lin, F.reshape(n, n, len(ACCOUNTS)), n_eff.reshape(n, n)


def account_entropy(F):
    """Normalised entropy of the account distribution, 0 owned to 1 contested."""
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(F > 0, F * np.log(F), 0.0), axis=-1)
    live = max(int((F.reshape(-1, F.shape[-1]).sum(0) > 0).sum()), 2)
    return H / np.log(live)


def account_dominant(F, floor=0.0):
    """Index of the leading account and its margin over the runner-up."""
    order = np.argsort(-F, axis=-1)
    top = np.take_along_axis(F, order[..., :1], -1)[..., 0]
    second = np.take_along_axis(F, order[..., 1:2], -1)[..., 0]
    idx = order[..., 0].astype(float)
    idx[top <= floor] = np.nan
    return idx, top - second


def account_probes(df, probes, axes=None, sigma=SIGMA):
    """f and its effective support at a handful of named points of the space.

    This is the table the gap argument turns on: a region that is empty of papers
    but sits in a well-owned part of account space is a region where filling the
    gap only re-tests a settled question.
    """
    axes = axes or PRINCIPAL
    out = []
    for name, u in probes.items():
        f, n_eff = account_field(df, np.asarray(u, float), axes=axes, sigma=sigma)
        row = dict(probe=name, n_eff=n_eff,
                   entropy=float(account_entropy(np.array([[list(f.values())]]))[0, 0]),
                   **{a: float(v) for a, v in zip(axes, u)}, **f)
        out.append(row)
    return pd.DataFrame(out)


def account_composition(df, by, axes=None):
    """Account mass per group: the h layer restricted to a partition of the corpus."""
    axes = axes or PRINCIPAL
    sub = df.dropna(subset=axes)
    groups = pd.Series(by).reindex(sub.index) if not callable(by) else sub.apply(by, axis=1)
    out = {}
    for g in pd.unique(groups.dropna()):
        m = (groups == g).to_numpy()
        push = pushforward(sub[m])
        out[g] = dict(n=int(m.sum()), **push)
    return pd.DataFrame(out).T


def account_predictivity(df, axes=None, sigma=SIGMA, n_perm=500, seed=0):
    """Does the design predict the account, or is f flat?

    Leave-one-out Nadaraya-Watson over the corpus, scored by log loss and by
    top-1 accuracy against the row's own dominant account, with a null obtained by
    permuting the account labels across designs. A flat field is the honest
    alternative hypothesis for every claim made from the account map, so it is
    tested rather than assumed.
    """
    axes = axes or PRINCIPAL
    sub, Y = account_rows(df.dropna(subset=axes))
    n = len(sub)
    if n < 5:
        return dict(n=n, logloss=float("nan"), accuracy=float("nan"),
                    logloss_null=float("nan"), accuracy_null=float("nan"),
                    p=float("nan"))
    P = sub[axes].to_numpy(float)
    w = sub["w"].to_numpy(float)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    K = w * np.exp(-d2 / (2 * sigma ** 2))
    np.fill_diagonal(K, 0.0)                     # leave one out
    truth = Y.argmax(1)

    def score(Yv):
        s = K.sum(1)
        ok = s > 1e-12
        F = np.zeros_like(Yv)
        F[ok] = (K[ok] @ Yv) / s[ok, None]
        p = np.clip(F[np.arange(n), truth], 1e-6, 1.0)
        return float(-(w * np.log(p)).sum() / w.sum()), \
            float((w * (F.argmax(1) == truth)).sum() / w.sum())

    ll, acc = score(Y)
    rng = np.random.default_rng(seed)
    null_ll, null_acc = [], []
    for _ in range(n_perm):
        l, a = score(Y[rng.permutation(n)])
        null_ll.append(l)
        null_acc.append(a)
    null_ll, null_acc = np.array(null_ll), np.array(null_acc)
    return dict(n=n, logloss=ll, accuracy=acc,
                logloss_null=float(null_ll.mean()),
                accuracy_null=float(null_acc.mean()),
                p=float((null_ll <= ll).sum() + 1) / (n_perm + 1))


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

    # The clustering runs on z-scores so that an axis with a narrow range does not
    # count for less than one with a wide range. Everything a reader sees, however,
    # is a rung: a centroid quoted as -0.81 is uninterpretable against a ladder that
    # runs 0 to 1, and plotting it on a rung axis puts it outside the panel. So the
    # centroids are carried in both units from here on, standardised for the metrics
    # and inverse-transformed for every figure, table and sentence.
    mu = np.mean(X_raw, axis=0)
    C_raw = C * std_devs + mu

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
                          
    return dict(sub=sub, X=X, X_raw=X_raw, mu=mu, sd=std_devs, w=w, axes=axes,
                k=k, method=method, labels=lab,
                centers=C_raw, centers_z=C,
                given=given, contingency=M, found_names=ua, given_names=ub,
                agreement=agree, mapping=mapping, ari=ari, p_ari=p_ari,
                ami=adjusted_mutual_info(lab, given),
                silhouette=silhouette(X, lab),
                stability=bootstrap_stability(X, w, k, n_boot=60, seed=2),
                curve=pd.DataFrame(curve))


# ---------------------------------------------------------------------------
# 2c. cluster diagnostics
# ---------------------------------------------------------------------------
#
# The claim the chapter rests on is not "a clustering algorithm returns three
# groups" — it will return three groups from noise — but "the corpus is dense in
# two places and sparse between them". These functions test that claim and the
# ones that sit under it, each against an explicit null:
#
#   permanova         are the groups separated at all, without assuming normality
#   permdisp          or do they merely differ in spread, which permanova alone
#                     cannot distinguish from a difference in location
#   axis_effects      which axes carry the separation, with an effect size
#   jaccard_stability which individual clusters survive resampling, not the mean
#   prediction_strength  how many clusters actually replicate out of sample
#   separation_profile   is the corpus bimodal along the axis joining two clusters
#
# Everything is permutation- or bootstrap-based: with 112 rows on a lattice, the
# distributional assumptions behind the parametric versions are not met.

def _sq_dists(X):
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, 0.0)
    return d2


def permanova(X, lab, n_perm=999, seed=0):
    """Anderson's pseudo-F on the Euclidean distance matrix, with a permutation null.

    Preferred to MANOVA here: the coordinates are bounded, discrete and far from
    multivariate normal, and Wilks' lambda has no defence in that setting. The
    statistic is the same variance ratio, computed from distances so that no
    distributional assumption is needed.
    """
    d2 = _sq_dists(X)
    n = len(X)
    groups = np.unique(lab)
    a = len(groups)
    if a < 2 or n <= a:
        return dict(F=float("nan"), p=float("nan"), R2=float("nan"), n=n, a=a)

    def stat(l):
        sst = d2.sum() / (2 * n)
        ssw = 0.0
        for g in groups:
            m = l == g
            ng = int(m.sum())
            if ng > 1:
                ssw += d2[np.ix_(m, m)].sum() / (2 * ng)
        ssa = sst - ssw
        return (ssa / (a - 1)) / (ssw / (n - a)) if ssw > 0 else np.inf, ssa / sst

    F, R2 = stat(lab)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(lab))[0] for _ in range(n_perm)])
    return dict(F=float(F), p=float((null >= F).sum() + 1) / (n_perm + 1),
                R2=float(R2), n=n, a=a)


def permdisp(X, lab, n_perm=999, seed=0):
    """Homogeneity of within-group dispersion.

    A significant permanova with a significant permdisp is ambiguous: groups that
    differ only in how tightly they are packed will produce both. Reported so that
    the location claim is not read off a spread difference.
    """
    groups = np.unique(lab)
    d = np.zeros(len(X))
    for g in groups:
        m = lab == g
        d[m] = np.linalg.norm(X[m] - X[m].mean(0), axis=1)

    def stat(l):
        gm = np.array([d[l == g].mean() for g in groups])
        ns = np.array([(l == g).sum() for g in groups])
        between = (ns * (gm - d.mean()) ** 2).sum() / max(len(groups) - 1, 1)
        within = sum(((d[l == g] - d[l == g].mean()) ** 2).sum() for g in groups)
        within /= max(len(X) - len(groups), 1)
        return between / within if within > 0 else np.inf

    F = stat(lab)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(lab)) for _ in range(n_perm)])
    return dict(F=float(F), p=float((null >= F).sum() + 1) / (n_perm + 1),
                spread={str(g): float(d[lab == g].mean()) for g in groups})


def axis_effects(X, lab, axes, n_perm=999, seed=0):
    """Which axes carry the separation, as eta squared with a permutation p.

    A partition that differs on task difficulty but not on task relevance is telling
    you the corpus is organised by engagement rather than by what the event means —
    which is a different claim from "the clusters are real", and needs its own test.
    """
    rng = np.random.default_rng(seed)
    out = []
    groups = np.unique(lab)
    for j, a in enumerate(axes):
        v = X[:, j]
        sst = ((v - v.mean()) ** 2).sum()

        def eta(l):
            if sst <= 0:
                return 0.0
            ssb = sum((l == g).sum() * (v[l == g].mean() - v.mean()) ** 2
                      for g in groups)
            return ssb / sst

        e = eta(lab)
        null = np.array([eta(rng.permutation(lab)) for _ in range(n_perm)])
        out.append(dict(axis=a, eta2=float(e),
                        p=float((null >= e).sum() + 1) / (n_perm + 1),
                        eta2_null=float(null.mean())))
    return pd.DataFrame(out)


def hedges_g(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0
    d = (a.mean() - b.mean()) / sp
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))       # small-sample correction


def pairwise_axis_table(X, lab, axes):
    """Hedges' g on every axis for every pair of clusters."""
    rows = []
    groups = np.unique(lab)
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            r = dict(pair=f"c{g1} vs c{g2}")
            for j, a in enumerate(axes):
                r[a] = hedges_g(X[lab == g1, j], X[lab == g2, j])
            r["centroid_distance"] = float(
                np.linalg.norm(X[lab == g1].mean(0) - X[lab == g2].mean(0)))
            rows.append(r)
    return pd.DataFrame(rows)


def jaccard_stability(X, w, lab, k, n_boot=100, seed=0):
    """Hennig's cluster-wise bootstrap: how often each cluster survives resampling.

    The single mean co-assignment number already reported is dominated by whichever
    cluster is largest, so a small cluster can dissolve completely without moving it.
    Reported per cluster, on the usual reading: below 0.6 dissolved, above 0.75 stable.
    """
    rng = np.random.default_rng(seed)
    groups = np.unique(lab)
    best = {int(g): [] for g in groups}
    for b in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        uniq = np.unique(idx)
        if len(uniq) <= k:
            continue
        lb, _, _ = weighted_kmeans(X[uniq], w[uniq], k, n_init=6, seed=int(seed + b))
        member = {int(g): set(uniq[lb == g]) for g in np.unique(lb)}
        for g in groups:
            orig = set(np.flatnonzero(lab == g)) & set(uniq)
            if not orig:
                continue
            best[int(g)].append(max(
                (len(orig & m) / len(orig | m) for m in member.values() if m),
                default=0.0))
    return {g: float(np.mean(v)) if v else float("nan") for g, v in best.items()}


def prediction_strength(X, w, ks, n_split=20, seed=0):
    """Tibshirani and Walther: how many clusters replicate on held-out data.

    The silhouette rises with k on a lattice, so it cannot choose k here; prediction
    strength can, and comes with a conventional cutoff at 0.8. For each split, the
    training centroids are used to predict the test partition, and the score is the
    worst cluster's rate of preserved co-membership.
    """
    rng = np.random.default_rng(seed)
    out = []
    for k in ks:
        scores = []
        for _ in range(n_split):
            perm = rng.permutation(len(X))
            a, b = perm[:len(X) // 2], perm[len(X) // 2:]
            if min(len(a), len(b)) <= k:
                continue
            la, Ca, _ = weighted_kmeans(X[a], w[a], k, n_init=6, seed=int(rng.integers(1e6)))
            lb, _, _ = weighted_kmeans(X[b], w[b], k, n_init=6, seed=int(rng.integers(1e6)))
            pred = ((X[b][:, None, :] - Ca[None, :, :]) ** 2).sum(-1).argmin(1)
            worst = 1.0
            for g in np.unique(lb):
                m = np.flatnonzero(lb == g)
                if len(m) < 2:
                    continue
                same = pred[m][:, None] == pred[m][None, :]
                np.fill_diagonal(same, False)
                worst = min(worst, same.sum() / (len(m) * (len(m) - 1)))
            scores.append(worst)
        out.append(dict(k=k, prediction_strength=float(np.mean(scores)) if scores else np.nan))
    return pd.DataFrame(out)


def _gaussian_mixture_1d(v, k, iters=300, seed=0):
    """Tiny EM, enough for the one- versus two-component comparison below."""
    rng = np.random.default_rng(seed)
    if k == 1:
        mu, sd = np.array([v.mean()]), np.array([max(v.std(), 1e-6)])
        pi = np.array([1.0])
    else:
        q = np.quantile(v, [0.25, 0.75])
        mu = q + rng.normal(0, 1e-3, k)
        sd = np.full(k, max(v.std(), 1e-6))
        pi = np.full(k, 1.0 / k)
    for _ in range(iters):
        p = pi * np.exp(-0.5 * ((v[:, None] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        tot = p.sum(1, keepdims=True)
        r = p / np.maximum(tot, 1e-300)
        nk = r.sum(0)
        if (nk < 1e-6).any():
            break
        pi = nk / len(v)
        mu = (r * v[:, None]).sum(0) / nk
        sd = np.sqrt(np.maximum((r * (v[:, None] - mu) ** 2).sum(0) / nk, 1e-8))
    ll = float(np.log(np.maximum(
        (pi * np.exp(-0.5 * ((v[:, None] - mu) / sd) ** 2)
         / (sd * np.sqrt(2 * np.pi))).sum(1), 1e-300)).sum())
    npar = 3 * len(mu) - 1
    return dict(mu=mu, sd=sd, pi=pi, loglik=ll,
                bic=float(npar * np.log(len(v)) - 2 * ll))


def _valley(X, w, seed=0):
    """Split into two, project onto the line joining the halves, measure the valley.

    Factored out because the null below has to be put through exactly this, and not
    through some simpler version of it.
    """
    lab, C, _ = weighted_kmeans(X, w, 2, n_init=12, seed=seed)
    d = C[0] - C[1]
    nrm = np.linalg.norm(d)
    if nrm == 0:
        return dict(gap=0.0, delta_bic=0.0, separation=0.0, proj=np.zeros(len(X)),
                    labels=lab, direction=d)
    u = d / nrm
    v = X @ u
    order = np.sort(v)
    lo, hi = np.quantile(order, [0.10, 0.90])
    inner = order[(order >= lo) & (order <= hi)]
    gap = float(np.diff(inner).max() / max(order.std(), 1e-9)) if len(inner) > 2 else 0.0
    one = _gaussian_mixture_1d(order, 1)
    two = _gaussian_mixture_1d(order, 2, seed=seed)
    sep = (abs(two["mu"][0] - two["mu"][1]) / np.sqrt((two["sd"] ** 2).mean())
           if len(two["mu"]) == 2 else 0.0)
    return dict(gap=gap, delta_bic=float(one["bic"] - two["bic"]),
                separation=float(sep), proj=v, labels=lab, direction=u,
                mixture=two)


def separation_profile(X, w, n_null=199, seed=0):
    """Is the corpus bimodal, tested against a null that gets the same treatment?

    The obvious version of this test is circular and worth spelling out, because it
    looks convincing: choose the direction that best separates two clusters, project
    onto it, then ask whether the projection is bimodal. It always is — the direction
    was selected to make it so, and a single Gaussian cloud put through the same steps
    produces a clean-looking valley and a decisive BIC. Testing the projection against
    a Gaussian fitted to that same projection does not repair this, since the
    selection happened before the fit.

    So the null here is not a distribution over projections but over corpora: draw a
    unimodal reference with the covariance of the real corpus, split it in two, pick
    its own best direction, and measure its own valley. The p-values ask whether the
    real corpus has a deeper valley than a shapeless cloud of the same shape and size
    does once both have been through the identical procedure. This is the gap
    statistic's logic applied to bimodality rather than to inertia.
    """
    obs = _valley(X, w, seed=seed)
    rng = np.random.default_rng(seed)
    mu, cov = X.mean(0), np.cov(X, rowvar=False)
    try:
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(X.shape[1]))
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-9)))
    ng, nb = [], []
    for b in range(n_null):
        Z = mu + rng.standard_normal(X.shape) @ L.T
        r = _valley(Z, w, seed=int(seed + b + 1))
        ng.append(r["gap"])
        nb.append(r["delta_bic"])
    ng, nb = np.array(ng), np.array(nb)
    v = np.sort(obs["proj"])
    ties = 1.0 - len(np.unique(np.round(v, 6))) / len(v)
    return dict(
        gap=obs["gap"], delta_bic=obs["delta_bic"], separation=obs["separation"],
        proj=obs["proj"], labels=obs["labels"], direction=obs["direction"],
        p_gap=float((ng >= obs["gap"]).sum() + 1) / (n_null + 1),
        p_bic=float((nb >= obs["delta_bic"]).sum() + 1) / (n_null + 1),
        null_gap=ng, null_bic=nb,
        gap_null_mean=float(ng.mean()), bic_null_mean=float(nb.mean()),
        tied_fraction=float(ties), n_null=n_null)


def cluster_diagnostics(clus, n_perm=999, seed=0, boot=100):
    """Every test above, run on the partition the corpus produced."""
    X, lab, w, axes = clus["X"], clus["labels"], clus["w"], clus["axes"]
    k = clus["k"]
    out = dict(
        permanova=permanova(X, lab, n_perm=n_perm, seed=seed),
        permdisp=permdisp(X, lab, n_perm=n_perm, seed=seed),
        axis=axis_effects(X, lab, axes, n_perm=n_perm, seed=seed),
        pairwise=pairwise_axis_table(X, lab, axes),
        jaccard=jaccard_stability(X, w, lab, k, n_boot=boot, seed=seed),
        strength=prediction_strength(X, w, list(clus["curve"]["k"]), seed=seed),
    )
    # The valley test does not take the k of this run, nor its labels: it splits the
    # corpus in two itself, because "two literatures with a sparse region between
    # them" is a claim about a binary split of the whole corpus, and because a null
    # can only be put through a procedure that does not consult the answer.
    sep = separation_profile(X, w, n_null=max(n_perm // 5, 99), seed=seed)
    given = clus.get("given")
    if given is not None:
        sep["composition"] = {
            f"h{j}": {str(g): int(((sep["labels"] == j) & (given == g)).sum())
                      for g in np.unique(given)}
            for j in (0, 1)}
    out["separation"] = sep
    out["pair"] = None
    return out


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
from matplotlib.patches import Circle, Rectangle, Patch          # noqa: E402
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
        # the subtitle carries the statistics, which are longer than the title; left
        # unwrapped it runs across the panel next door exactly as an unwrapped title
        # would, so it wraps at the same width and the pad grows by its line count
        slines = textwrap.wrap(sub, width=int(width * 1.25)) or [sub]
        ax.text(0, 1.008, "\n".join(slines), transform=ax.transAxes,
                fontsize=fontsize - 1.1, color="#6C7C85", va="bottom", ha="left",
                linespacing=1.25)
        pad = pad + len(slines) * (fontsize + 1.5)
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
    """Cluster colour; rows failing `live` are drawn as context.

    Marker shape follows confidence only when MARK_CONFIDENCE is on. With it off
    every row is a filled circle and the panel carries one variable, not two.
    """
    if jitter:
        xa, xb = spread(df[a].to_numpy(float), df[b].to_numpy(float))
    else:
        xa, xb = df[a].to_numpy(float), df[b].to_numpy(float)
    live = np.ones(len(df), bool) if live is None else np.asarray(live, bool)
    groups = (CONF_MARKER.items() if MARK_CONFIDENCE else [(None, "o")])
    for conf, marker in groups:
        for cl, style in CLUSTERS.items():
            m = (df["cluster"].to_numpy() == cl)
            if conf is not None:
                m = m & (df["confidence"].to_numpy() == conf)
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


def scatter_overlay(ax, over, a, b, size=34, zorder=6, jitter=True):
    """Draw the held-out thesis paradigms on top of an estimate they took no part in.

    Deliberately a different marker rather than a different colour: these rows were
    excluded from the density, the regions and the clustering, so they are evidence
    about the map rather than part of it, and the panel should not let them be
    mistaken for corpus.
    """
    if over is None or not len(over):
        return
    sub = over.dropna(subset=[a, b])
    if not len(sub):
        return
    if jitter:
        xa, xb = spread(sub[a].to_numpy(float), sub[b].to_numpy(float))
    else:
        xa, xb = sub[a].to_numpy(float), sub[b].to_numpy(float)
    ax.scatter(xa, xb, s=size, marker="*", color=CLUSTERS["thesis"]["color"],
               edgecolors="white", linewidths=0.5, alpha=0.95, zorder=zorder)


def corpus_legend(fig, ncol=4, y=-0.02, context=False, confidence=None,
                  overlay=False):
    handles = [Line2D([], [], marker="o", ls="", color=s["color"],
                      markeredgecolor="white", markersize=5, label=s["label"])
               for k, s in CLUSTERS.items() if not (overlay and k == "thesis")]
    if confidence is None:
        confidence = MARK_CONFIDENCE
    if confidence:   # 3D scatter draws one marker per call, so shape carries nothing there
        handles += [Line2D([], [], marker=m, ls="", color=INK, markersize=4.4,
                           label=f"confidence {c}") for c, m in CONF_MARKER.items()]
    if overlay:
        handles.append(Line2D([], [], marker="*", ls="",
                              color=CLUSTERS["thesis"]["color"],
                              markeredgecolor="white", markersize=8,
                              label="thesis paradigms (held out of every estimate)"))
    if context:
        handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="none",
                              markeredgecolor=GREY, markersize=4.4,
                              label="excluded by an off-plane constraint"))
    fig.legend(handles=handles, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, y), handletextpad=0.3, columnspacing=1.2)


# --- figure 1: the corpus in projection -----------------------------------

def fig_projections(df, ladders, out, overlay=None, cfg=None):
    """Six planes, six identical square panels, no region boxes.

    The old version of this figure drew the candidate regions here, which is
    what made it read as emptier than the data: a box in the (x, y) plane is a
    slab in the space, and points that a third constraint excludes still fall
    inside its shadow. Region membership now has a figure of its own.

    Each panel carries the exact marginal density beneath its scatter, on one shared
    scale, so that the eye reads where the corpus concentrates rather than trying to
    judge it from overplotted markers on a lattice.
    """
    planes = [("x", "y"), ("x", "z"), ("x", "t"),
              ("y", "z"), ("y", "t"), ("z", "t")]
    sub = df.dropna(subset=PRINCIPAL)
    sigma = getattr(cfg, "sigma", SIGMA) if cfg is not None else SIGMA
    P, w = sub[PRINCIPAL].to_numpy(float), sub["w"].to_numpy(float)
    lin = np.linspace(0, 1, 70)
    shade = matplotlib.colors.LinearSegmentedColormap.from_list(
        "wall", ["#FFFFFF", "#DCE4E9", "#8FA5B1"])
    fig, axes = plt.subplots(2, 3, figsize=(3 * PANEL_W + 1.1, 2 * PANEL_W + 1.0))
    for ax, (a, b) in zip(axes.ravel(), planes):
        rungs = ([v for v, *_ in ladders.get(a, {}).get("rungs", [])],
                 [v for v, *_ in ladders.get(b, {}).get("rungs", [])])
        d = density(P, w, (PRINCIPAL.index(a), PRINCIPAL.index(b)), (lin, lin), sigma)
        ax.contourf(lin, lin, (d / d.max()).T, levels=np.linspace(0.05, 1, 10),
                    cmap=shade, alpha=0.75, zorder=0)
        square(ax, label_of(a, ladders), label_of(b, ladders), rungs)
        scatter_corpus(ax, sub, a, b)
        scatter_overlay(ax, overlay, a, b)
        r = np.corrcoef(sub[a], sub[b])[0, 1]
        ax.text(0.035, 0.965, f"$r$ = {r:+.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.4, color="#54646D",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
    fig.suptitle(f"the corpus in the six principal planes  ·  n = {len(sub)} paradigms"
                 "  ·  shading is the marginal density",
                 fontsize=8, y=0.995, color="#54646D")
    corpus_legend(fig, ncol=4, y=-0.035, overlay=overlay is not None and len(overlay) > 0)
    save(fig, out, "fig1_projections")
    return len(sub)


# --- figure 2: the candidate regions, tested ------------------------------

def holes_as_regions(table, axes, k=2, floor=1e-6):
    """Turn the discovered empty boxes into region specifications.

    The two candidate regions are hypotheses written by hand from the mechanistic
    question; these are their unsupervised counterparts, read off the geometry of the
    corpus alone. Expressing them in the same form lets both go through the same
    figure, the same funnel and the same occupancy count, so the comparison between a
    region someone proposed and a region the data volunteered is like for like.

    Only a bound sitting exactly on the edge of the cube is dropped, since that is the
    one case where it constrains nothing. Dropping a bound merely because it is close
    to the edge widens the region past where the box actually stops, and a box
    advertised as empty then shows occupants in its own funnel.
    """
    out = {}
    colours = ["#3B2E80", "#157F7F", "#8C6239", "#B0447A"]
    for i, (_, row) in enumerate(table.head(k).iterrows()):
        cons = []
        for a in axes:
            lo, hi = float(row[f"{a}_lo"]), float(row[f"{a}_hi"])
            # round inward, never outward. Rounding a lower bound down or an upper
            # bound up widens the box past where the growth stopped, which is exactly
            # where the nearest paradigm sits: a box advertised as empty would then
            # show occupants in its own funnel.
            lo_r = math.ceil(lo * 100) / 100
            hi_r = math.floor(hi * 100) / 100
            if lo_r > floor:
                cons.append((a, ">=", lo_r))
            if hi_r < 1 - floor:
                cons.append((a, "<=", hi_r))
        if not cons:
            continue
        name = f"E{i + 1}"
        out[name] = dict(
            title=("largest empty box: "
                   + ", ".join(f"{a} {'≥' if op == '>=' else '≤'} {v:g}"
                               for a, op, v in cons)
                   if i == 0 else
                   "next empty box: "
                   + ", ".join(f"{a} {'≥' if op == '>=' else '≤'} {v:g}"
                               for a, op, v in cons)),
            constraints=cons, color=colours[i % len(colours)],
            radius=float(row["radius"]), volume=float(row["volume"]))
    return out


def fig_regions(df, ladders, out, overlay=None, regions=None, panels=None,
                stem="fig2_regions"):
    """One row per region: two conditional planes and the constraint funnel.

    A panel shows only the paradigms that satisfy the constraints on the axes
    *not* drawn, so what is inside the box on screen is inside the region in
    the space. The funnel then reports the same thing as a count.

    Parameterised over the set of regions so that the boxes the geometry discovers
    can be put through exactly the same treatment as the two specified by hand.
    Anything else would compare a region drawn one way against a region drawn
    another, which is the comparison the figure exists to make fair.
    """
    sub = df.dropna(subset=PRINCIPAL)
    regions = REGIONS if regions is None else regions
    panels = panels or {"G1": [("x", "y"), ("x", "t")], "G2": [("x", "z"), ("z", "t")]}
    names = list(regions)
    fig = plt.figure(figsize=(3 * PANEL_W + 1.5, len(names) * PANEL_W + 1.2))
    gs = fig.add_gridspec(len(names), 3, width_ratios=[1, 1, 1.35],
                          wspace=0.45, hspace=0.62)
    counts = {}
    for i, name in enumerate(names):
        spec = regions[name]
        members = satisfies(sub, spec["constraints"])
        counts[name] = int(members.sum())
        for j, (a, b) in enumerate(panels.get(name, [("x", "y"), ("x", "t")])):
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
            scatter_overlay(ax, overlay, a, b)
            if LABEL_REGIONS:
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
        # the funnel already counts the region row by row, so the title names what
        # the region is rather than repeating its label and its occupancy
        head = f"{name}  ·  {spec['title']}" if LABEL_REGIONS else spec["title"]
        ax.set_title("\n".join(textwrap.wrap(head, 44)),
                     fontsize=7.2, color=spec["color"], loc="left", pad=4)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    corpus_legend(fig, ncol=4, y=-0.06, context=True,
                  overlay=overlay is not None and len(overlay) > 0)
    save(fig, out, stem)
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


def fig_cube(df, ladders, out, overlay=None, cfg=None):
    """Two 3D views, each carrying every constraint of the region it shows.

    Choosing the triple this way is the whole point: in (x, y, t) the G1 box is
    the region, not its shadow, so a marker inside the box is a paradigm inside
    the region and the picture cannot mislead.

    Drawn in the same idiom as figure 0: the walls carry the marginal density of the
    pair they span, so each view shows both where the corpus is in three dimensions
    and what its projections look like, and the region box stays quiet enough to read
    the points through it.
    """
    sub = df.dropna(subset=PRINCIPAL)
    sigma = getattr(cfg, "sigma", SIGMA) if cfg is not None else SIGMA
    views = [("G1", ("x", "y", "t"), 19, -56), ("G2", ("x", "z", "t"), 19, -56)]
    shade = matplotlib.colors.LinearSegmentedColormap.from_list(
        "wall", ["#FFFFFF", "#D6DEE3", "#93A6B0"])
    fig = plt.figure(figsize=(7.4, 3.8))
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

        P = sub[[a, b, c]].to_numpy(float)
        w = sub["w"].to_numpy(float)
        lin = np.linspace(0, 1, 60)
        for idx, zdir, offset in (((0, 1), "z", 0.0), ((0, 2), "y", 1.0),
                                  ((1, 2), "x", 0.0)):
            d = density(P, w, idx, (lin, lin), sigma)
            d = d / d.max()
            G1_, G2_ = np.meshgrid(lin, lin, indexing="ij")
            args = {"z": (G1_, G2_, d), "y": (G1_, d, G2_), "x": (d, G1_, G2_)}[zdir]
            ax.contourf(*args, zdir=zdir, offset=offset,
                        levels=np.linspace(0.04, 1, 9), cmap=shade, alpha=0.55,
                        zorder=0, antialiased=True)

        _box3d(ax, [(lo[a], hi[a]), (lo[b], hi[b]), (lo[c], hi[c])], spec["color"], 0.09)
        inside = satisfies(sub, spec["constraints"]).to_numpy()
        pa, pb = spread(sub[a].to_numpy(float), sub[b].to_numpy(float))
        pc = sub[c].to_numpy(float)
        colors = np.array([CLUSTERS[c_]["color"] for c_ in sub["cluster"]])
        ax.scatter(pa[~inside], pb[~inside], pc[~inside], s=15,
                   c=colors[~inside], edgecolors="white", linewidths=0.3,
                   alpha=0.92, depthshade=True, zorder=4)
        if inside.any():
            # occupants are marked by a dark ring rather than by size: a region is
            # being argued to be nearly empty, and inflating the few points in it
            # works against the reader seeing that
            ax.scatter(pa[inside], pb[inside], pc[inside], s=20, c=colors[inside],
                       edgecolors=INK, linewidths=0.8, depthshade=True, zorder=6)
        if overlay is not None and len(overlay):
            ov = overlay.dropna(subset=[a, b, c])
            if len(ov):
                oa, ob = spread(ov[a].to_numpy(float), ov[b].to_numpy(float))
                ax.scatter(oa, ob, ov[c].to_numpy(float), s=42, marker="*",
                           color=CLUSTERS["thesis"]["color"], edgecolors="white",
                           linewidths=0.5, depthshade=False, zorder=7)
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
        head = (f"{'AB'[i]}  ({a}, {b}, {c}) — {name}: {int(inside.sum())} of "
                f"{len(sub)} inside" if LABEL_REGIONS else
                f"{'AB'[i]}  ({a}, {b}, {c})")
        ax.set_title(head, fontsize=7.4, loc="left", pad=-2)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor("white")
            pane.set_edgecolor(RULE)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"].update(color="#E7EBED", linewidth=0.5)
    corpus_legend(fig, ncol=4, y=0.015, confidence=False,
                  overlay=overlay is not None and len(overlay) > 0)
    save(fig, out, "fig3_cube")


# --- figure 0: what the space is -----------------------------------------

def _arrow3(ax, start, end, color=INK, lw=1.1):
    ax.plot(*zip(start, end), color=color, lw=lw, solid_capstyle="round", zorder=2)
    ax.scatter(*[[e] for e in end], s=12, color=color, marker="o", depthshade=False)


def fig_space(df, ladders, out, axes=("x", "y", "t"), cfg=None, overlay=None):
    """What the coordinates mean, and where the corpus sits in them.

    A gives each ladder its own column so the rung labels cannot collide, with the
    occupancy of every rung drawn as a bar to the left of the spine: the reader sees
    the definition and the coverage of an axis in one object. Rung labels wrap rather
    than truncate — a rung called "continuous control, non-…" is not a definition of
    anything, and the whole point of the panel is that the reader can check the coding
    against the axis. B places the corpus in three dimensions, with the marginal
    density of each pair cast onto the corresponding wall, so the shape of every
    projection is visible without hunting for it in another figure.
    """
    a, b, c = axes
    sub = df.dropna(subset=PRINCIPAL)
    fig = plt.figure(figsize=(7.6, 4.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.16, 1.16, 1.16, 2.72], wspace=0.05,
                          left=0.030, right=0.988, bottom=0.11, top=0.88)

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
                # the region bands are context, not content: narrow and pale, so the
                # ladder they annotate stays the thing the eye lands on
                ax.add_patch(Rectangle((-0.042, lo), 0.084, hi - lo,
                                       facecolor=spec["color"], alpha=0.10, lw=0,
                                       zorder=0))
                ax.text(-0.062, (lo + hi) / 2, slab, fontsize=4.8, rotation=90,
                        ha="center", va="center", color=spec["color"], alpha=0.85)
        # how much vertical room a label has before it reaches the rung below
        spacing = (min(np.diff(sorted(v for v, *_ in rungs))) if len(rungs) > 1
                   else 0.20)
        for (v, lab, _), n in zip(rungs, counts):
            if n:                                   # occupancy bar, drawn inward
                ax.add_patch(Rectangle((0, v - 0.019), 0.24 * n / top, 0.038,
                                       facecolor=CLUSTERS["thesis"]["color"], alpha=0.28,
                                       lw=0, zorder=2))
            ax.plot([-0.035, 0.035], [v, v], color=INK, lw=0.85, zorder=3,
                    solid_capstyle="butt")
            txt = re.sub(r"\s*/.*$", "", str(lab)).strip()
            lines = textwrap.wrap(txt, width=21, break_long_words=False) or [txt]
            # never let a label reach the next rung: two lines fit at the tightest
            # spacing any of these ladders uses, three do not
            room = max(int((spacing - 0.045) / 0.031), 1)
            if len(lines) > room:
                lines = lines[:room]
                lines[-1] = textwrap.shorten(" ".join(
                    textwrap.wrap(txt, width=21)[room - 1:]), width=20,
                    placeholder="…")
            ax.text(0.31, v + 0.014, f"{v:g}", fontsize=5.4, va="bottom", color=INK)
            ax.text(0.31, v - 0.014, "\n".join(lines), fontsize=5.1, va="top",
                    color="#6C7C85", linespacing=1.22)
            if n:
                ax.text(0.24 * n / top + 0.022, v, str(n), fontsize=4.8, va="center",
                        ha="left", color=CLUSTERS["thesis"]["color"])
        ax.set_xlim(-0.13, 1.52)
        ax.set_ylim(-0.075, 1.10)
        ax.axis("off")
        name = re.sub(r"\s*\(.*?\)\s*$", "", ladders.get(k, {}).get("name", "")).strip()
        ax.text(0, 1.155, label_of(k, short=True), fontsize=10, ha="center", color=INK)
        ax.text(0.26, 1.16, "\n".join(textwrap.wrap(
            name or FALLBACK_LABEL[k], width=22, break_long_words=False)[:2]),
            fontsize=5.5, ha="left", va="center", color="#6C7C85", linespacing=1.2)
        if i == 0:
            ax.text(-0.13, 1.245, "A   the rung ladders, and how many paradigms sit on "
                    "each rung", fontsize=7.6, ha="left", color=INK)

    # ---- B: the corpus in three dimensions, with its marginals on the walls
    ax3 = fig.add_subplot(gs[0, 3], projection="3d")
    pa, pb = spread(sub[a].to_numpy(float), sub[b].to_numpy(float))
    pc = sub[c].to_numpy(float)

    # each wall carries the exact marginal density of the pair it spans, rather than a
    # second copy of the scatter: with a hundred points on a lattice the shadow
    # scatters were mostly overplot, and the density is what the reader wants from a
    # projection anyway
    sigma = getattr(cfg, "sigma", SIGMA) if cfg is not None else SIGMA
    P = sub[[a, b, c]].to_numpy(float)
    w = sub["w"].to_numpy(float)
    lin = np.linspace(0, 1, 60)
    shade = matplotlib.colors.LinearSegmentedColormap.from_list(
        "wall", ["#FFFFFF", "#D6DEE3", "#93A6B0"])
    for idx, zdir, offset in (((0, 1), "z", 0.0), ((0, 2), "y", 1.0),
                              ((1, 2), "x", 0.0)):
        d = density(P, w, idx, (lin, lin), sigma)
        d = d / d.max()
        G1_, G2_ = np.meshgrid(lin, lin, indexing="ij")
        args = {"z": (G1_, G2_, d), "y": (G1_, d, G2_), "x": (d, G1_, G2_)}[zdir]
        ax3.contourf(*args, zdir=zdir, offset=offset, levels=np.linspace(0.04, 1, 9),
                     cmap=shade, alpha=0.55, zorder=0, antialiased=True)

    for name, spec in REGIONS.items():
        if not all(k in (a, b, c) for k, _, _ in spec["constraints"]):
            continue
        lo = {k: 0.0 for k in (a, b, c)}
        hi = {k: 1.0 for k in (a, b, c)}
        for k, op, v in spec["constraints"]:
            lo[k] = max(lo[k], v) if op == ">=" else lo[k]
            hi[k] = min(hi[k], v) if op == "<=" else hi[k]
        _box3d(ax3, [(lo[a], hi[a]), (lo[b], hi[b]), (lo[c], hi[c])], spec["color"], 0.09)
        inside = int(satisfies(sub, spec["constraints"]).sum())
        if LABEL_REGIONS:
            # a small tag on the corner of the box rather than a heavy label over it,
            # which competes with the points it is meant to be counting
            ax3.text(hi[a], lo[b], hi[c] + 0.04, f"{name} · {inside}", fontsize=5.8,
                     color=spec["color"], ha="right", va="bottom", zorder=8)

    for cl, style in CLUSTERS.items():
        m = (sub["cluster"] == cl).to_numpy()
        if m.any():
            ax3.scatter(pa[m], pb[m], pc[m], s=15, color=style["color"], alpha=0.95,
                        edgecolors="white", linewidths=0.3, depthshade=True, zorder=5)
    if overlay is not None and len(overlay):
        ov = overlay.dropna(subset=[a, b, c])
        if len(ov):
            oa, ob = spread(ov[a].to_numpy(float), ov[b].to_numpy(float))
            ax3.scatter(oa, ob, ov[c].to_numpy(float), s=42, marker="*",
                        color=CLUSTERS["thesis"]["color"], edgecolors="white",
                        linewidths=0.5, depthshade=False, zorder=7)
    for k, nm in zip((a, b, c), ("x", "y", "z")):
        getattr(ax3, f"set_{nm}lim")(0, 1)
        getattr(ax3, f"set_{nm}ticks")([0, 0.5, 1])
        getattr(ax3, f"set_{nm}ticklabels")(["0", ".5", "1"])
    ax3.set_xlabel(label_of(a), labelpad=-2, fontsize=6.6)
    ax3.set_ylabel(label_of(b), labelpad=-2, fontsize=6.6)
    ax3.set_zlabel(label_of(c), labelpad=-2, fontsize=6.6)
    ax3.tick_params(pad=0.4, labelsize=6.0)
    ax3.set_box_aspect((1, 1, 1))
    ax3.view_init(elev=19, azim=-56)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor(RULE)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis._axinfo["grid"].update(color="#E7EBED", linewidth=0.5)
    ax3.text2D(0.02, 1.03, f"B   the corpus on ({a}, {b}, {c}); each wall carries the "
               f"marginal density\n       of the {len(sub)} paradigms on the pair it "
               f"spans", transform=ax3.transAxes, fontsize=7.6, va="bottom", color=INK)
    corpus_legend(fig, ncol=4, y=0.005, confidence=False,
                  overlay=overlay is not None and len(overlay) > 0)
    save(fig, out, "fig0_space")


# --- figure 4: motor vs non-motor ----------------------------------------

def fig_task_axes(df, ladders, out, overlay=None):
    sub = df.dropna(subset=["x", "y", "x1", "y1"])
    fig, axs = plt.subplots(1, 3, figsize=(3 * PANEL_W + 1.5, PANEL_W + 1.0))
    square(axs[0], label_of("x", ladders), label_of("y", ladders))
    scatter_corpus(axs[0], sub, "x", "y")
    scatter_overlay(axs[0], overlay, "x", "y")
    axs[0].set_title("A  motor plane", loc="left", fontsize=7.6)
    square(axs[1], label_of("x1", ladders), label_of("y1", ladders))
    scatter_corpus(axs[1], sub, "x1", "y1")
    scatter_overlay(axs[1], overlay, "x1", "y1")
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
    corpus_legend(fig, ncol=4, y=-0.10,
                  overlay=overlay is not None and len(overlay) > 0)
    save(fig, out, "fig4_task_axes")
    return float(d.mean())


# --- figure 5: coverage deficit and gaps ---------------------------------

def fig_gaps(df, cfg, ladders, gaps, out, overlay=None):
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
        scatter_overlay(ax, overlay, a, b, size=26)
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


# --- figure 6b: the account field over the design space -------------------
#
# Figure 6 asks what the field would call one design. This one asks how the
# accounts are laid out over the space, which is the question the gap argument
# needs: a region can be empty of papers and still sit inside a part of account
# space that is already owned, and filling it would then settle nothing.

def _live_accounts(F, floor=0.03, peak=0.30):
    """Accounts worth a panel: either present everywhere or dominant somewhere.

    Mean mass alone keeps accounts that never lead anywhere and only ever appear as
    a few per cent of a distribution, which buys a near-white panel; the peak test
    keeps an account that owns one corner of the plane and nothing else.
    """
    flat = F.reshape(-1, F.shape[-1])
    return [a for i, a in enumerate(ACCOUNTS)
            if flat[:, i].mean() > floor or flat[:, i].max() > peak]


def _dominant_rgba(F, n_eff, min_neff=ACCOUNT_MIN_NEFF):
    """Colour by the leading account, opacity by its margin, grey where unsupported."""
    idx, margin = account_dominant(F)
    rgba = np.zeros(F.shape[:2] + (4,))
    for k, a in enumerate(ACCOUNTS):
        m = idx == k
        if not m.any():
            continue
        rgba[m, :3] = matplotlib.colors.to_rgb(ACCOUNT_COLOR[a])
        rgba[m, 3] = 0.30 + 0.70 * np.clip(margin[m] / 0.5, 0, 1)
    thin = n_eff < min_neff
    rgba[thin, :3] = matplotlib.colors.to_rgb("#E7EBED")
    rgba[thin, 3] = 0.85
    return rgba


def _overlay(ax, df, a, b, gaps=None, region=None, points=True):
    if points:
        pa, pb = spread(df[a].to_numpy(float), df[b].to_numpy(float))
        ax.scatter(pa, pb, s=4.5, color=INK, alpha=0.45, linewidths=0, zorder=4)
    if region:
        (xa, xb, _) = rect_on_plane(REGIONS[region]["constraints"], a, b)
        ax.add_patch(Rectangle((xa[0], xb[0]), xa[1] - xa[0], xb[1] - xb[0],
                               facecolor="none", edgecolor=INK, lw=0.9, ls=(0, (3, 2)),
                               zorder=5))
    if gaps is not None and len(gaps):
        for kind, marker, size in (("frontier", "o", 34), ("island", "*", 60)):
            g = gaps[gaps["kind"] == kind]
            if not len(g):
                continue
            ax.scatter(g[a], g[b], s=size, marker=marker, facecolors="none",
                       edgecolors="#3B2E80", linewidths=1.1, zorder=6)


def fig_account_field(df, ladders, gaps, out, plane=("x", "y"), n=61, sigma=SIGMA):
    a, b = plane
    lin, F, n_eff = account_plane(df, a, b, n=n, sigma=sigma)
    live = _live_accounts(F)
    ext = [0, 1, 0, 1]
    xlab, ylab = label_of(a, short=True), label_of(b, short=True)

    ncol = 4
    nrow_small = int(np.ceil(len(live) / ncol))
    fig = plt.figure(figsize=(7.4, 2.55 + 2.05 * nrow_small))
    gs = fig.add_gridspec(1 + nrow_small, 12, hspace=0.62, wspace=1.5,
                          height_ratios=[1.16] + [1] * nrow_small)

    # A: which account owns each design
    ax = fig.add_subplot(gs[0, 0:4])
    ax.imshow(np.transpose(_dominant_rgba(F, n_eff), (1, 0, 2)), origin="lower",
              extent=ext, interpolation="bilinear", zorder=1)
    square(ax, xlab, ylab)
    _overlay(ax, df.dropna(subset=[a, b]), a, b, gaps, region="G1")
    panel_title(ax, "A  which account owns each design", width=26)

    # B: how contested it is
    ax = fig.add_subplot(gs[0, 4:8])
    H = account_entropy(F)
    H = np.where(n_eff < ACCOUNT_MIN_NEFF, np.nan, H)
    cmap = matplotlib.colormaps["PuBuGn"].with_extremes(bad="#E7EBED")
    im = ax.imshow(H.T, origin="lower", extent=ext, vmin=0, vmax=1, cmap=cmap,
                   interpolation="bilinear", zorder=1)
    square(ax, xlab, ylab)
    _overlay(ax, df.dropna(subset=[a, b]), a, b, gaps, region="G1")
    panel_title(ax, "B  how contested that ownership is", width=28)
    # the label rides above a shortened bar rather than beside it: a rotated label
    # is wide enough at this figure size to reach the panel to its right, and a
    # full-height bar puts its own caption level with the next panel's title
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03, ticks=[0, 0.5, 1],
                      shrink=0.70, anchor=(0.0, 0.0))
    cb.ax.set_title("entropy of $f$", fontsize=6.0, color="#42525B", pad=4)
    cb.ax.tick_params(labelsize=5.8)
    cb.outline.set_linewidth(0.4)

    # C: how much corpus is actually behind the estimate
    ax = fig.add_subplot(gs[0, 8:12])
    im = ax.imshow(n_eff.T, origin="lower", extent=ext, cmap="Greys",
                   vmin=0, vmax=max(np.nanmax(n_eff), 1), interpolation="bilinear",
                   zorder=1)
    ax.contour(lin, lin, n_eff.T, levels=[ACCOUNT_MIN_NEFF], colors=["#C1425A"],
               linewidths=0.9, zorder=2)
    square(ax, xlab, ylab)
    _overlay(ax, df.dropna(subset=[a, b]), a, b, gaps, region="G1", points=False)
    panel_title(ax, "C  rows behind the estimate", width=28,
                sub=f"red contour: effective $n$ = {ACCOUNT_MIN_NEFF:g}")
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03, shrink=0.70,
                      anchor=(0.0, 0.0))
    cb.ax.set_title("effective $n$", fontsize=6.0, color="#42525B", pad=4)
    cb.ax.tick_params(labelsize=5.8)
    cb.outline.set_linewidth(0.4)

    # the individual account fields, on the same plane and the same colour scale
    for i, acc in enumerate(live):
        r, c = divmod(i, ncol)
        ax = fig.add_subplot(gs[1 + r, 3 * c:3 * c + 3])
        k = ACCOUNTS.index(acc)
        Fa = np.where(n_eff < ACCOUNT_MIN_NEFF, np.nan, F[:, :, k])
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            acc, ["#FFFFFF", ACCOUNT_COLOR[acc]]).with_extremes(bad="#E7EBED")
        ax.imshow(Fa.T, origin="lower", extent=ext, vmin=0, vmax=1, cmap=cmap,
                  interpolation="bilinear", zorder=1)
        ax.contour(lin, lin, Fa.T, levels=[0.5], colors=["white"], linewidths=0.8,
                   zorder=2)
        # a panel with nothing underneath it needs its own tick labels, whether or not
        # it happens to be in the last row of the grid: the final row is short
        last = i + ncol >= len(live)
        square(ax, xlab if last else "", ylab if c == 0 else "")
        if not last:
            ax.set_xticklabels([])
        if c != 0:
            ax.set_yticklabels([])
        sub = df.dropna(subset=[a, b])
        own = sub[sub["account"] == acc]
        if len(own):
            pa, pb = spread(own[a].to_numpy(float), own[b].to_numpy(float))
            ax.scatter(pa, pb, s=5, color=INK, alpha=0.6, linewidths=0, zorder=4)
        panel_title(ax, f"{acc} — {ACCOUNT_LABEL[acc]}", width=19, fontsize=6.6,
                    sub=f"{len(own)} rows, dominant")

    handles = [Patch(facecolor=ACCOUNT_COLOR[a_], edgecolor="none",
                     label=f"{a_}  {ACCOUNT_LABEL[a_]}") for a_ in live]
    handles.append(Patch(facecolor="#E7EBED", edgecolor="none",
                         label=f"effective $n$ < {ACCOUNT_MIN_NEFF:g}"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.035), handletextpad=0.4, columnspacing=1.4,
               fontsize=6.6)
    save(fig, out, "fig6b_account_field")
    return dict(plane=plane, live=live, entropy=float(np.nanmean(H)),
                n_eff_median=float(np.median(n_eff)))


# --- figure 6c: the account field where the argument needs it -------------

def fig_account_probes(df, cfg, gaps, out, clus=None, sigma=SIGMA):
    """f read off at the points the prose quotes, and per-cluster composition.

    Panel A is the account-space version of the occupancy funnel: if the bar over
    the gap looks like the bar over the cluster next to it, the gap is empty of
    papers but not of theory.
    """
    axes = cfg.axes
    probes = {}
    # short names only: at ten probes across a text-width figure a tick label is
    # about fifty points wide, which is one word. The coordinate of every probe is
    # in account_probes.csv and in the report, and the gaps are in table 2.
    for name in ("motor", "surprise", "bridge"):
        m = df["cluster"] == name
        sub = df[m].dropna(subset=axes)
        if len(sub):
            probes[f"{name}\ncentroid"] = sub[axes].mean().to_numpy()
    for name in REGIONS:
        probes[f"{name}\ncentroid"] = centroid(name, axes)
    seen = {}
    for _, g in gaps.iterrows():
        seen[g["kind"]] = seen.get(g["kind"], 0) + 1
        probes[f"{g['kind']}\ngap {seen[g['kind']]}"] = np.array([g[a] for a in axes])
    thesis = df[df["thesis"]].dropna(subset=axes)
    if len(thesis):
        probes["thesis\nparadigm"] = thesis[axes].mean().to_numpy()

    tab = account_probes(df, probes, axes=axes, sigma=sigma)
    live = [a for a in ACCOUNTS if tab[a].max() > 0.02]

    have_clus = clus is not None
    fig = plt.figure(figsize=(7.4, 6.0 if have_clus else 3.8))
    gs = fig.add_gridspec(2 if have_clus else 1, 2,
                          height_ratios=[1.3, 1][:2 if have_clus else 1],
                          hspace=0.78, wspace=0.32)

    # A: f at every probe, stacked
    ax = fig.add_subplot(gs[0, :])
    bottom = np.zeros(len(tab))
    xs = np.arange(len(tab))
    for acc in live:
        v = tab[acc].to_numpy(float)
        ax.bar(xs, v, bottom=bottom, width=0.66, color=ACCOUNT_COLOR[acc],
               edgecolor="white", linewidth=0.5, label=acc)
        for xi, (vi, bi) in enumerate(zip(v, bottom)):
            if vi > 0.13:
                ax.text(xi, bi + vi / 2, acc, ha="center", va="center", fontsize=5.8,
                        color="white", fontweight="bold")
        bottom += v
    ax.set_xticks(xs)
    ax.set_xticklabels(tab["probe"], fontsize=5.8, linespacing=1.35)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"$P(c \mid u)$")
    for xi, (ne, h) in enumerate(zip(tab["n_eff"], tab["entropy"])):
        ax.text(xi, 1.015, f"$n_{{\\rm eff}}$ {ne:.1f}\n$H$ {h:.2f}", ha="center",
                va="bottom", fontsize=5.4, color="#6C7C85")
    panel_title(ax, "A  what the field would call each of these designs", width=90,
                fontsize=7.6, pad=18)
    ax.legend(fontsize=6.0, ncol=min(len(live), 8), loc="lower center",
              bbox_to_anchor=(0.5, -0.40), frameon=False, columnspacing=1.0,
              handlelength=1.1)

    if have_clus:
        # B: composition of the hand labels
        ax = fig.add_subplot(gs[1, 0])
        comp = account_composition(df, df["cluster"], axes=axes)
        order = [c for c in CLUSTERS if c in comp.index]
        _stack(ax, comp.loc[order], live,
               [f"{CLUSTERS[c]['label']}\nn = {int(comp.loc[c, 'n'])}" for c in order])
        panel_title(ax, "B  accounts inside each hand label", width=30)

        # C: composition of the partition the geometry found
        ax = fig.add_subplot(gs[1, 1])
        # cluster_corpus resets the index of its own working frame, so the labels are
        # matched back by paradigm id. Positional alignment happened to work while the
        # corpus was a plain range index and broke the moment rows were held out.
        by_id = dict(zip(clus["sub"]["paradigm_id"],
                         [f"c{v}" for v in clus["labels"]]))
        found = df["paradigm_id"].map(by_id)
        comp = account_composition(df, found, axes=axes)
        order = sorted(comp.index)
        _stack(ax, comp.loc[order], live,
               [f"{c}\nn = {int(comp.loc[c, 'n'])}" for c in order])
        panel_title(ax, "C  accounts inside each discovered cluster", width=30)

    save(fig, out, "fig6c_account_probes")
    return tab


def _stack(ax, comp, live, labels):
    xs = np.arange(len(comp))
    bottom = np.zeros(len(comp))
    for acc in live:
        v = comp[acc].to_numpy(float)
        ax.bar(xs, v, bottom=bottom, width=0.6, color=ACCOUNT_COLOR[acc],
               edgecolor="white", linewidth=0.5)
        for xi, (vi, bi) in enumerate(zip(v, bottom)):
            if vi > 0.14:
                ax.text(xi, bi + vi / 2, acc, ha="center", va="center", fontsize=5.6,
                        color="white", fontweight="bold")
        bottom += v
    ax.set_xticks(xs)
    ax.set_xticklabels(["\n".join(textwrap.wrap(l.split("\n")[0], 13,
                                                 break_long_words=False)
                                  + l.split("\n")[1:]) for l in labels],
                       fontsize=6.0, linespacing=1.3)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of account mass")


# --- figure 13: the discovered empty boxes in three dimensions --------------

def fig_empty_boxes_3d(df, regions, ladders, out, cfg=None, overlay=None):
    """The boxes of method 2, drawn in the idiom of figures 0 and 3.

    A box on a plane is a shadow; a box in the cube is the region. Since the whole
    point of these boxes is that a design inside them meets nothing published, they
    are worth seeing as volumes with the corpus around them rather than as rectangles
    a point can appear to fall into from the wrong angle.
    """
    if not regions:
        return
    sub = df.dropna(subset=PRINCIPAL)
    sigma = getattr(cfg, "sigma", SIGMA) if cfg is not None else SIGMA
    shade = matplotlib.colors.LinearSegmentedColormap.from_list(
        "wall", ["#FFFFFF", "#D6DEE3", "#93A6B0"])
    names = list(regions)
    fig = plt.figure(figsize=(3.75 * len(names), 3.9))
    for i, name in enumerate(names):
        spec = regions[name]
        used = [a for a, _, _ in spec["constraints"]]
        trio = [a for a in PRINCIPAL if a in used][:3]
        while len(trio) < 3:
            trio += [a for a in PRINCIPAL if a not in trio][:1]
        a, b, c = trio
        ax = fig.add_subplot(1, len(names), i + 1, projection="3d")
        P = sub[[a, b, c]].to_numpy(float)
        w = sub["w"].to_numpy(float)
        lin = np.linspace(0, 1, 55)
        for idx, zdir, offset in (((0, 1), "z", 0.0), ((0, 2), "y", 1.0),
                                  ((1, 2), "x", 0.0)):
            d = density(P, w, idx, (lin, lin), sigma)
            d = d / d.max()
            GA, GB = np.meshgrid(lin, lin, indexing="ij")
            args = {"z": (GA, GB, d), "y": (GA, d, GB), "x": (d, GA, GB)}[zdir]
            ax.contourf(*args, zdir=zdir, offset=offset,
                        levels=np.linspace(0.04, 1, 9), cmap=shade, alpha=0.55,
                        zorder=0, antialiased=True)
        lo = {k: 0.0 for k in trio}
        hi = {k: 1.0 for k in trio}
        for k, op, v in spec["constraints"]:
            if k not in trio:
                continue
            lo[k] = max(lo[k], v) if op == ">=" else lo[k]
            hi[k] = min(hi[k], v) if op == "<=" else hi[k]
        _box3d(ax, [(lo[a], hi[a]), (lo[b], hi[b]), (lo[c], hi[c])], spec["color"], 0.13)
        inside = satisfies(sub, spec["constraints"]).to_numpy()
        pa, pb = spread(sub[a].to_numpy(float), sub[b].to_numpy(float))
        pc = sub[c].to_numpy(float)
        colors = np.array([CLUSTERS[k]["color"] for k in sub["cluster"]])
        ax.scatter(pa[~inside], pb[~inside], pc[~inside], s=15, c=colors[~inside],
                   edgecolors="white", linewidths=0.3, alpha=0.92, depthshade=True,
                   zorder=4)
        if inside.any():
            ax.scatter(pa[inside], pb[inside], pc[inside], s=20, c=colors[inside],
                       edgecolors=INK, linewidths=0.8, depthshade=True, zorder=6)
        if overlay is not None and len(overlay):
            ov = overlay.dropna(subset=[a, b, c])
            if len(ov):
                oa, ob = spread(ov[a].to_numpy(float), ov[b].to_numpy(float))
                ax.scatter(oa, ob, ov[c].to_numpy(float), s=42, marker="*",
                           color=CLUSTERS["thesis"]["color"], edgecolors="white",
                           linewidths=0.5, depthshade=False, zorder=7)
        for k, nm in zip(trio, ("x", "y", "z")):
            getattr(ax, f"set_{nm}lim")(0, 1)
            getattr(ax, f"set_{nm}ticks")([0, 0.5, 1])
            getattr(ax, f"set_{nm}ticklabels")(["0", ".5", "1"])
        ax.set_xlabel(label_of(a, short=True), labelpad=-4)
        ax.set_ylabel(label_of(b, short=True), labelpad=-4)
        ax.set_zlabel(label_of(c, short=True), labelpad=-4)
        ax.tick_params(pad=0.6, labelsize=6.0)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=19, azim=-56)
        off = [k for k, _, _ in spec["constraints"] if k not in trio]
        ax.set_title(f"{'ABCD'[i]}  ({a}, {b}, {c})"
                     + (f" — {', '.join(sorted(set(off)))} not shown" if off else ""),
                     fontsize=7.2, loc="left", pad=-2)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor("white")
            pane.set_edgecolor(RULE)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"].update(color="#E7EBED", linewidth=0.5)
    corpus_legend(fig, ncol=4, y=0.012, confidence=False,
                  overlay=overlay is not None and len(overlay) > 0)
    save(fig, out, "fig13_empty_boxes_3d")


# --- figure 14: joint low-density regions ---------------------------------

def fig_density_regions(df, cfg, res, ladders, out, planes=(("x", "y"), ("z", "t"))):
    """Method 3: the emptiest part of the joint density, as connected regions.

    Panel A is the honest part. How many distinct empty regions the corpus has is not
    a fact but a function of the level at which the density is cut, so the level is
    swept and the choice is shown rather than asserted.
    """
    tab, lab, lin = res["table"], res["labels"], res["lin"]
    sub = df.dropna(subset=PRINCIPAL)
    axes = cfg.axes
    fig = plt.figure(figsize=(7.4, 4.4))
    gs = fig.add_gridspec(1, 3, wspace=0.5, left=0.075, right=0.985, bottom=0.30,
                          top=0.82)

    # A -- how the component structure depends on the level
    ax = fig.add_subplot(gs[0, 0])
    sw = res["sweep"]
    ax.plot(sw["quantile"], sw["components"], "-o", ms=3.4, lw=1.1, color=INK)
    ax.axvline(res["quantile"], color="#C1425A", lw=1.0, ls="--")
    ax.text(res["quantile"], ax.get_ylim()[1], " level used", fontsize=5.8,
            color="#C1425A", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("density quantile used as the level")
    ax.set_ylabel("distinct low-density regions")
    panel_title(ax, "A  the level is a choice, so it is swept", width=30,
                sub="too high and the empty space is one region; too low and it "
                    "fragments")

    # B, C -- the components on two planes, with their inscribed boxes
    colours = ["#4B2E83", "#157F7F", "#D2892A", "#B0447A"]
    for j, (a, b) in enumerate(planes):
        ax = fig.add_subplot(gs[0, j + 1])
        ia, ib = axes.index(a), axes.index(b)
        for n, (_, row) in enumerate(tab.iterrows()):
            m = (lab == int(row["component"]))
            # a component is four-dimensional; on a plane it is shown as the set of
            # cells that project into it, which is a shadow and drawn faintly
            proj = m.any(axis=tuple(k for k in range(len(axes)) if k not in (ia, ib)))
            if ia > ib:
                proj = proj.T
            ax.contourf(lin, lin, proj.T.astype(float), levels=[0.5, 1.5],
                        colors=[colours[n % 4]], alpha=0.13, zorder=0)
            ax.add_patch(Rectangle(
                (row[f"{a}_lo"], row[f"{b}_lo"]),
                row[f"{a}_hi"] - row[f"{a}_lo"], row[f"{b}_hi"] - row[f"{b}_lo"],
                facecolor="none", edgecolor=colours[n % 4], lw=1.2, ls="--", zorder=4))
        square(ax, label_of(a, ladders, short=True), label_of(b, ladders, short=True))
        scatter_corpus(ax, sub, a, b, size=11)
        panel_title(ax, f"{'BC'[j]}  the regions on $({a}, {b})$", width=30,
                    sub="shaded: the region's shadow · dashed: its inscribed box")

    lines = []
    for n, (_, row) in enumerate(tab.iterrows()):
        lines.append(
            f"region {n + 1}: {row['volume'] * 100:.1f}% of the searchable volume, "
            f"{int(row['occupancy'])} paradigms inside its inscribed box "
            + ", ".join(f"{a} [{row[f'{a}_lo']:.2f}, {row[f'{a}_hi']:.2f}]"
                        for a in axes))
    fig.text(0.075, 0.175, "\n".join(textwrap.wrap("  ·  ".join(lines), 118)),
             fontsize=6.0, color="#42525B", va="top", linespacing=1.5)
    save(fig, out, "fig14_density_regions")


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
    sub, lab, given = res["sub"], res["labels"], res["given"]
    axes = res["axes"]
    # rung units throughout: res["centers"] and res["X_raw"] are the inverse of the
    # standardisation the clustering ran on, so every panel below shares the scale of
    # the scatter and of the ladders
    C, X = res["centers"], res.get("X_raw", res["X"])
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
                cx, cy = cc[axes.index(a)], cc[axes.index(b)]
                ax.scatter([cx], [cy], s=80, marker="X", color=CLUSTER_PALETTE[j % 8],
                           edgecolors=INK, linewidths=0.7, zorder=5)
                # anchor the tag on whichever side keeps it inside the square, and
                # clip it: a centroid near a corner used to push its label onto the
                # panel next door
                ha = "left" if cx < 0.82 else "right"
                va = "bottom" if cy < 0.88 else "top"
                ax.annotate(f"c{j}", (cx, cy), fontsize=6.8, ha=ha, va=va,
                            xytext=(4 if ha == "left" else -4,
                                    4 if va == "bottom" else -4),
                            textcoords="offset points", annotation_clip=True,
                            color=CLUSTER_PALETTE[j % 8], fontweight="bold", zorder=6)
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
    # the interquartile bars of different clusters land on the same tick, so each
    # cluster is dodged by a fraction of the spacing; the centroid line keeps the
    # same offset so a line and its spread stay visually attached
    dodge = 0.13 if len(C) > 1 else 0.0
    off = (np.arange(len(C)) - (len(C) - 1) / 2) * dodge
    for j, cc in enumerate(C):
        n = int((lab == j).sum())
        ax.plot(xs + off[j], cc, "-o", ms=4, lw=1.4, color=CLUSTER_PALETTE[j % 8],
                label=f"c{j}  (n = {n})", zorder=3)
        for xi in range(len(axes)):
            lo, hi = np.percentile(X[lab == j, xi], [25, 75])
            ax.plot([xi + off[j]] * 2, [lo, hi], color=CLUSTER_PALETTE[j % 8], lw=3.4,
                    alpha=0.25, solid_capstyle="round", zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([label_of(k, short=True) for k in axes], fontsize=8.5)
    ax.set_xlim(-0.5, len(axes) - 0.5)
    for xi, k in enumerate(axes):
        ax.text(xi, -0.20, textwrap.shorten(
            re.sub(r"\s*\(.*?\)\s*$", "", ladders.get(k, {}).get("name", "")
                   or FALLBACK_LABEL[k]), 20, placeholder="…"),
            fontsize=5.4, ha="center", color="#6C7C85", transform=ax.get_xaxis_transform())
    # the ladders run 0 to 1, so the panel does too, whatever units the clustering
    # itself used; headroom at the top is for the legend, not for data
    ax.set_ylim(-0.03, 1.28)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "", ".5", "", "1"])
    ax.set_ylabel("rung")
    ax.axhline(1.0, color=RULE, lw=0.5, zorder=0)
    ax.legend(fontsize=6.2, ncol=min(4, len(C)), loc="upper left", frameon=False,
              handlelength=1.4, columnspacing=1.1, borderpad=0.2)
    panel_title(ax, "E  centroid of each discovered cluster, with its interquartile "
                    "range", width=72, fontsize=7.6,
                sub="rung units; the partition itself was found on standardised axes")
    save(fig, out, "fig8_clusters")


# --- figure 9: is the partition real, and is there a valley in it? --------

def fig_separation(clus, diag, ladders, out):
    """The tests behind the two-literature claim, in one figure.

    A and B are the pair that matters, and only as a pair: A on its own is the
    circular version of the test, since the direction it projects onto was chosen to
    make the split look deep. B is what makes A admissible — the identical procedure
    run on unimodal clouds, which produce valleys of their own.
    """
    sep = diag["separation"]
    fig = plt.figure(figsize=(7.4, 5.2))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.12, 1], hspace=0.80, wspace=1.9)

    # A -- the corpus projected onto its own best two-way direction
    ax = fig.add_subplot(gs[0, 0:8])
    v = sep["proj"]
    lab2, given = sep["labels"], clus["given"]
    bins = np.linspace(v.min(), v.max(), 26)
    for j, col in ((0, "#1B3A5C"), (1, "#E08A2E")):
        share = ""
        if "composition" in sep:
            c = sep["composition"][f"h{j}"]
            top = sorted(c.items(), key=lambda kv: -kv[1])[:2]
            share = "  " + ", ".join(f"{n} {g}" for g, n in top if n)
        ax.hist(v[lab2 == j], bins=bins, color=col, alpha=0.62, zorder=2,
                label=f"half {j}  (n = {int((lab2 == j).sum())}){share}")
    order = np.sort(v)
    lo, hi = np.quantile(order, [0.10, 0.90])
    inner = order[(order >= lo) & (order <= hi)]
    if len(inner) > 2:
        j = int(np.argmax(np.diff(inner)))
        ax.axvspan(inner[j], inner[j + 1], color="#C1425A", alpha=0.16, zorder=1)
    ax.set_xlabel("projection onto the corpus's own best two-way direction")
    ax.set_ylabel("paradigms")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.32)
    ax.legend(fontsize=6.0, loc="upper right", frameon=False)
    panel_title(ax, "A  is the corpus thin between the two literatures?", width=52,
                fontsize=7.6,
                sub=(f"widest interior gap {sep['gap']:.2f} sd · two-component fit "
                     f"wins by {sep['delta_bic']:.0f} in BIC · {sep['tied_fraction']:.0%} "
                     f"of the projection is tied, which is the caveat on both"))

    # B -- the same procedure on unimodal data, which is what makes A a test
    ax = fig.add_subplot(gs[0, 8:12])
    ax.hist(sep["null_gap"], bins=22, color=GREY, alpha=0.75, zorder=1,
            label=f"unimodal reference\n({sep['n_null']} draws)")
    ax.axvline(sep["gap"], color="#C1425A", lw=1.3, zorder=3)
    pstr = ("$p$ < 0.01" if sep["p_gap"] < 0.01 else f"$p$ = {sep['p_gap']:.3f}")
    lo_x, hi_x = ax.get_xlim()
    ax.set_xlim(lo_x, max(hi_x, sep["gap"] * 1.08))
    right_room = (sep["gap"] - lo_x) / (ax.get_xlim()[1] - lo_x) < 0.7
    ax.text(sep["gap"], ax.get_ylim()[1] * 0.99,
            (" corpus\n " if right_room else "corpus \n") + pstr, fontsize=6.0,
            color="#C1425A", ha="left" if right_room else "right", va="top")
    ax.set_xlabel("widest interior gap (sd)")
    ax.set_ylabel("draws")
    ax.legend(fontsize=5.6, loc="upper left", frameon=False)
    panel_title(ax, "B  the same split, on data with no clusters in it", width=26,
                sub=f"BIC margin $p$ = {sep['p_bic']:.3f}")

    # C -- separation, and whether it is only a spread difference
    ax = fig.add_subplot(gs[1, 0:3])
    pa, pd_ = diag["permanova"], diag["permdisp"]
    rows = [("PERMANOVA", pa["F"], pa["p"]), ("PERMDISP", pd_["F"], pd_["p"])]
    ax.barh([0, 1], [r[1] for r in rows], color=["#4B2E83", "#AEB8BC"], height=0.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.4)
    ax.invert_yaxis()
    for i, (_, f, p) in enumerate(rows):
        ax.text(f + max(pa["F"], pd_["F"]) * 0.04, i,
                f"F = {f:.1f}\n" + ("p < .001" if p < 0.001 else f"p = {p:.3f}"),
                va="center", fontsize=5.8)
    ax.set_xlim(0, max(pa["F"], pd_["F"]) * 1.75)
    ax.set_xlabel("pseudo-$F$")
    panel_title(ax, "C  location, and spread", width=22,
                sub=f"$R^2$ = {pa['R2']:.2f}")

    # D -- which axes carry it
    ax = fig.add_subplot(gs[1, 3:6])
    tab = diag["axis"]
    xs = np.arange(len(tab))
    ax.bar(xs, tab["eta2"], width=0.6, color="#4B2E83", edgecolor="white", linewidth=0.5)
    ax.plot(xs, tab["eta2_null"], "o", ms=3.2, color=GREY, zorder=3, label="chance")
    for xi, (e, p) in enumerate(zip(tab["eta2"], tab["p"])):
        ax.text(xi, e + 0.03, "***" if p < 0.001 else "**" if p < 0.01
                else "*" if p < 0.05 else "n.s.", ha="center", fontsize=5.6,
                color=INK if p < 0.05 else "#6C7C85")
    ax.set_xticks(xs)
    ax.set_xticklabels([label_of(a, short=True) for a in tab["axis"]], fontsize=8)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel(r"$\eta^2$")
    ax.legend(fontsize=5.8, loc="upper right", frameon=False)
    panel_title(ax, "D  which axes carry the split", width=22)

    # E -- per-cluster stability
    ax = fig.add_subplot(gs[1, 6:9])
    jac = diag["jaccard"]
    keys = sorted(jac)
    ax.bar(range(len(keys)), [jac[g] for g in keys], width=0.6,
           color=[CLUSTER_PALETTE[g % 8] for g in keys], edgecolor="white",
           linewidth=0.5)
    ax.set_xlim(-0.62, len(keys) - 0.5 + 1.55)
    for lvl, txt in ((0.6, "dissolved"), (0.75, "stable")):
        ax.plot([-0.62, len(keys) - 0.46], [lvl, lvl], color=GREY, lw=0.6, ls="--",
                zorder=0)
        ax.text(len(keys) - 0.40, lvl, txt, fontsize=5.2, ha="left", va="center",
                color="#6C7C85")
    for i, g in enumerate(keys):
        ax.text(i, jac[g] + 0.02, f"{jac[g]:.2f}", ha="center", fontsize=5.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"c{g}" for g in keys], fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("mean Jaccard")
    panel_title(ax, "E  which clusters survive resampling", width=22)

    # F -- how many clusters replicate
    ax = fig.add_subplot(gs[1, 9:12])
    st = diag["strength"]
    ax.plot(st["k"], st["prediction_strength"], "-o", ms=3.2, lw=1.1, color=INK)
    ax.axhline(0.8, color="#C1425A", lw=0.7, ls="--", zorder=0)
    ax.text(st["k"].max(), 0.815, "cutoff", fontsize=5.4, ha="right", color="#C1425A")
    ax.axvline(clus["k"], color=CLUSTERS["surprise"]["color"], lw=0.9, ls=":", zorder=0)
    ax.set_xlabel("clusters $k$")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("prediction strength")
    panel_title(ax, "F  how many clusters replicate", width=22)

    save(fig, out, "fig9_separation")


def fig_feasible(df, cfg, empty, ladders, out):
    """Does the corpus obey the constraints that are supposed to be inviolable?

    The mask is the one part of the gap machinery that cannot be checked from its own
    output: it decides what the search is allowed to see, so anything it excludes is
    absent from every figure downstream and its exclusions look like emptiness. The
    only external check available is the corpus itself, and it is a real check —
    every published paradigm inside an excluded region is a counterexample to the claim
    that the region cannot be occupied.
    """
    axes = cfg.axes
    sub = df.dropna(subset=axes)
    tab, offenders = constraint_audit(sub, axes)
    fig = plt.figure(figsize=(7.4, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.46, left=0.07, right=0.985, bottom=0.34,
                          top=0.84)

    lin = np.linspace(0, 1, 200)
    for col, (a, b) in enumerate((("x", "y"), ("z", "t"))):
        ax = fig.add_subplot(gs[0, col])
        GA, GB = np.meshgrid(lin, lin, indexing="ij")
        U = np.full((GA.size, len(axes)), 0.5)
        U[:, axes.index(a)], U[:, axes.index(b)] = GA.ravel(), GB.ravel()
        ok = feasible(U, axes, enabled=True).reshape(GA.shape)
        ax.contourf(lin, lin, (~ok).T.astype(float), levels=[0.5, 1.5],
                    colors=["#F2DCE1"], zorder=0)
        ax.contour(lin, lin, (~ok).T.astype(float), levels=[0.5], colors=["#C1425A"],
                   linewidths=1.0, zorder=1)
        square(ax, label_of(a, ladders, short=True), label_of(b, ladders, short=True))
        scatter_corpus(ax, sub, a, b, size=13)
        viol = pd.concat([v for k, v in offenders.items()
                          if a in k.split() or b in k.split()] or
                         [sub.iloc[:0]]).drop_duplicates(subset="paradigm_id")
        viol = viol[~feasible(viol[axes].to_numpy(float), axes, enabled=True)]
        if len(viol):
            ax.scatter(viol[a], viol[b], s=110, facecolors="none",
                       edgecolors="#C1425A", linewidths=1.4, zorder=7)
            # violators can be coincident, so the labels are dealt out around the
            # marker rather than all placed below and to the right of it
            spots = [(9, 3), (9, -9), (-9, 6), (-9, -12), (9, 15)]
            for n, (_, r) in enumerate(viol.iterrows()):
                dx, dy = spots[n % len(spots)]
                ax.annotate(str(r["citekey"])[:18], (r[a], r[b]), fontsize=5.4,
                            color="#C1425A", xytext=(dx, dy),
                            ha="left" if dx > 0 else "right",
                            textcoords="offset points", zorder=8,
                            annotation_clip=False)
        panel_title(ax, f"{'AB'[col]}  the excluded region on ({a}, {b})", width=34,
                    sub=(f"{len(viol)} published paradigm"
                         f"{'' if len(viol) == 1 else 's'} inside it"))

    # C -- what the mask decides about the answer
    ax = fig.add_subplot(gs[0, 2])
    if empty is not None:
        masked = empty.get("masked_compare")
        rows = [("constraints off", empty["table"].iloc[0] if len(empty["table"])
                 else None, "#4B2E83")]
        if masked is not None and len(masked["table"]):
            rows.append(("constraints on", masked["table"].iloc[0], "#C1425A"))
        for i, (name, r, col) in enumerate(rows):
            if r is None:
                continue
            ax.barh([i], [r["radius"]], height=0.45, color=col, alpha=0.85)
            ax.text(r["radius"] + 0.012, i,
                    f"  r = {r['radius']:.2f}\n  ("
                    + ", ".join(f"{r[a]:.2f}" for a in axes) + ")",
                    va="center", fontsize=5.8, color=INK)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([n for n, _, _ in rows], fontsize=6.6)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.25)
        ax.set_xlabel("radius of the largest empty ball")
    panel_title(ax, "C  what the mask decides", width=30,
                sub="the deepest hole under each setting")

    lines = ["  ".join(
        f"{r['constraint']} — {r['violations']} violate it"
        for _, r in tab.iterrows())] if len(tab) else []
    fig.text(0.07, 0.18, "\n".join(textwrap.wrap(
        "Each constraint claims that a combination of coordinates cannot be realised. "
        "The corpus is the test of that claim, and it is a test the first two "
        "constraints fail. " + (lines[0] if lines else ""), 112)),
        fontsize=6.2, color="#42525B", va="top", linespacing=1.5)
    save(fig, out, "fig11_feasible")
    return tab


# --- figure 10: empty space, measured geometrically -----------------------

def fig_empty_space(df, cfg, res, ladders, out, plane=("x", "y")):
    """Where the holes are, how big they are, and which part of the cube is off-limits.

    Panel A is the one to look at first, because it answers the question this figure
    is most often asked: several apparently empty corners of the design space are not
    unexplored but impossible, and are excluded before any search runs.
    """
    axes = cfg.axes
    P, w, sub = matrix(df, axes)
    tab = res["table"]
    a, b = plane
    ia, ib = axes.index(a), axes.index(b)
    fig = plt.figure(figsize=(7.4, 5.1))
    gs = fig.add_gridspec(2, 12, height_ratios=[1, 0.78], hspace=0.52, wspace=2.7)

    # A -- the feasible set
    ax = fig.add_subplot(gs[0, 0:4])
    lin = np.linspace(0, 1, 200)
    GA, GB = np.meshgrid(lin, lin, indexing="ij")
    U = np.zeros((GA.size, len(axes)))
    U[:, ia], U[:, ib] = GA.ravel(), GB.ravel()
    for j, k in enumerate(axes):
        if j not in (ia, ib):
            U[:, j] = 0.5
    ok = feasible(U, axes).reshape(GA.shape)
    ax.contourf(lin, lin, (~ok).T.astype(float), levels=[0.5, 1.5],
                colors=["#E7EBED"], zorder=0)
    ax.contour(lin, lin, (~ok).T.astype(float), levels=[0.5], colors=["#AEB8BC"],
               linewidths=0.8, zorder=1)
    square(ax, label_of(a, ladders, short=True), label_of(b, ladders, short=True))
    scatter_corpus(ax, sub, a, b, size=10)
    ax.text(0.16, 0.86, "structurally\nimpossible", fontsize=6.0, color="#6C7C85",
            ha="center", va="center", zorder=4)
    panel_title(ax, "A  what the search is allowed to look at", width=30,
                sub=f"{res['feasible_fraction']:.0%} of the cube is feasible")

    # B -- distance to the nearest published paradigm, through the deepest hole
    ax = fig.add_subplot(gs[0, 4:8])
    if len(tab):
        centre = np.array([tab.iloc[0][k] for k in axes])
        U2 = np.tile(centre, (GA.size, 1))
        U2[:, ia], U2[:, ib] = GA.ravel(), GB.ravel()
        D = np.sqrt(((U2[:, None, :] - P[None, :, :]) ** 2).sum(-1)).min(1)
        D = np.where(feasible(U2, axes), D, np.nan).reshape(GA.shape)
        cmap = matplotlib.colormaps["BuPu"].with_extremes(bad="#E7EBED")
        im = ax.contourf(lin, lin, D.T, levels=12, cmap=cmap, zorder=0)
        for i, r in tab.iterrows():
            ax.add_patch(Circle((r[a], r[b]), r["radius"], facecolor="none",
                                edgecolor="#C1425A", lw=1.0,
                                ls="-" if i == 0 else (0, (2, 2)), zorder=4))
            ax.scatter([r[a]], [r[b]], s=18, marker="+", color="#C1425A", zorder=5)
        cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02, shrink=0.62,
                          anchor=(0, 0))
        cb.ax.set_title("distance", fontsize=5.8, color="#42525B", pad=3)
        cb.ax.tick_params(labelsize=5.2)
        cb.outline.set_linewidth(0.4)
    square(ax, label_of(a, ladders, short=True), label_of(b, ladders, short=True))
    scatter_corpus(ax, sub, a, b, size=10)
    panel_title(ax, "B  distance to the nearest published paradigm", width=30,
                sub="slice through the largest empty ball")

    # C -- is the biggest hole bigger than chance leaves?
    ax = fig.add_subplot(gs[0, 8:12])
    ax.hist(res["null"], bins=20, color=GREY, alpha=0.75,
            label=f"{len(res['null'])} uniform corpora\nof the same size")
    ax.axvline(res["radius"], color="#C1425A", lw=1.3)
    ax.text(res["radius"], ax.get_ylim()[1] * 0.98, " observed\n "
            + ("$p$ < 0.01" if res["p"] < 0.01 else f"$p$ = {res['p']:.3f}"),
            fontsize=6.0, color="#C1425A", ha="right", va="top")
    ax.set_xlabel("radius of the largest empty ball")
    ax.set_ylabel("draws")
    ax.legend(fontsize=5.6, loc="upper left", frameon=False)
    panel_title(ax, "C  bigger than random scatter leaves?", width=30)

    # D -- the holes as design specifications
    ax = fig.add_subplot(gs[1, :])
    if len(tab):
        n = len(tab)
        colors = ["#4B2E83", "#C1425A", "#157F7F", "#D2892A"]
        for j, k in enumerate(axes):
            base = j * (n + 1.1)
            for i, r in tab.iterrows():
                y = base + i
                ax.barh(y, r[f"{k}_hi"] - r[f"{k}_lo"], left=r[f"{k}_lo"], height=0.62,
                        color=colors[i % 4], alpha=0.75, zorder=2)
                ax.plot([r[k]], [y], marker="|", ms=7, color="white", zorder=3)
            ax.text(-0.035, base + (n - 1) / 2, f"${k}$", fontsize=9, ha="right",
                    va="center", color=INK)
        ax.set_yticks([])
        ax.set_xlim(-0.005, 1.005)
        ax.set_ylim(-0.8, len(axes) * (n + 1.1) - 0.6)
        ax.set_xlabel("rung")
        ax.invert_yaxis()
        for sp in ("left", "right", "top"):
            ax.spines[sp].set_visible(False)
        handles = [Patch(facecolor=colors[i % 4], alpha=0.75,
                         label=f"hole {i + 1}: radius {r['radius']:.2f}, "
                               f"box volume {r['volume']:.3f}"
                               + (f", inside {r['in_region']}"
                                  if r["in_region"] != "—" else ""))
                   for i, r in tab.iterrows()]
        ax.legend(handles=handles, fontsize=6.0, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.30), frameon=False)
    panel_title(ax, "D  the same holes as maximal empty boxes: the range of each "
                    "coordinate a design could take and still meet nothing published",
                width=96, fontsize=7.6)
    save(fig, out, "fig10_empty_space")


# --- figure 8b: both partitions, and the gaps, on one plane ---------------

def fig_partitions(clus, gaps, ladders, out, planes=(("x", "y"), ("x", "t")),
                   overlay=None):
    """The discovered clusters as territory, the assigned labels as points.

    Figure 8 puts the two partitions in adjacent panels, which asks the reader to
    hold one in memory while looking at the other. Here the clusters are drawn as
    the region of the plane they occupy and the hand labels keep the points, so
    disagreement is visible directly: a point of one colour inside another cluster's
    territory is a paradigm the geometry and the reader classify differently. The
    gaps are on the same axes, which is the only way to see whether a gap lies
    between the clusters or beyond them.
    """
    sub, lab, given = clus["sub"], clus["labels"], clus["given"]
    fig, axs = plt.subplots(1, len(planes),
                            figsize=(len(planes) * PANEL_W + 1.6, PANEL_W + 1.05))
    axs = np.atleast_1d(axs)
    for ax, (a, b) in zip(axs, planes):
        square(ax, label_of(a, ladders), label_of(b, ladders))
        pts = sub[[a, b]].to_numpy(float)
        for j in np.unique(lab):
            m = lab == j
            col = CLUSTER_PALETTE[j % 8]
            hull_pts = pts[m]
            if len(np.unique(hull_pts, axis=0)) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    h = ConvexHull(hull_pts)
                    poly = hull_pts[h.vertices]
                    ax.fill(poly[:, 0], poly[:, 1], color=col, alpha=0.13, lw=0,
                            zorder=1)
                    ax.plot(np.append(poly[:, 0], poly[0, 0]),
                            np.append(poly[:, 1], poly[0, 1]), color=col, lw=0.9,
                            alpha=0.65, zorder=2)
                except Exception:
                    pass
            c = hull_pts.mean(0)
            ax.scatter([c[0]], [c[1]], s=70, marker="X", color=col, edgecolors=INK,
                       linewidths=0.6, zorder=6)
            ax.annotate(f"c{j}", (c[0], c[1]), fontsize=6.6, fontweight="bold",
                        color=col, xytext=(5, 5), textcoords="offset points",
                        annotation_clip=True, zorder=7)
        scatter_corpus(ax, sub, a, b, size=16, zorder=4)
        scatter_overlay(ax, overlay, a, b, size=40)
        for _, g in gaps.iterrows():
            marker = "o" if g["kind"] == "frontier" else "*"
            ax.scatter(g[a], g[b], marker=marker, s=42 if marker == "o" else 90,
                       facecolors="none" if marker == "o" else "#3B2E80",
                       edgecolors="#3B2E80", linewidths=1.1, zorder=8)
        ax.set_title(f"({a}, {b})", loc="left", fontsize=7.6)
    handles = [Patch(facecolor=CLUSTER_PALETTE[j % 8], alpha=0.30,
                     label=f"cluster c{j} ({int((lab == j).sum())})")
               for j in np.unique(lab)]
    present = set(sub["cluster"])
    handles += [Line2D([], [], marker="o", ls="", color=s["color"],
                       markeredgecolor="white", markersize=5, label=s["label"])
                for k, s in CLUSTERS.items() if k in present]
    if overlay is not None and len(overlay):
        handles.append(Line2D([], [], marker="*", ls="",
                              color=CLUSTERS["thesis"]["color"],
                              markeredgecolor="white", markersize=8,
                              label="thesis paradigms (held out)"))
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor="none",
                       markeredgecolor="#3B2E80", markersize=5, label="frontier gap"),
                Line2D([], [], marker="*", ls="", color="#3B2E80", markersize=8,
                       label="island gap")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.10),
               fontsize=6.4, handletextpad=0.3, columnspacing=1.2)
    fig.suptitle("shaded territory: the partition the geometry found  ·  point colour: "
                 "the label assigned by hand", fontsize=7.6, y=1.005, color="#54646D")
    save(fig, out, "fig8b_partitions")


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


def html_payload(df, ladders, cfg, gaps, raw, clus=None, holes=None):
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
    pcols = [f"p_{a}" for a in ACCOUNTS]
    have_p = set(pcols) <= set(df.columns)
    for _, r in df.iterrows():
        # the page estimates f itself so that it can react to sigma and to the plane,
        # which means it needs the distribution over accounts rather than the argmax:
        # with one-hot labels the same kernel regression collapses onto whichever
        # account the two or three nearest rows happen to carry
        pa = [float(r[c]) if have_p and np.isfinite(r[c]) else 0.0 for c in pcols]
        if sum(pa) <= 0 and r["account"] in ACCOUNTS:
            pa = [1.0 if a == r["account"] else 0.0 for a in ACCOUNTS]
        s = sum(pa)
        pa = [round(v / s, 4) for v in pa] if s > 0 else None
        pts.append({
            "id": r["paradigm_id"], "key": r["citekey"] or r["paradigm_id"],
            "title": r["title"][:90], "year": num(r["year"]),
            "cluster": r["cluster"], "conf": r["confidence"], "w": num(r["w"]),
            "acc": r["account"], "pacc": pa, "topic": r["topic"],
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
        "accountLabel": ACCOUNT_LABEL,
        "accountColor": ACCOUNT_COLOR,
        "accountMinNeff": ACCOUNT_MIN_NEFF,
        "gaps": gaps.to_dict("records"),
        "holes": (holes.to_dict("records") if holes is not None else []),
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
  <p class="sub">The motor control and environmental surprise literature have
     developed through distinct and largely independent experimental traditions. 
     Consequently, studies are often compared through verbal descriptions or 
     theoretical labels rather than through an explicit characterization of their 
     task structures. To make these comparisons more systematic, we develop a 
     formal representation of the experimental paradigms reviewed in this thesis.
     <span class="coord">thesis paradigm · __THESISCOORD__</span></p>
     <p class="sub">__CLUSMETA__</p>
</header>

<section>
  <p class="eyebrow">step 1 — the coordinates</p>
  <h2>Geometrical representation of the experimental paradigms</h2>
  <p class="lead"> Every experimental design is a point in the low-dimensional space.
       Among different axes, we can consider how much capacity the task commits (motor task difficulty),
       how long the motor command stays open to revision (motor task timescale), how deep a hierarchy the
       perturbing event's statistics demand (surprise hierarchy), and how much of the task the event carries 
       (task relevance of the surprise).
       One row of the workbook is one paradigm, so a paper running four experiments occupies
       four points. </p>
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

<section id="fitsec">
  <p class="eyebrow">step 3 — the labels, checked against the geometry</p>
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
  <p class="eyebrow">step 3b — emptiness as a low-density basin</p>
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
      <button class="seg" id="tgap" aria-pressed="true">single-point gaps</button>
      <button class="seg" id="tpts" aria-pressed="true">paradigms</button></div>
    <div class="ctl"><label>boxes</label>
      <button class="seg" id="greg" aria-pressed="false">candidate regions</button>
      <button class="seg" id="ghole" aria-pressed="false">empty boxes</button></div>
    <div class="readout">reachability of the box<b id="reach">—</b>
      <span id="reachnote"></span></div>
  </div>
  <div class="plot"><div id="dens" style="height:460px;width:100%"></div></div>
  <p class="readout" style="margin-top:14px">largest empty ball<b id="gball">—</b>
     <span id="gnote"></span></p>
  <p class="note">The single markers are points: the cells of highest and lowest
     reachability inside the uncovered set. The <b>boxes</b> are the wider claim.
     <i>Candidate regions</i> are specified in advance from the mechanistic question;
     <i>empty boxes</i> are found without them, by growing a box from the point furthest
     from any published paradigm until it meets one. Every box drawn here is a shadow of
     a four-dimensional region, so a point can fall inside it on screen while a
     constraint on one of the two axes not shown puts it outside — which is why the
     occupancy counts come from the funnel in step 2 and not from this picture.
     Reachability is measured from the centre of that box to the nearest published
     paradigm: not whether anyone has been there, but how far it is from somewhere they
     have.</p>
</section>

<section>
  <p class="eyebrow">step 4 — where the corpus is not</p>
  <h2>The low-density regions, against the partition that found them</h2>
  <p class="lead">A gap argument has to put two things side by side: the territory each
     discovered cluster occupies, and the regions the corpus does not reach.
     The empty regions themselves are drawn in step 3b, on the density they were found
     from. The <b>single-point gaps</b> here are the cells of highest and lowest
     reachability inside the uncovered set — a much narrower claim than a box, and off by
     default for that reason.</p>
  <div class="controls">
    <div class="ctl"><label>plane</label><select id="gplane"></select></div>
    <div class="ctl"><label>show</label>
      <button class="seg" id="fgap" aria-pressed="false">single-point gaps</button></div>
    <div class="ctl"><label>boxes</label>
      <button class="seg" id="greg2" aria-pressed="false">candidate regions</button>
      <button class="seg" id="ghole2" aria-pressed="false">empty boxes</button></div>
    <div class="readout">clusters<b id="gcount">—</b>
      <span id="gcnote"></span></div>
  </div>
  <div class="plot"><div id="fboth" style="height:460px;width:100%"></div></div>
  <p class="note">A point inside another cluster's territory is a paradigm the geometry
     and the reader classify differently. The hulls are drawn on the two axes shown, so
     two clusters that separate on a third will overlap here: that is a fact about the
     plane, not about the partition.</p>
</section>

<section>
  <p class="eyebrow">step 5 — an empty region has to be empty in account space too</p>
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
  <p class="note" id="accnote"></p>
</section>

<section>
  <p class="eyebrow">step 5b — the same map, over the whole space</p>
  <h2>Which account owns which design</h2>
  <p class="lead">Evaluating f on a grid rather than at a point turns the account labels
     into a field over the space, and lets the emptiness of a region be checked in
     account space as well as in paradigm space. The kernel is conditioned on the two
     plotted coordinates only, so the plane below is a marginal of the field rather than
     a slice through fixed values of the other two. Where the estimate rests on fewer
     than <span class="coord" id="minneff"></span> effective rows the cell is drawn as
     unsupported rather than as a confident answer: that pale region is the honest part
     of the picture.</p>
  <div class="controls">
    <div class="ctl"><label>plane</label><select id="aplane"></select></div>
    <div class="ctl"><label>show</label><select id="amode">
      <option value="dominant">which account owns each design</option>
      <option value="entropy">how contested that ownership is</option>
      <option value="support">rows behind the estimate</option>
      <option value="single">one account at a time</option>
    </select></div>
    <div class="ctl"><label>account</label><select id="aacc"></select></div>
    <div class="ctl"><label>show</label>
      <button class="seg" id="apts" aria-pressed="true">paradigms</button>
      <button class="seg" id="agap" aria-pressed="true">gaps</button></div>
    <div class="readout">accounts on this plane<b id="alive">—</b>
      <span id="aentropy"></span></div>
  </div>
  <div class="plot"><div id="afield" style="height:470px;width:100%"></div></div>
  <div class="swatches" id="aswatch"></div>
  <p class="eyebrow" style="margin-top:26px">account mass inside each group</p>
  <div class="plot"><div id="acomp" style="height:300px;width:100%"></div></div>
  <p class="note">The bars are the h layer restricted to a partition of the corpus: the
     hand labels first, then, when the page was built with clustering on, the groups the
     geometry found. Two literatures that turn out to hold the same accounts in the same
     proportions are two names for one body of work; two that do not are the reason the
     gap between them is worth running.</p>
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
/* the gaps section has its own plane and its own overlays: it is a different question
   from "do the two partitions agree", and sharing state made one control move both */
state.gap = {plane:['x','t'], gaps:false};
/* the box overlays belong to the density plot, which is where the empty regions were
   found; the partitions plot shows the partition only */
state.boxes = {regions:false, holes:false};
state.acc = {plane:['x','y'], mode:'dominant', account:'SAL', pts:true, gaps:true};

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
      // sized up: at 4.5px against a white cube with grid lines the corpus reads as
      // dust rather than as points, and the ones inside the box need to stay
      // distinguishable from it without becoming the only thing visible
      marker:{ size:arr.map(p => (state.query && matches(p)) ? 14
                                : inBox(p, state.box) ? 11 : 7.5),
               opacity: 0.95,
               color: state.colorby === 'year' ? arr.map(p=>p.year) : colour,
               colorscale: state.colorby === 'year' ? 'Viridis' : undefined,
               showscale: state.colorby === 'year',
               symbol:arr.map(p=>SYM[p.conf]||'circle'),
               line:{width:arr.map(p => (state.query && matches(p)) ? 2.6 : 0.9),
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
function boxOnPlane(cons, a, b){
  /* the projection of a 4D constraint box onto two of its axes, with the constraints
     that fall off the plane returned so the caller can mark the rectangle a shadow */
  const lo = {}, hi = {}, off = [];
  PRIN.forEach(k => { lo[k] = 0; hi[k] = 1; });
  cons.forEach(c => {
    const k = c[0], op = c[1], v = c[2];
    if (op === '>=') lo[k] = Math.max(lo[k], v); else hi[k] = Math.min(hi[k], v);
    if (k !== a && k !== b) off.push(k);
  });
  return {x0:lo[a], x1:hi[a], y0:lo[b], y1:hi[b], off};
}
function boxShapes(a, b){
  const shapes = [], anns = [];
  if (state.boxes.regions){
    Object.entries(D.regions || {}).forEach(([name, spec]) => {
      const r = boxOnPlane(spec.constraints, a, b);
      shapes.push({type:'rect', x0:r.x0, x1:r.x1, y0:r.y0, y1:r.y1,
        line:{color:spec.color, width:1.6, dash:r.off.length ? 'dot' : 'solid'},
        fillcolor:rgba(spec.color, 0.06), layer:'above'});
      anns.push({x:r.x1, y:r.y1, text:name + (r.off.length ? ' shadow' : ''),
        showarrow:false, xanchor:'right', yanchor:'bottom',
        font:{size:10, color:spec.color},
        bgcolor:'rgba(255,255,255,0.65)'});
    });
  }
  if (state.boxes.holes){
    (D.holes || []).forEach((h, i) => {
      shapes.push({type:'rect', x0:h[a+'_lo'], x1:h[a+'_hi'],
        y0:h[b+'_lo'], y1:h[b+'_hi'],
        line:{color:'#1B2A33', width:1.2, dash:'dash'},
        fillcolor:'rgba(27,42,51,0.04)', layer:'above'});
      if (i === 0) anns.push({x:h[a+'_lo'], y:h[b+'_hi'], text:'largest empty box',
        showarrow:false, xanchor:'left', yanchor:'bottom',
        font:{size:10, color:'#1B2A33'}, bgcolor:'rgba(255,255,255,0.65)'});
    });
  }
  const h0 = (D.holes || [])[0];
  if (h0 && el('gball')){
    el('gball').textContent = h0.radius.toFixed(2);
    el('gnote').textContent = 'centre ' + PRIN.map(k => `${k} ${fmt(h0[k])}`).join(' · ')
      + (h0.in_region && h0.in_region !== '—' ? ` · inside ${h0.in_region}` : '');
  }
  return {shapes, anns};
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
  const bx = boxShapes(a, b);
  Plotly.react('dens', traces, {
    shapes:bx.shapes, annotations:bx.anns,
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
      y:keys, marker:{color:keys.map(a=>D.accountColor[a] || '#7A5EA8')},
      hovertemplate:'%{y}: %{x:.2f}<extra></extra>'}], {
    margin:{l:46,r:14,t:8,b:34}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'white',
    xaxis:{title:{text:'P(account | design at box centre)', font:{size:11}}, range:[0,1]}
  }, {displayModeBar:false, responsive:true});

  const mass = {}; D.accounts.forEach(a => mass[a] = 0); let mt = 0;
  pool().forEach(p => { if (p.acc && mass[p.acc] !== undefined) { mass[p.acc] += p.w; mt += p.w; } });
  const mk = D.accounts.filter(a => mt > 0 && mass[a]/mt > 0.001);
  Plotly.react('push', [{type:'bar', x:mk, y:mk.map(a=>mass[a]/mt),
      marker:{color:mk.map(a=>D.accountColor[a] || '#7A5EA8')},
      hovertemplate:'%{x}: %{y:.2f}<extra></extra>'}], {
    margin:{l:46,r:14,t:8,b:34}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'white',
    yaxis:{title:{text:'share of corpus mass', font:{size:11}}}
  }, {displayModeBar:false, responsive:true});

  el('accnote').textContent = tot > 0
    ? `the estimate at the box centre rests on an effective ${(tot*tot/P.reduce((s,p)=>{
        const d=(c.x-p.x)**2+(c.y-p.y)**2+(c.z-p.z)**2+(c.t-p.t)**2;
        const k=p.w*Math.exp(-d/s2); return s+k*k; },0)).toFixed(1)} rows`
    : 'no scored row carries an account near this box';
}

/* ---------- step 4b: the account field over the space ----------
   Same Nadaraya-Watson estimator as figure 6b, conditioned on the two plotted
   coordinates only, so what is drawn is a marginal of the field rather than a slice
   through fixed values of the other axes. Recomputed in the browser rather than
   shipped as an image, because it has to react to sigma and to the plane. */
const NACC = () => D.accounts.length;
function accRows(){
  return pool().filter(p => p.pacc && p.cluster !== 'thesis');
}
function accField(a, b, n, sigma){
  const P = accRows().filter(p => p[a] !== null && p[b] !== null);
  const lin = Array.from({length:n}, (_,i)=>i/(n-1));
  const K = NACC(), s2 = 2*sigma*sigma;
  const F = [], NE = [];
  for (let i=0;i<n;i++){
    F.push([]); NE.push(new Float64Array(n));
    for (let j=0;j<n;j++){
      const v = new Float64Array(K);
      let s = 0, ss = 0;
      for (const p of P){
        const k = p.w*Math.exp(-((lin[i]-p[a])**2 + (lin[j]-p[b])**2)/s2);
        if (k < 1e-9) continue;
        s += k; ss += k*k;
        for (let m=0;m<K;m++) v[m] += k*p.pacc[m];
      }
      if (s > 0) for (let m=0;m<K;m++) v[m] /= s;
      F[i].push(v);
      NE[i][j] = s > 0 ? s*s/ss : 0;
    }
  }
  return {lin, F, NE, n:P.length};
}
function accEntropy(v, live){
  let h = 0;
  for (let m=0;m<v.length;m++) if (v[m] > 0) h -= v[m]*Math.log(v[m]);
  return h/Math.log(Math.max(live, 2));
}
function accLive(F){
  const K = NACC(), mass = new Float64Array(K);
  let cells = 0;
  F.forEach(row => row.forEach(v => { cells++; for (let m=0;m<K;m++) mass[m] += v[m]; }));
  return D.accounts.filter((a,m) => cells && mass[m]/cells > 0.02);
}
/* one heatmap per account, each showing only the cells that account leads: colour
   names the account and intensity is its margin over the runner-up, which a single
   categorical heatmap cannot express. */
function drawAccountField(){
  const [a,b] = state.acc.plane, n = 45;
  const {lin, F, NE, n:nrows} = accField(a, b, n, state.sigma);
  const live = accLive(F);
  const thin = D.accountMinNeff;
  const K = NACC();
  const blank = () => Array.from({length:n}, ()=>new Array(n).fill(null));
  const traces = [];
  let meanH = 0, cells = 0;

  const hover = blank();
  for (let i=0;i<n;i++) for (let j=0;j<n;j++){
    const v = F[i][j];
    const top = D.accounts.map((c,m)=>[c,v[m]]).sort((p,q)=>q[1]-p[1]).slice(0,3)
                 .filter(p=>p[1]>0.02).map(p=>`${p[0]} ${p[1].toFixed(2)}`).join(' · ');
    hover[j][i] = `${a} ${lin[i].toFixed(2)} · ${b} ${lin[j].toFixed(2)}`
      + `<br>${top || 'no account nearby'}<br>effective n ${NE[i][j].toFixed(1)}`;
    if (NE[i][j] >= thin){ meanH += accEntropy(v, live.length); cells++; }
  }
  meanH = cells ? meanH/cells : 0;

  if (state.acc.mode === 'dominant'){
    live.forEach(accName => {
      const m = D.accounts.indexOf(accName);
      const z = blank();
      let any = false;
      for (let i=0;i<n;i++) for (let j=0;j<n;j++){
        if (NE[i][j] < thin) continue;
        const v = F[i][j];
        let best = -1, second = -1;
        for (let q=0;q<K;q++){ if (v[q] > best){ second = best; best = v[q]; }
                               else if (v[q] > second) second = v[q]; }
        if (v[m] === best && best > 0){ z[j][i] = 0.3 + 0.7*Math.min((best-second)/0.5, 1); any = true; }
      }
      if (!any) return;
      traces.push({type:'heatmap', x:lin, y:lin, z, zmin:0, zmax:1, showscale:false,
        colorscale:[[0,rgba(D.accountColor[accName],0.10)],[1,D.accountColor[accName]]],
        text:hover, hovertemplate:'%{text}<extra></extra>', hoverongaps:false, zsmooth:'best'});
    });
  } else if (state.acc.mode === 'single'){
    const m = D.accounts.indexOf(state.acc.account);
    const z = blank();
    for (let i=0;i<n;i++) for (let j=0;j<n;j++)
      if (NE[i][j] >= thin) z[j][i] = F[i][j][m];
    traces.push({type:'heatmap', x:lin, y:lin, z, zmin:0, zmax:1, zsmooth:'best',
      colorscale:[[0,'#FFFFFF'],[1,D.accountColor[state.acc.account]]],
      colorbar:{title:{text:`P(${state.acc.account} | design)`, font:{size:11}},
                thickness:11, len:0.85},
      text:hover, hovertemplate:'%{text}<extra></extra>', hoverongaps:false});
  } else if (state.acc.mode === 'entropy'){
    const z = blank();
    for (let i=0;i<n;i++) for (let j=0;j<n;j++)
      if (NE[i][j] >= thin) z[j][i] = accEntropy(F[i][j], live.length);
    traces.push({type:'heatmap', x:lin, y:lin, z, zmin:0, zmax:1, zsmooth:'best',
      colorscale:'PuBuGn', reversescale:true,
      colorbar:{title:{text:'normalised entropy of f', font:{size:11}},
                thickness:11, len:0.85},
      text:hover, hovertemplate:'%{text}<extra></extra>', hoverongaps:false});
  } else {
    const z = blank();
    let mx = 0;
    for (let i=0;i<n;i++) for (let j=0;j<n;j++){ z[j][i] = NE[i][j]; if (NE[i][j] > mx) mx = NE[i][j]; }
    traces.push({type:'heatmap', x:lin, y:lin, z, zmin:0, zmax:mx || 1, zsmooth:'best',
      colorscale:'Greys',
      colorbar:{title:{text:'effective rows behind f', font:{size:11}},
                thickness:11, len:0.85},
      text:hover, hovertemplate:'%{text}<extra></extra>'});
    traces.push({type:'contour', x:lin, y:lin, z, showscale:false, hoverinfo:'skip',
      contours:{start:thin, end:thin, size:1, coloring:'none', showlabels:false},
      line:{color:'#C1425A', width:1.4}});
  }

  if (state.acc.pts){
    const P = accRows().filter(p => p[a] !== null && p[b] !== null);
    const s = spread(P, a, b, a, 0.012);
    traces.push({type:'scatter', mode:'markers', showlegend:false,
      x:s.map(v=>v.x), y:s.map(v=>v.y),
      text:P.map(p=>`<b>${p.key}</b><br>${p.title}<br>dominant account ${p.acc||'—'}`),
      hovertemplate:'%{text}<extra></extra>',
      marker:{size:6, color:P.map(p=>D.accountColor[p.acc] || '#8A9AA2'),
              line:{width:0.9, color:'white'}}});
  }
  if (state.acc.gaps){
    ['frontier','island'].forEach(kind => {
      const g = D.gaps.filter(v=>v.kind===kind);
      if (!g.length) return;
      traces.push({type:'scatter', mode:'markers', name:kind+' gap',
        x:g.map(v=>v[a]), y:g.map(v=>v[b]),
        text:g.map(v=>`${kind} gap<br>`+PRIN.map(k=>`${k} ${fmt(v[k])}`).join(' · ')),
        hovertemplate:'%{text}<extra></extra>',
        marker:{symbol: kind==='frontier'?'circle-open':'star', size: kind==='frontier'?13:15,
                color:'#3B2E80', line:{width:2, color:'#3B2E80'}}});
    });
  }
  Plotly.react('afield', traces, {
    margin:{l:52,r:10,t:10,b:46}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#E7EBED',
    xaxis:{title:{text:D.axes[a].label, font:{size:12}}, range:[0,1], constrain:'domain'},
    yaxis:{title:{text:D.axes[b].label, font:{size:12}}, range:[0,1],
           scaleanchor:'x', scaleratio:1},
    legend:{orientation:'h', y:1.06, x:0, font:{size:11}}
  }, {displayModeBar:false, responsive:true});

  el('alive').textContent = live.length;
  el('aentropy').textContent = `mean entropy ${meanH.toFixed(2)} over the supported plane`
    + ` · ${nrows} rows carry an account`;
  el('aswatch').innerHTML = live.map(c =>
      `<span class="s"><span class="dot" style="background:${D.accountColor[c]}"></span>`
      + `${c} — ${D.accountLabel[c]}</span>`).join('')
    + `<span class="s"><span class="dot" style="background:#E7EBED"></span>`
    + `effective n &lt; ${thin}</span>`;
}
/* the h layer restricted to a partition: hand labels, and the discovered clusters
   when the page was built with -k */
function drawAccountComposition(){
  const K = NACC();
  const groups = [];
  Object.keys(D.clusters).forEach(g => {
    const rows = D.pts.filter(p => p.pacc && p.cluster === g);
    if (rows.length) groups.push({name:(D.clusters[g].label||g)+`<br>n = ${rows.length}`, rows});
  });
  if (FIT) FIT.foundNames.forEach((nm,i) => {
    const rows = D.pts.filter(p => p.pacc && p.found === i);
    if (rows.length) groups.push({name:nm+`<br>n = ${rows.length}`, rows});
  });
  const share = g => {
    const v = new Float64Array(K); let tot = 0;
    g.rows.forEach(p => { for (let m=0;m<K;m++) v[m] += p.w*p.pacc[m]; tot += p.w; });
    return tot ? Array.from(v, x => x/tot) : Array.from(v);
  };
  const S = groups.map(share);
  const traces = D.accounts.map((c,m) => ({
    type:'bar', name:c, x:groups.map(g=>g.name), y:S.map(v=>v[m]),
    marker:{color:D.accountColor[c]},
    hovertemplate:`${c} — ${D.accountLabel[c]}: %{y:.2f}<extra></extra>`
  })).filter((t,m) => S.some(v => v[m] > 0.01));
  Plotly.react('acomp', traces, {
    barmode:'stack', margin:{l:52,r:10,t:10,b:52}, paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'white', bargap:0.42,
    yaxis:{title:{text:'share of account mass', font:{size:11}}, range:[0,1]},
    legend:{orientation:'h', y:1.08, x:0, font:{size:10}}
  }, {displayModeBar:false, responsive:true});
}
function buildAccountControls(){
  const planes = [];
  for (let i=0;i<PRIN.length;i++) for (let j=i+1;j<PRIN.length;j++) planes.push([PRIN[i],PRIN[j]]);
  el('aplane').innerHTML = planes.map(([a,b]) =>
    `<option value="${a},${b}">${a} vs ${b}</option>`).join('');
  el('aplane').value = state.acc.plane.join(',');
  el('aplane').onchange = () => { state.acc.plane = el('aplane').value.split(','); drawAccountField(); };
  el('aacc').innerHTML = D.accounts.map(c =>
    `<option value="${c}">${c} — ${D.accountLabel[c]}</option>`).join('');
  el('aacc').value = state.acc.account;
  el('aacc').onchange = () => { state.acc.account = el('aacc').value;
    state.acc.mode = 'single'; el('amode').value = 'single'; drawAccountField(); };
  el('amode').onchange = () => { state.acc.mode = el('amode').value; drawAccountField(); };
  el('apts').onclick = () => { state.acc.pts = !state.acc.pts;
    el('apts').setAttribute('aria-pressed', state.acc.pts); drawAccountField(); };
  el('agap').onclick = () => { state.acc.gaps = !state.acc.gaps;
    el('agap').setAttribute('aria-pressed', state.acc.gaps); drawAccountField(); };
  el('minneff').textContent = D.accountMinNeff;
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
/* Both partitions on one pair of axes: the discovered clusters as filled territory
   (a convex hull of their members on this plane) and the hand labels as the point
   colour, so a disagreement is a point sitting in the wrong territory rather than
   something the reader has to find by comparing two pictures. */
function hull(pts){
  if (pts.length < 3) return pts;
  const P = pts.slice().sort((a,b) => a[0]-b[0] || a[1]-b[1]);
  const cross = (o,a,b) => (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]);
  const lower = [];
  for (const p of P){
    while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = P.length-1; i >= 0; i--){
    const p = P[i];
    while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop(); upper.pop();
  return lower.concat(upper);
}
function drawBoth(){
  if (!FIT) return;
  const [a,b] = state.gap.plane;
  const P = fitPts().filter(p => p[a] !== null && p[b] !== null);
  const traces = [];
  const founds = [...new Set(P.map(p => p.found))].sort((x,y) => x-y);
  founds.forEach(f => {
    const pts = P.filter(p => p.found === f).map(p => [p[a], p[b]]);
    const h = hull(pts);
    if (h.length >= 3){
      traces.push({type:'scatter', mode:'lines', fill:'toself', hoverinfo:'skip',
        x:h.concat([h[0]]).map(v=>v[0]), y:h.concat([h[0]]).map(v=>v[1]),
        fillcolor:rgba(cpal(f), 0.13), line:{color:cpal(f), width:1.2},
        name:`cluster c${f} (${pts.length})`, legendgroup:'c'+f});
    }
    const cx = pts.reduce((s,v)=>s+v[0],0)/pts.length;
    const cy = pts.reduce((s,v)=>s+v[1],0)/pts.length;
    traces.push({type:'scatter', mode:'markers+text', x:[cx], y:[cy], text:[`c${f}`],
      textposition:'top right', textfont:{size:11, color:cpal(f)}, hoverinfo:'skip',
      showlegend:false, legendgroup:'c'+f,
      marker:{symbol:'x', size:13, color:cpal(f), line:{width:1, color:'#1B2A33'}}});
  });
  const s = spread(P, a, b, a, 0.012);
  Object.entries(D.clusters).forEach(([k, v]) => {
    const idx = P.map((p,i) => p.cluster === k ? i : -1).filter(i => i >= 0);
    if (!idx.length) return;
    traces.push({type:'scatter', mode:'markers', name:v.label,
      x:idx.map(i=>s[i].x), y:idx.map(i=>s[i].y),
      text:idx.map(i=>`<b>${P[i].key}</b><br>${P[i].title}`
        + `<br>hand label ${P[i].cluster} · found c${P[i].found}`),
      hovertemplate:'%{text}<extra></extra>',
      marker:{size:7, color:v.color, line:{width:0.8, color:'white'}}});
  });
  if (state.gap.gaps){
    ['frontier','island'].forEach(kind => {
      const g = D.gaps.filter(v => v.kind === kind);
      if (!g.length) return;
      traces.push({type:'scatter', mode:'markers', name:kind+' gap',
        x:g.map(v=>v[a]), y:g.map(v=>v[b]),
        text:g.map(v=>`${kind} gap<br>`+PRIN.map(k=>`${k} ${fmt(v[k])}`).join(' · ')),
        hovertemplate:'%{text}<extra></extra>',
        marker:{symbol: kind==='frontier'?'circle-open':'star',
                size: kind==='frontier'?14:16, color:'#3B2E80',
                line:{width:2, color:'#3B2E80'}}});
    });
  }
  const bx = boxShapes(a, b);
  el('gcount').textContent = founds.length;
  el('gcnote').textContent = founds.map(f =>
    `c${f} ${P.filter(p => p.found === f).length}`).join(' · ');
  Plotly.react('fboth', traces, {
    shapes:bx.shapes, annotations:bx.anns,
    margin:{l:54,r:12,t:10,b:48}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'white',
    xaxis:{title:{text:D.axes[a].label, font:{size:12}}, range:[-0.05,1.05], constrain:'domain'},
    yaxis:{title:{text:D.axes[b].label, font:{size:12}}, range:[-0.05,1.05],
           scaleanchor:'x', scaleratio:1},
    legend:{orientation:'h', y:1.08, x:0, font:{size:10}}
  }, {displayModeBar:false, responsive:true});
}
function refreshFit(){
  if (!FIT) return;
  fitScatter('ffoundplot', p => cpal(p.found));
  fitScatter('fgivenplot', p => COL[p.cluster] || '#8A9AA2');
  drawBoth();
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
  const gplanes = [];
  for (let i=0;i<PRIN.length;i++) for (let j=i+1;j<PRIN.length;j++)
    gplanes.push([PRIN[i], PRIN[j]]);
  el('gplane').innerHTML = gplanes.map(([a,b]) =>
    `<option value="${a},${b}">${a} vs ${b}</option>`).join('');
  el('gplane').value = state.gap.plane.join(',');
  el('gplane').onchange = () => {
    state.gap.plane = el('gplane').value.split(','); drawBoth(); };
  el('fgap').onclick = () => {
    state.gap.gaps = !state.gap.gaps;
    el('fgap').setAttribute('aria-pressed', state.gap.gaps);
    drawBoth();
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
  el('bs').addEventListener('change', () => { drawDensity(); drawAccounts(); drawAccountField(); });
  el('vs').textContent = state.sigma.toFixed(3);

  const toggle = (id, key, after) => el(id).onclick = () => {
    state[key] = !state[key];
    el(id).setAttribute('aria-pressed', state[key]);
    after();
  };
  toggle('tlo','dropLo', () => { refreshBox(); drawDensity(); });
  toggle('tth','hideThesis', () => { refreshBox(); drawDensity(); });
  /* the box overlays are one piece of state with two sets of buttons, so turning
     them on in either section turns them on in both and the two plots cannot end up
     showing different regions */
  [['greg','ghole'], ['greg2','ghole2']].forEach(ids => {
    ids.forEach((id, n) => {
      const key = n === 0 ? 'regions' : 'holes';
      el(id).onclick = () => {
        state.boxes[key] = !state.boxes[key];
        [['greg','greg2'],['ghole','ghole2']][n].forEach(other =>
          el(other) && el(other).setAttribute('aria-pressed', state.boxes[key]));
        drawDensity();
        drawBoth();
      };
    });
  });
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
}
buildControls();
buildFitControls();
buildAccountControls();
refreshBox();
drawDensity();
drawAccountField();
drawAccountComposition();
refreshFit();
</script>
</body></html>
"""


def write_html(path, df, ladders, cfg, gaps, raw, source, inline=None,
               clus=None, holes=None):
    js, offline = plotly_bundle()
    if inline is False:
        js, offline = ('<script src="https://cdn.plot.ly/plotly-3.0.0.min.js" '
                       'charset="utf-8"></script>', False)
    payload = html_payload(df, ladders, cfg, gaps, raw, clus, holes)
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
            .replace("__CLUSMETA__",
                     f"clustering: {clus['method']}, k = {clus['k']}, on "
                     + ", ".join(clus["axes"]) if clus else
                     "clustering: not run for this page (pass -k)")
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


def report(df, raw, cfg, gaps, counts, mean_disp, f_thesis, n_eff, corr, path,
           clus=None, acc=None, diag=None, held=None, empty=None):
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
    add("gaps, method 1: low smoothed density (kernel, sigma = "
        f"{cfg.sigma:g}), searched over the feasible 4D grid")
    add("  a cell qualifies when its coverage deficit is at least "
        f"{EMPTY_DEFICIT:g} and no paradigm lies inside it;")
    add("  frontier and island gaps are the qualifying cells of highest and lowest")
    add("  reachability, taken separately because the arg max of deficit alone is")
    add("  always the far corner of the cube")
    for _, g in gaps.iterrows():
        add(f"  {g['kind']:<9} ({', '.join(f'{g[a]:.2f}' for a in cfg.axes)})"
            f"  reach {g['reach']:.3f}")
    if empty is not None:
        tab = empty["table"]
        add("")
        add("gaps, method 2: largest empty regions of the joint space (no kernel)")
        add("  the corpus is treated as a point set in the searchable part of the unit")
        add("  cube; the largest empty ball is the point furthest from any published")
        add("  paradigm, and its maximal box is grown from that centre until it meets")
        add("  a paradigm or the edge of the searchable set")
        if USE_FEASIBLE:
            add(f"  structural constraints ON: {empty['feasible_fraction']:.2f} of the "
                "cube is searchable;")
            add("  the rest is treated as impossible rather than unexplored and is "
                "never")
            add("  reported as a gap — but see the constraint audit below before "
                "relying on that")
        else:
            add("  structural constraints OFF (the default): the whole cube is "
                "searched, so")
            add("  a hole reported here may be a design that cannot be run")
        add(f"  largest empty ball radius {empty['radius']:.3f} against "
            f"{empty['null_mean']:.3f} for uniform corpora of the same size "
            f"(p = {empty['p']:.4f})")
        um = empty.get("masked_compare")
        if um is not None and len(um["table"]):
            u0 = um["table"].iloc[0]
            other = "on" if not USE_FEASIBLE else "off"
            add(f"  with the structural constraints {other} the largest hole instead has")
            add(f"    radius {u0['radius']:.3f} at ("
                + ", ".join(f"{u0[a]:.2f}" for a in cfg.axes) + ")")
        lowd = empty.get("lowd")
        if lowd is not None and len(lowd["table"]):
            add("")
            add("gaps, method 3: connected regions of the joint low-density set")
            add("  the joint density is evaluated over the full 4D grid, cut at the "
                f"{lowd['quantile']:.1%} quantile,")
            add("  and the surviving cells are grouped into connected components; the "
                "largest box")
            add("  fitting inside each component is reported so the region still reads "
                "as a design")
            add(f"  {lowd['n_components']} components at this level "
                f"(the level itself is swept in fig14)")
            for n, (_, rr) in enumerate(lowd["table"].iterrows()):
                add(f"  region {n + 1}: {rr['volume'] * 100:.1f}% of the searchable "
                    f"volume, {int(rr['cells'])} cells, "
                    f"{int(rr['occupancy'])} paradigms in its box")
                add("      box: " + ", ".join(
                    f"{a} [{rr[f'{a}_lo']:.2f}, {rr[f'{a}_hi']:.2f}]"
                    for a in cfg.axes))
        for i, r in tab.iterrows():
            add(f"  hole {i + 1}: centre ("
                + ", ".join(f"{r[a]:.2f}" for a in cfg.axes)
                + f")  radius {r['radius']:.2f}  box volume {r['volume']:.3f}"
                + f"  region {r['in_region']}")
            add("      box: " + ", ".join(
                f"{a} [{r[f'{a}_lo']:.2f}, {r[f'{a}_hi']:.2f}]" for a in cfg.axes)
                + f"   nearest {r['nearest']}")
    if acc is not None:
        add("")
        add("account space")
        push = pushforward(df[~df["thesis"]])
        add("  pushforward f#mu: " + ", ".join(
            f"{k} {v:.2f}" for k, v in sorted(push.items(), key=lambda kv: -kv[1])
            if v > 0.005))
        pr = acc["pred"]
        if np.isfinite(pr["logloss"]):
            add(f"  does the design predict the account? leave-one-out over "
                f"{pr['n']} rows carrying one:")
            add(f"    log loss {pr['logloss']:.3f} against {pr['logloss_null']:.3f} "
                f"for permuted labels (p = {pr['p']:.4f})")
            add(f"    top-1 accuracy {pr['accuracy']:.1%} against "
                f"{pr['accuracy_null']:.1%}")
        add(f"  account field drawn on the "
            f"({', '.join(acc['field']['plane'])}) plane; accounts carrying "
            f"more than 2% of it: {', '.join(acc['field']['live'])}")
        add(f"  mean normalised entropy of f over the supported plane "
            f"{acc['field']['entropy']:.2f} "
            f"(0 = one account owns every design, 1 = all of them equally)")
        add("  f at the points the prose quotes (n_eff is the effective number of "
            "rows behind each):")
        tab = acc["probes"]
        cols = [a for a in ACCOUNTS if tab[a].max() > 0.02]
        add("    " + "probe".ljust(30) + "n_eff".rjust(7) + "H".rjust(6)
            + "".join(c.rjust(7) for c in cols))
        for _, r in tab.iterrows():
            add("    " + r["probe"].replace("\n", " ")[:30].ljust(30)
                + f"{r['n_eff']:7.1f}" + f"{r['entropy']:6.2f}"
                + "".join(f"{r[c]:7.2f}" for c in cols))
    if held is not None and len(held):
        sub_h = held.dropna(subset=cfg.axes)
        add("")
        add(f"held-out thesis paradigms ({len(sub_h)} scored on all principal axes)")
        add("  these took no part in the density, the region counts, the gap search,")
        add("  the clustering or the account field: this is where the map puts them")
        P, w, corpus = matrix(df, cfg.axes)
        for _, r in sub_h.iterrows():
            u = r[cfg.axes].to_numpy(float)
            d = np.exp(-((P - u) ** 2).sum(1) / (2 * cfg.sigma ** 2))
            reach = float((w * d).sum() / w.sum())
            j = int(((P - u) ** 2).sum(1).argmin())
            inside = [n for n, spec in REGIONS.items()
                      if all((u[cfg.axes.index(k)] >= v) if op == ">="
                             else (u[cfg.axes.index(k)] <= v)
                             for k, op, v in spec["constraints"])]
            add(f"  {str(r['citekey'] or r['paradigm_id'])[:26]:<26} "
                + "(" + ", ".join(f"{u[i]:.2f}" for i in range(len(cfg.axes))) + ")"
                + f"  in {'+'.join(inside) if inside else 'no region'}"
                + f", reach {reach:.3f}"
                + f", nearest published {str(corpus.iloc[j]['citekey'])[:22]} at "
                  f"{np.linalg.norm(P[j] - u):.2f}")
        for n in REGIONS:
            k = int(satisfies(sub_h, REGIONS[n]["constraints"]).sum())
            add(f"  {n}: {counts[n]} published, {k} thesis")
    ctab, _ = constraint_audit(df, cfg.axes)
    if len(ctab):
        add("")
        add("constraint audit: does the corpus obey the structural constraints?")
        add("  a published paradigm inside an excluded region is a counterexample to "
            "the")
        add("  claim that the region cannot be occupied")
        n_scored = len(df.dropna(subset=cfg.axes))
        for _, c in ctab.iterrows():
            verdict = "holds" if c["violations"] == 0 else "FAILS"
            add(f"  {c['constraint']:<26} {verdict:<6} {c['violations']} of "
                f"{n_scored} violate it   ({c['rationale']})")
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
        add("  centroids in rung units (the partition was found on standardised axes)")
        for j, c in enumerate(clus["centers"]):
            n = int((clus["labels"] == j).sum())
            add(f"  c{j} (n = {n}): " +
                ", ".join(f"{a} {v + 0.0:.2f}" if abs(v) > 5e-3 else f"{a} 0.00"
                          for a, v in zip(clus["axes"], c)))
        M = clus["contingency"]
        add("  " + "found".ljust(8) + "".join(g[:9].rjust(10) for g in clus["given_names"]))
        for i, row_ in enumerate(M):
            add("  " + f"c{i}".ljust(8) + "".join(str(v).rjust(10) for v in row_))
        add("  k sweep:")
        add("    " + clus["curve"].round(3).to_string(index=False).replace("\n", "\n    "))
    if diag is not None:
        pa, pdz = diag["permanova"], diag["permdisp"]
        add("")
        add("cluster diagnostics (all nulls are permutation or bootstrap)")
        add(f"  PERMANOVA  pseudo-F {pa['F']:.2f}, R2 {pa['R2']:.2f}, "
            f"p = {pa['p']:.4f} ({pa['n']} rows, {pa['a']} groups)")
        add(f"  PERMDISP   pseudo-F {pdz['F']:.2f}, p = {pdz['p']:.4f}  "
            "(a small p here means the groups also differ in spread, so the "
            "location claim needs care)")
        add("    within-group dispersion: " + ", ".join(
            f"c{g} {v:.2f}" for g, v in pdz["spread"].items()))
        add("  variance explained per axis (eta squared, permutation p):")
        for _, r in diag["axis"].iterrows():
            add(f"    {r['axis']:<4} eta2 {r['eta2']:.3f}  p = {r['p']:.4f}   "
                f"(chance {r['eta2_null']:.3f})")
        add("  pairwise Hedges g per axis:")
        add("    " + diag["pairwise"].round(2).to_string(index=False)
            .replace("\n", "\n    "))
        add("  cluster-wise bootstrap stability (Hennig; <0.6 dissolved, "
            ">0.75 stable):")
        add("    " + ", ".join(f"c{g} {v:.2f}" for g, v in
                               sorted(diag["jaccard"].items())))
        add("  prediction strength (Tibshirani and Walther; cutoff 0.8):")
        add("    " + diag["strength"].round(3).to_string(index=False)
            .replace("\n", "\n    "))
        sep = diag["separation"]
        if sep is not None:
            add("  valley: the corpus split in two by itself and projected onto its "
                "own best direction,")
            add("  against unimodal references put through exactly the same procedure")
            add(f"    widest interior gap {sep['gap']:.3f} sd against "
                f"{sep['gap_null_mean']:.3f} for the references, p = {sep['p_gap']:.4f}")
            add(f"    two-component BIC margin {sep['delta_bic']:.1f} against "
                f"{sep['bic_null_mean']:.1f} for the references, p = {sep['p_bic']:.4f}")
            add(f"    ({sep['n_null']} references drawn from a Gaussian with the "
                "corpus covariance)")
            if "composition" in sep:
                for j in (0, 1):
                    c = sep["composition"][f"h{j}"]
                    add(f"    half {j}: " + ", ".join(f"{g} {n}" for g, n in c.items()
                                                      if n))
            add(f"    {sep['tied_fraction']:.0%} of the projected values are tied; the "
                "references are continuous and have none, so both p-values are read "
                "as upper bounds on the evidence")
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
                  n_eff, corr, ladders, source, clus=None, accounts=None,
                  diag=None):
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

    # the account field: whether f is informative at all, and what it returns where
    # the argument needs it. Written to say nothing when the layer was not computed.
    field_txt = ""
    if accounts is not None:
        pr, tab = accounts["pred"], accounts["probes"]
        fld = accounts["field"]

        def probe_line(mask):
            r = tab[mask]
            if not len(r):
                return None
            r = r.iloc[0]
            top = sorted(((c, r[c]) for c in ACCOUNTS if r[c] > 0.02),
                         key=lambda kv: -kv[1])[:3]
            return (r["probe"].replace("\n", " "),
                    ", ".join(f"{c} {v:.2f}" for c, v in top),
                    float(r["n_eff"]), float(r["entropy"]))

        pieces = []
        if np.isfinite(pr["logloss"]):
            verdict_f = ("carries real information about the account"
                         if pr["p"] < 0.05 else
                         "cannot be distinguished from a field that returns the same "
                         "distribution everywhere")
            pieces.append(
                f"Before reading anything off the field it is worth asking whether the "
                f"design predicts the account at all. Leave-one-out over the {pr['n']} "
                f"rows that carry one gives a log loss of {pr['logloss']:.2f} against "
                f"{pr['logloss_null']:.2f} when the accounts are permuted across designs "
                f"(p = {pr['p']:.3f}), and recovers the dominant account of "
                f"{pr['accuracy']:.0%} of rows against {pr['accuracy_null']:.0%} by "
                f"chance, so the map {verdict_f}.")
        pieces.append(
            f"Drawn over the ({', '.join(fld['plane'])}) plane, the field is owned rather "
            f"than shared: mean normalised entropy {fld['entropy']:.2f} across the "
            f"supported part of the plane, with {', '.join(fld['live'])} the accounts "
            f"holding more than two per cent of it (`fig6b_account_field`). The pale "
            f"regions of panel C are the honest ones — there the estimate rests on fewer "
            f"than {ACCOUNT_MIN_NEFF:g} effective rows and should not be quoted.")
        gap_rows = [probe_line(tab["probe"].str.startswith(k)) for k in
                    ("frontier gap", "island gap")]
        gap_rows = [g for g in gap_rows if g]
        if gap_rows:
            pieces.append(
                "Read at the points the argument turns on (`fig6c_account_probes`): "
                + "; ".join(f"{name} returns {top} on an effective {ne:.1f} rows "
                            f"(H = {h:.2f})" for name, top, ne, h in gap_rows)
                + ". A gap whose account distribution is thin and contested is a gap in "
                  "theory as well as in paradigms; one that inherits a confident "
                  "distribution from its neighbours is a design nobody has run because "
                  "everyone already knows what it would show.")
        field_txt = "\n\n".join(pieces)

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

    # the diagnostics: written to make the weaker readings available rather than
    # only the strong one, because every one of these tests can pass on noise
    diag_txt = ""
    if diag is not None:
        pa, pdz = diag["permanova"], diag["permdisp"]
        tab = diag["axis"]
        carry = tab[tab["p"] < 0.05].sort_values("eta2", ascending=False)
        flat = tab[tab["p"] >= 0.05]
        jac = diag["jaccard"]
        weak = [g for g, v in jac.items() if v < 0.6]
        firm = [g for g, v in jac.items() if v >= 0.75]
        st = diag["strength"]
        ok_k = st[st["prediction_strength"] >= 0.8]["k"]
        pieces = [
            f"That the groups differ at all survives a test that does not assume "
            f"normality, which the parametric alternative would need and these bounded "
            f"lattice-valued coordinates would not satisfy: permutational MANOVA on the "
            f"Euclidean distances gives pseudo-F = {pa['F']:.1f} "
            + ("(p < 0.001)" if pa["p"] < 0.001 else f"(p = {pa['p']:.3f})")
            + f" with $R^2$ = {pa['R2']:.2f}. "
            + (f"Dispersion is also heterogeneous (PERMDISP pseudo-F = {pdz['F']:.1f}, "
               + ("p < 0.001" if pdz["p"] < 0.001 else f"p = {pdz['p']:.3f}")
               + "), so part of what separates the groups is how tightly each is packed "
                 "rather than where its centre lies, and the location claim should be "
                 "made no more strongly than the axis-by-axis result below supports."
               if pdz["p"] < 0.05 else
               f"Within-group dispersion is homogeneous (PERMDISP pseudo-F = "
               f"{pdz['F']:.1f}, p = {pdz['p']:.3f}), so the separation is a difference "
               f"in location rather than in spread."),
            "The separation is not carried equally by the axes. "
            + (", ".join(f"${a}$ ($\\eta^2$ = {e:.2f})" for a, e in
                         zip(carry["axis"], carry["eta2"]))
               + " reach significance against permuted labels"
               if len(carry) else "No axis reaches significance against permuted labels")
            + (", while " + ", ".join(f"${a}$ ($\\eta^2$ = {e:.2f}, p = {p:.2f})"
                                      for a, e, p in
                                      zip(flat["axis"], flat["eta2"], flat["p"]))
               + " does not, which is the more interesting result: the corpus is "
                 "organised by that axis no more than chance would organise it."
               if len(flat) else "."),
            "Stability is reported per cluster rather than as one number, because a "
            "small cluster can dissolve entirely without moving the mean: "
            + ", ".join(f"c{g} at {v:.2f}" for g, v in sorted(jac.items()))
            + " on Hennig's bootstrap Jaccard"
            + (f", so c{', c'.join(str(g) for g in firm)} "
               f"{'is' if len(firm) == 1 else 'are'} stable on the usual reading"
               if firm else "")
            + (f" and c{', c'.join(str(g) for g in weak)} "
               f"{'is' if len(weak) == 1 else 'are'} not."
               if weak else ".")
            + (f" Prediction strength on held-out halves supports k \u2264 {int(ok_k.max())} "
               f"at the conventional cutoff of 0.8"
               if len(ok_k) else " Prediction strength reaches the conventional cutoff "
                                 "of 0.8 at no k in the sweep")
            + (f", and the run was made at k = {clus['k']}." if clus else "."),
        ]
        sep = diag["separation"]
        if sep is not None:
            both = sep["p_gap"] < 0.05 and sep["p_bic"] < 0.05
            either = sep["p_gap"] < 0.05 or sep["p_bic"] < 0.05
            valley = ("the corpus is thinner between its two halves than a shapeless "
                      "cloud of the same shape and size is between its own"
                      if both else
                      "one of the two statistics separates the corpus from the "
                      "references and the other does not, which is weak evidence for "
                      "a valley and should be reported as such" if either else
                      "the corpus is no thinner between its two halves than an "
                      "unclustered cloud of the same shape and size, so the valley "
                      "does not survive the control")
            comp = ""
            if "composition" in sep:
                parts = []
                for j in (0, 1):
                    c = {g: n for g, n in sep["composition"][f"h{j}"].items() if n}
                    top = sorted(c.items(), key=lambda kv: -kv[1])[:2]
                    parts.append(f"half {j} is {', '.join(f'{n} {g}' for g, n in top)}")
                comp = (" The two halves are not the two hand labels either: "
                        + "; ".join(parts) + ".")
            pieces.append(
                f"None of that yet tests the claim the chapter actually makes, which is "
                f"not that a partition exists but that the corpus is *sparse between* "
                f"the two literatures. The obvious test of that is circular, and the "
                f"circularity is worth stating because the result looks convincing: pick "
                f"the direction that best separates two clusters, project onto it, and "
                f"the projection is bimodal by construction — a single Gaussian cloud put "
                f"through those steps yields a clean valley and a decisive BIC. So the "
                f"null here is a null over corpora rather than over projections. A "
                f"unimodal reference with the covariance of the real corpus is drawn, "
                f"split in two, given its own best direction and measured with its own "
                f"valley, {sep['n_null']} times. Against that, the corpus's widest "
                f"interior gap is {sep['gap']:.2f} sd against a reference mean of "
                f"{sep['gap_null_mean']:.2f} (p = {sep['p_gap']:.3f}), and its "
                f"two-component BIC margin is {sep['delta_bic']:.0f} against "
                f"{sep['bic_null_mean']:.0f} (p = {sep['p_bic']:.3f}): "
                + valley + "." + comp
                + f" One caveat belongs with both numbers: the axes are ladders, so "
                  f"{sep['tied_fraction']:.0%} of the projected values are tied, while "
                  f"the references are continuous and have none. Ties manufacture empty "
                  f"intervals, so each p-value is an upper bound on the evidence rather "
                  f"than a lower one.")
        diag_txt = "\n\n".join(pieces)

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

{field_txt}

## Do the two literatures exist, or were they assumed?

The cluster label on each row was assigned by the reviewer. Whether the corpus separates
that way is a different question, and the geometry can answer it without being told the
answer. {clus_txt}

{diag_txt}

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
                         minneff=f"{ACCOUNT_MIN_NEFF:g}",
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
    "fig6b_account_field": (
        "The account field over the design space",
        "The same kernel regression as figure 6, evaluated on a grid rather than at a "
        "point, with the kernel conditioned on the two plotted coordinates alone so that "
        "each panel is a marginal of the field rather than a slice through fixed values "
        "of the remaining axes. A colour is the leading account and opacity its margin "
        "over the runner-up; B the normalised entropy of $f$, low where one account owns "
        "the design and high where several are live; C the effective number of rows "
        "behind each cell, with the red contour at {minneff}. Pale cells in A and B fall "
        "below that contour and are drawn as unsupported rather than as confident "
        "answers. The lower panels are the individual accounts on the same plane, with "
        "the rows carrying each as their dominant label. The dashed rectangle is the "
        "shadow of $G_1$."),
    "fig6c_account_probes": (
        "The account field where the argument needs it",
        "A $f$ read off at the centroid of each literature, at the candidate regions, at "
        "the gaps of figure 5 and at the thesis paradigm, with the effective number of "
        "rows and the normalised entropy above each bar. This is the account-space "
        "version of the occupancy funnel: a gap whose distribution is thin and contested "
        "is a gap in theory as well as in paradigms, whereas one that inherits a "
        "confident distribution from its neighbours is a design nobody has run because "
        "the answer is taken to be known. B and C restrict the $h$ layer to the hand "
        "labels and to the partition the geometry found."),
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
    "fig10_empty_space": (
        "Empty space, measured geometrically",
        "A the searchable set. When the structural constraints are on, the shaded "
        "wedges are excluded before any search runs; they are off by default because "
        "five published paradigms fall inside them (figure 11). B distance from each point of the plane "
        "to the nearest published paradigm, on the slice through the centre of the "
        "largest empty ball, with the recovered balls drawn as circles; because the "
        "slice passes through the centre these are the true cross-sections. C the "
        "radius of the largest empty ball against uniform corpora of the same size on "
        "the same feasible set. D the same holes as maximal empty boxes: the range "
        "each coordinate may take while the design still meets nothing published."),
    "fig12_empty_boxes": (
        "The regions the geometry proposes, drawn like the regions specified by hand",
        "The two largest empty boxes of the corpus, put through the same figure as the "
        "candidate regions of figure 2: two conditional planes and the constraint "
        "funnel, with each panel showing only the paradigms that satisfy the "
        "constraints on the axes not drawn. The funnel is the check that matters — a "
        "box found by growing from the point furthest from any published paradigm "
        "until it meets one should reach zero occupancy on its own count, and does. "
        "Unlike $G_1$ and $G_2$ these regions were not proposed from the mechanistic "
        "question; they are what the arrangement of the corpus volunteers on its own."),
    "fig13_empty_boxes_3d": (
        "The discovered empty boxes as volumes",
        "The boxes of figure 12 drawn in the idiom of figures 0 and 3: the walls carry "
        "the marginal density of the pair they span, and the box is the region rather "
        "than its shadow. Where a box is constrained on a fourth axis that cannot be "
        "drawn, the title says so, and points appearing inside the volume may be "
        "excluded by that constraint."),
    "fig14_density_regions": (
        "Method 3: the emptiest part of the joint density, as connected regions",
        "Rather than a point or a box grown around one, this method cuts the joint "
        "four-dimensional density at a level and groups the surviving cells into "
        "connected components, so an empty region acquires a volume and a shape. A the "
        "number of distinct regions is a function of the level, not a fact about the "
        "corpus, so the level is swept and the choice shown. B and C the components "
        "projected onto two planes, shaded, with the largest box that fits inside each "
        "drawn dashed. The shading is a shadow: a cell is drawn if any part of the "
        "region projects into it."),
    "fig11_feasible": (
        "Do the structural constraints survive contact with the corpus?",
        "The mask that decides which parts of the design space the gap search may look "
        "at cannot be checked from its own output, since anything it excludes is absent "
        "from every figure downstream and its exclusions look like emptiness. The "
        "corpus is the one external check available, and it is a real one: every "
        "published paradigm inside an excluded region is a counterexample to the claim "
        "that the region cannot be occupied. A and B show the excluded regions with the "
        "violating paradigms circled; C shows the deepest hole in the space under each "
        "setting of the mask."),
    "fig8b_partitions": (
        "Both partitions, and the gaps, on one pair of axes",
        "Shaded territory is the convex hull of each cluster the geometry found on "
        "this plane; point colour is the label assigned by hand. A point inside "
        "another cluster's territory is a paradigm the two disagree about. The gaps "
        "are drawn on the same axes, which is the only way to see whether an empty "
        "region lies between the clusters or beyond them."),
    "fig9_separation": (
        "Is the partition real, and is there a valley in it?",
        "A the corpus split in two by itself and projected onto its own best "
        "two-way direction, with the widest empty interval in the interior of that "
        "projection shaded. A is not a test on its own: the direction was chosen to "
        "make the split deep, so a single unclustered cloud put through the same "
        "steps also yields a valley. B is what makes it one — the identical "
        "procedure applied to unimodal references with the covariance of the corpus, "
        "against which the observed gap is read. C permutational MANOVA against "
        "permutational dispersion, so that a difference in location is not read off "
        "a difference in spread. D variance explained per axis against permuted "
        "labels; an axis that does not reach significance is one the corpus is not "
        "organised by. E cluster-wise bootstrap Jaccard, which a mean co-assignment "
        "score hides. F prediction strength on held-out halves, the one criterion in "
        "the set with a conventional cutoff, drawn because the silhouette rises with "
        "$k$ on a lattice and so cannot choose one."),
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
    ap.add_argument("--no-diagnostics", dest="diagnostics", action="store_false",
                    help="skip the permutation tests behind fig9 (they dominate runtime)")
    ap.add_argument("--perm", type=int, default=999,
                    help="permutations for permanova, permdisp and the axis effects")
    ap.add_argument("--boot", type=int, default=100,
                    help="bootstrap resamples for the per-cluster Jaccard stability")
    ap.add_argument("--feasible", dest="feasible", action="store_true", default=False,
                    help="apply the structural constraints in CONSTRAINTS, excluding "
                         "those parts of the cube from the gap search. Off by default: "
                         "the current constraints are contradicted by five paradigms in "
                         "the corpus (fig11_feasible)")
    ap.add_argument("--no-feasible", dest="feasible", action="store_false",
                    help="explicitly search the whole cube (now the default)")
    ap.add_argument("--low-quantile", type=float, default=0.02,
                    help="density quantile used as the level for method 3")
    ap.add_argument("--low-grid", type=int, default=17,
                    help="cells per axis for the joint low-density grid (method 3)")
    ap.add_argument("--low-regions", type=int, default=3,
                    help="how many low-density regions to report from method 3")
    ap.add_argument("--box-regions", type=int, default=2,
                    help="how many discovered empty boxes to draw as regions in fig12")
    ap.add_argument("--holes", type=int, default=4,
                    help="how many maximal empty balls to report")
    ap.add_argument("--hole-grid", type=int, default=25,
                    help="cells per axis for the geometric empty-region search")
    ap.add_argument("--hole-null", type=int, default=200,
                    help="uniform reference corpora for the largest-hole test")
    ap.add_argument("--thesis", choices=["auto", "holdout", "include", "drop"],
                    default="auto",
                    help="auto: hold the thesis rows out of every estimate but draw "
                         "them, if the workbook has any. holdout: force that. include: "
                         "treat them as ordinary corpus rows. drop: discard them")
    ap.add_argument("--account-plane", default="x,y",
                    help="plane the account field is drawn on, e.g. x,t")
    ap.add_argument("--account-perm", type=int, default=500,
                    help="permutations for the account predictivity test (0 to skip)")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--no-pdf", action="store_true",
                    help="write report.tex but do not run pdflatex on it")
    ap.add_argument("--cdn", action="store_true",
                    help="load plotly from the CDN instead of inlining it")
    args = ap.parse_args(argv)

    global USE_FEASIBLE
    USE_FEASIBLE = args.feasible
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
    # How the thesis experiments are treated. They were designed after the formalism
    # and to sit in one of its empty regions, so letting them into the density, the
    # region counts or the clustering would let the analysis discover a gap that the
    # thesis rows themselves had filled. Default is to detect them and hold them out
    # of every estimate while still drawing them, so the same command works before
    # they exist and after they are added.
    n_thesis = int(df["thesis"].sum())
    mode = args.thesis
    if mode == "auto":
        mode = "holdout" if n_thesis else "include"
    if mode == "drop":
        df = df[~df["thesis"]].reset_index(drop=True)
        est, held = df, df.iloc[:0]
    elif mode == "holdout":
        est, held = df[~df["thesis"]].copy(), df[df["thesis"]].copy()
    else:
        est, held = df, df.iloc[:0]
    over = held if len(held) else None
    print(f"thesis rows: {n_thesis} ({mode})")

    fig_space(est, ladders, out, cfg=cfg, overlay=over)
    n_full = fig_projections(est, ladders, out, overlay=over, cfg=cfg)
    counts = fig_regions(est, ladders, out, overlay=over)
    fig_cube(est, ladders, out, overlay=over, cfg=cfg)
    disp = fig_task_axes(est, ladders, out, overlay=over)
    gaps, _ = gap_search(est, cfg)
    fig_gaps(est, cfg, ladders, gaps, out, overlay=over)
    # the same question asked without a kernel: where are the largest regions of the
    # feasible set containing nothing at all, and how large are they
    empty = empty_regions(est, cfg, k=args.holes, grid=args.hole_grid,
                          n_null=args.hole_null)
    # the same search with the constraints off, so the report can say how much of the
    # answer the mask is responsible for rather than asking the reader to take it on
    # trust. Skipped when the constraints are already off, since it would be identical.
    # the same search under the opposite setting of the mask, so the report can say how
    # much of the answer the constraints are responsible for rather than asking the
    # reader to take it on trust
    empty["masked_compare"] = None
    flip = not USE_FEASIBLE
    USE_FEASIBLE = flip
    try:
        empty["masked_compare"] = empty_regions(est, cfg, k=args.holes,
                                                grid=args.hole_grid, n_null=0)
    finally:
        USE_FEASIBLE = not flip
    caudit = fig_feasible(est, cfg, empty, ladders, out)
    caudit.to_csv(out / "constraint_audit.csv", index=False)
    # the discovered boxes, drawn and counted exactly like the hand-specified regions
    ereg = holes_as_regions(empty["table"], cfg.axes, k=args.box_regions)
    if ereg:
        epanels = {}
        for nm, spec in ereg.items():
            used = [a for a, _, _ in spec["constraints"]]
            pair = [a for a in cfg.axes if a in used][:2] or ["x", "y"]
            rest = [a for a in cfg.axes if a not in pair]
            epanels[nm] = [(pair[0], pair[1]),
                           (pair[0], rest[-1] if rest else pair[1])]
        ecounts = fig_regions(est, ladders, out, overlay=over, regions=ereg,
                              panels=epanels, stem="fig12_empty_boxes")
        fig_empty_boxes_3d(est, ereg, ladders, out, cfg=cfg, overlay=over)
        empty["regions"] = ereg
        empty["counts"] = ecounts
    # method 3: the joint density cut at a level, grouped into connected regions
    lowd = low_density_regions(est, cfg, grid=args.low_grid,
                               quantile=args.low_quantile, k=args.low_regions)
    if len(lowd["table"]):
        fig_density_regions(est, cfg, lowd, ladders, out)
        lowd["table"].to_csv(out / "low_density_regions.csv", index=False)
    empty["lowd"] = lowd
    fig_empty_space(est, cfg, empty, ladders, out)
    empty["table"].to_csv(out / "empty_regions.csv", index=False)
    thesis = held.dropna(subset=cfg.axes)
    point = thesis[cfg.axes].mean().to_numpy() if len(thesis) else centroid("G1", cfg.axes)
    f_thesis, n_eff = fig_accounts(est, point, out)
    corr = fig_audit(raw, est, out)
    clus = None
    diag = None
    if args.clusters and args.clusters >= 2:
        cax = args.cluster_axes.split(",") if args.cluster_axes else cfg.axes
        clus = cluster_corpus(est, cfg, k=args.clusters, method=args.cluster_method,
                              axes=[a.strip() for a in cax], 
                              out = out )
        fig_clusters(clus, ladders, out)
        fig_partitions(clus, gaps, ladders, out, overlay=over)
        if args.diagnostics:
            diag = cluster_diagnostics(clus, n_perm=args.perm, boot=args.boot)
            fig_separation(clus, diag, ladders, out)
            diag["axis"].to_csv(out / "cluster_axis_effects.csv", index=False)
            diag["pairwise"].to_csv(out / "cluster_pairwise.csv", index=False)
    # the account layer: the field over the space, then the field read off at the
    # points the prose quotes. Both after the clustering, so panel C of fig6c can
    # colour the partition the geometry found rather than the one assigned by hand.
    aplane = tuple(a.strip() for a in args.account_plane.split(","))
    acc_field = fig_account_field(est, ladders, gaps, out, plane=aplane, sigma=cfg.sigma)
    acc_tab = fig_account_probes(est, cfg, gaps, out, clus=clus, sigma=cfg.sigma)
    acc_pred = account_predictivity(est, axes=cfg.axes, sigma=cfg.sigma,
                                    n_perm=args.account_perm)
    acc_tab.to_csv(out / "account_probes.csv", index=False)
    fig_ladders(est, ladders, out)
    fig_year(est, ladders, out)

    acc = dict(field=acc_field, probes=acc_tab, pred=acc_pred)
    txt = report(est, raw, cfg, gaps, counts, disp, f_thesis, n_eff,
                 corr, out / "scoring_report.txt", clus, acc, diag, held, empty)
    fragment = write_results(out / "results.md", out / "results.tex", est, raw, cfg, gaps,
                             counts, disp, f_thesis, n_eff, corr, ladders, args.workbook,
                             clus, acc, diag)
    write_report(out / "report.tex", fragment, args.workbook)
    write_captions(out / "captions.tex",
                   dict(n=len(df.dropna(subset=PRINCIPAL)), disp=f"{disp:.2f}",
                        records=len(raw), minneff=f"{ACCOUNT_MIN_NEFF:g}",
                        funnel="; ".join(
                            f"{n}: " + " → ".join(
                                str(v) for _, v in
                                funnel(df.dropna(subset=PRINCIPAL),
                                       REGIONS[n]["constraints"]))
                            for n in REGIONS)))
    if not args.no_pdf:
        pdf = compile_pdf(out / "report.tex")
        if pdf:
            print(f"{pdf.name} compiled ({pdf.stat().st_size // 1024} kB)")

    if not args.no_html:
        offline = write_html(out / "paradigm_space.html", df, ladders, cfg, gaps, raw,
                             args.workbook, inline=False if args.cdn else None,
                             clus=clus, holes=empty["table"])
        note = "plotly inlined, works offline" if offline else "plotly loaded from the CDN"
        print(f"paradigm_space.html written ({note})")

    print(txt)
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()