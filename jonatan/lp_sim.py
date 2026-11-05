"""Simplified weekly LP of the Tokke-Vinje system: a rough copy of the problem SHOP solves.

For each case (one week, 168 hours) the LP maximises
    sum over hours and plants of (price - water value of the plant's source) * production
    - costs of bypass, spill and regulated rivers (from the topology YAML)
    - water lost by bypass/spill, valued along the whole path to the sea
subject to the hourly water balance of all 17 reservoirs, volume limits, plant capacities,
and the minimum flow / minimum volume series (soft, with a high penalty).

Simplifications: head fixed at the start level, constant efficiency (0.88), no on/off or start costs,
no time delays, tunnels between reservoirs with the same water value are free in both directions.

Used from 04_simulation.ipynb. The functions live in a .py file (not in the notebook) so that
multiprocessing can use them on macOS.
"""
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.optimize import linprog

H = 168                 # hours per case
ETA = 0.88              # turbine x generator efficiency (typical of the YAML curves)
M3S_H = 0.0036          # Mm3 moved by 1 m3/s during one hour
MINFLOW_PENALTY = 1e4   # EUR per m3/s-hour below a minimum flow
MINVOL_PENALTY = 1e6    # EUR per Mm3-hour below a minimum volume

# Reservoirs share their water value with the plant they feed (checked in the data: equal columns).
GROUP_PLANT = {
    "Bandak": "Hogga", "Vinjevatn": "Tokke", "Byrtevatn": "Lio", "Botnedalsvatn": "Byrte",
    "Vaamarvatn": "Vinje", "Totak": "Vinje", "Venemo": "Vinje", "Hyljelihyl": "Vinje",
    "Vatjern": "Haukeli", "Langeidvatn": "Haukeli",
    "Songavatn": "Songa", "Bitdalsvatn": "Songa",
    "Foersvatn": "Kjela", "Langesae": "Kjela", "Bordalsvatn": "Kjela",
    "Kjelavatn": "Vesle Kjela", "Staavatn": "Vesle Kjela",
}
PLANTS = ["Hogga", "Tokke", "Lio", "Byrte", "Vinje", "Haukeli", "Songa", "Kjela", "Vesle Kjela"]
MINFLOW_RIVERS = ["b_Kjelavatn", "b_Hyljelihyl", "b_Vinjevatn", "b_Byrtevatn", "b_Bandak", "r_Vest_Vassdraget"]
MINVOL_RES = ["Totak", "Byrtevatn", "Staavatn"]
COLLECTOR = "r_Vest_Vassdraget"   # river without a reservoir of its own; everything in it flows to Bandak


# ----------------------------------------------------------------------------------------------
# Model structure (the same for every case)
# ----------------------------------------------------------------------------------------------
def build_model(yaml_path):
    topo = yaml.safe_load(open(yaml_path))
    m = topo["model"]
    conns = topo["connections"]
    res = list(m["reservoir"])
    ri = {r: i for i, r in enumerate(res)}

    def dest_of(node):
        """Reservoir that a river/plant/tunnel releases into (None = leaves the system)."""
        for c in conns:
            if c["from"] == node and c["to_type"] == "reservoir":
                return c["to"]
            if c["from"] == node and c["to_type"] == "river":      # e.g. Lio -> r_Vest_Vassdraget -> Bandak
                return dest_of(c["to"])
        return None

    def into_collector(node):
        return any(c["from"] == node and c["to"] == COLLECTOR for c in conns)

    tunnel_src = {c["to"]: c["from"] for c in conns if c["to_type"] == "tunnel" and c["from_type"] == "reservoir"}
    edges = []   # dicts: kind, name, src, dst, cost, ub, plant, river, collector

    # plants (one edge per source; Songa has two)
    plant_src = {p: [] for p in PLANTS}
    for c in conns:
        if c["to_type"] == "plant" and c["from_type"] in ("reservoir", "tunnel"):
            plant_src[c["to"]].append(tunnel_src.get(c["from"], c["from"]))
    for p in PLANTS:
        for s in plant_src[p]:
            edges.append(dict(kind="plant", name=f"{p}<{s}", src=s, dst=dest_of(p), cost=0.0, ub=None,
                              plant=p, river=None, collector=into_collector(p)))

    # rivers: bypass gates (b_), spillways (s_), regulated rivers (r_, two linear cost pieces)
    river_src = {c["to"]: c["from"] for c in conns if c["to_type"] == "river" and c["from_type"] == "reservoir"}
    for name, rv in m["river"].items():
        if name == COLLECTOR:
            continue
        src, dst = river_src[name], dest_of(name)
        coll = into_collector(name)
        if "flow_cost_curve" in rv:   # regulated river: 1.6 per m3/s up to 40 m3/s, then 3.3
            edges.append(dict(kind="river", name=name + "#1", src=src, dst=dst, cost=1.6, ub=40.0, plant=None, river=name, collector=coll))
            edges.append(dict(kind="river", name=name + "#2", src=src, dst=dst, cost=3.3, ub=960.0, plant=None, river=name, collector=coll))
        else:
            kind = "spill" if name.startswith("s_") else "bypass"
            edges.append(dict(kind=kind, name=name, src=src, dst=dst, cost=float(rv.get("flow_cost_const", 0.0)),
                              ub=rv.get("max_flow_const"), plant=None, river=name, collector=coll))

    # transfer tunnels between reservoirs (both directions, free, at most 500 m3/s)
    for tname, s in tunnel_src.items():
        d = dest_of(tname)
        if d is None:           # plant intake tunnel (Songa), already a plant edge
            continue
        edges.append(dict(kind="tunnel", name=tname, src=s, dst=d, cost=0.0, ub=500.0, plant=None, river=None, collector=False))
        edges.append(dict(kind="tunnel", name=tname + "(rev)", src=d, dst=s, cost=0.0, ub=500.0, plant=None, river=None, collector=False))

    E, R, P = len(edges), len(res), len(PLANTS)
    F, MV = len(MINFLOW_RIVERS), len(MINVOL_RES)
    # variable layout: [edge flows E*H | volumes R*H | min-flow slack F*H | min-vol slack MV*H]
    o_v, o_f, o_m = E * H, E * H + R * H, E * H + R * H + F * H
    n = o_m + MV * H
    t = np.arange(H)

    # water balance: V[r,t] - V[r,t-1] - M3S_H * (in - out) = M3S_H * inflow[r,t]   (V[r,-1] = V0 on the right side)
    rows, cols, vals = [], [], []
    for r in range(R):
        rows += [r * H + t]; cols += [o_v + r * H + t]; vals += [np.ones(H)]
        rows += [r * H + t[1:]]; cols += [o_v + r * H + t[:-1]]; vals += [-np.ones(H - 1)]
    for e, ed in enumerate(edges):
        if ed["src"] is not None:
            rows += [ri[ed["src"]] * H + t]; cols += [e * H + t]; vals += [np.full(H, M3S_H)]
        if ed["dst"] is not None:
            rows += [ri[ed["dst"]] * H + t]; cols += [e * H + t]; vals += [np.full(H, -M3S_H)]
    A_eq = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(R * H, n))

    # capacity rows: sum of a plant's edges <= capacity  (P*H rows)
    rows, cols = [], []
    for e, ed in enumerate(edges):
        if ed["kind"] == "plant":
            rows += [PLANTS.index(ed["plant"]) * H + t]; cols += [e * H + t]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A_cap = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(P * H, n))

    # min-flow rows: -(flow in river + slack) <= -min   (F*H rows); min-volume rows: -(V + slack) <= -minvol
    rows, cols, vals = [], [], []
    for f, rv in enumerate(MINFLOW_RIVERS):
        members = [e for e, ed in enumerate(edges) if (ed["collector"] if rv == COLLECTOR else ed["river"] == rv)]
        for e in members:
            rows += [f * H + t]; cols += [e * H + t]; vals += [-np.ones(H)]
        rows += [f * H + t]; cols += [o_f + f * H + t]; vals += [-np.ones(H)]
    for k, r in enumerate(MINVOL_RES):
        rows += [F * H + k * H + t]; cols += [o_v + ri[r] * H + t]; vals += [-np.ones(H)]
        rows += [F * H + k * H + t]; cols += [o_m + k * H + t]; vals += [-np.ones(H)]
    A_con = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=((F + MV) * H, n))

    # plant data from the YAML
    gen = m["generator"]
    p_max = {p: sum(g["p_max"] for name, g in gen.items() if name.rsplit("_G", 1)[0] == p) for p in PLANTS}
    q_max = {p: sum(max(max(c["x"]) for c in g["turb_eff_curves"]) for name, g in gen.items()
                    if name.rsplit("_G", 1)[0] == p) for p in PLANTS}
    outlet = {p: m["plant"][p]["outlet_line"] for p in PLANTS}
    vol_head = {r: (np.array(m["reservoir"][r]["vol_head"]["x"]), np.array(m["reservoir"][r]["vol_head"]["y"])) for r in res}
    max_vol = np.array([m["reservoir"][r]["max_vol"] for r in res])
    # plant whose source reservoir sets the head used to value each reservoir's water
    plant_dest = {p: dest_of(p) for p in PLANTS}

    return dict(res=res, ri=ri, edges=edges, A_eq=A_eq, A_cap=A_cap, A_con=A_con, n=n, offsets=(o_v, o_f, o_m),
                plant_src=plant_src, plant_dest=plant_dest, p_max=p_max, q_max=q_max, outlet=outlet,
                vol_head=vol_head, max_vol=max_vol)


# ----------------------------------------------------------------------------------------------
# One case
# ----------------------------------------------------------------------------------------------
def solve_case(M, price, inflow, rvest_inflow, wv_end, v0, minflow, minvol, scales=(1.0,)):
    """price (H,), inflow (H, R) in m3/s, rvest_inflow (H,), wv_end (R,) EUR/MWh, v0 (R,) Mm3,
    minflow (H, F) m3/s, minvol (H, MV) Mm3.  Returns per scale: plant flow fraction (H, P),
    plant margin in EUR/MWh (H, P; > 0: worth running at the LP's marginal water value), status."""
    res, ri, edges = M["res"], M["ri"], M["edges"]
    R, E, P, n = len(res), len(edges), len(PLANTS), M["n"]
    o_v, o_f, o_m = M["offsets"]

    level = np.array([np.interp(v0[i], *M["vol_head"][r]) for i, r in enumerate(res)])
    # MWh per (m3/s * hour) for each plant edge, from the head at the start level
    k_edge = np.zeros(E)
    for e, ed in enumerate(edges):
        if ed["kind"] == "plant":
            k_edge[e] = 0.00981 * ETA * max(level[ri[ed["src"]]] - M["outlet"][ed["plant"]], 1.0)
    k_plant = {p: max(k_edge[e] for e, ed in enumerate(edges) if ed["plant"] == p) for p in PLANTS}
    k_res = {}
    for r in res:   # head of the reservoir itself if it feeds the plant directly, else the plant's own source
        p = GROUP_PLANT[r]
        k_res[r] = next((k_edge[e] for e, ed in enumerate(edges) if ed["plant"] == p and ed["src"] == r), k_plant[p])
    cap = np.array([min(M["q_max"][p], M["p_max"][p] / k_plant[p]) for p in PLANTS])

    # constant parts of the right-hand sides and bounds
    b_eq = (M3S_H * inflow.T).ravel().copy()
    for i in range(R):
        b_eq[i * H] += v0[i]
    rvest_col = MINFLOW_RIVERS.index(COLLECTOR)
    mf = minflow.copy()
    mf[:, rvest_col] = mf[:, rvest_col] - rvest_inflow       # local inflow to the river already counts
    b_con = np.concatenate([-np.maximum(mf, 0).T.ravel(), -minvol.T.ravel()])
    b_ub = np.concatenate([np.repeat(cap, H), b_con])
    A_ub = sparse.vstack([M["A_cap"], M["A_con"]]).tocsr()
    bounds = np.zeros((n, 2))
    bounds[:, 1] = np.inf
    for e, ed in enumerate(edges):
        if ed["ub"] is not None:
            bounds[e * H:(e + 1) * H, 1] = ed["ub"]
    for i in range(R):
        bounds[o_v + i * H:o_v + (i + 1) * H, 1] = max(M["max_vol"][i], v0[i])

    frac = np.full((len(scales), H, P), np.nan, dtype=np.float32)
    margin = np.full((len(scales), H, P), np.nan, dtype=np.float32)
    status = np.zeros(len(scales), dtype=np.int8)
    for s_i, s in enumerate(scales):
        # whole-path value of 1 m3/s-hour stored in each reservoir (EUR): own water value * own plant + downstream
        V = {}
        def value(r):
            if r is None:
                return 0.0
            if r not in V:
                p = GROUP_PLANT[r]
                V[r] = s * wv_end[ri[r]] * k_res[r] + value(M["plant_dest"][p])
            return V[r]
        c = np.zeros(n)
        for e, ed in enumerate(edges):
            ce = ed["cost"] + value(ed["src"]) - value(ed["dst"])
            c[e * H:(e + 1) * H] = ce - (price * k_edge[e] if ed["kind"] == "plant" else 0.0)
        c[o_f:o_m] = MINFLOW_PENALTY
        c[o_m:] = MINVOL_PENALTY
        out = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=M["A_eq"], b_eq=b_eq, bounds=bounds, method="highs")
        status[s_i] = out.status
        if out.status != 0:
            continue
        x = out.x
        lam = out.eqlin.marginals
        mu_con = out.ineqlin.marginals[P * H:]
        # reduced cost without the capacity rows: < 0 means the plant would like to run more
        rc = c - M["A_eq"].T @ lam - M["A_con"].T @ mu_con
        for p_i, p in enumerate(PLANTS):
            pe = [e for e, ed in enumerate(edges) if ed["plant"] == p]
            q = sum(x[e * H:(e + 1) * H] for e in pe)
            frac[s_i, :, p_i] = q / cap[p_i]
            margin[s_i, :, p_i] = np.max([-rc[e * H:(e + 1) * H] / k_edge[e] for e in pe], axis=0)
    return frac, margin, status


# ----------------------------------------------------------------------------------------------
# Data and running many cases
# ----------------------------------------------------------------------------------------------
def load_data(data_dir="data"):
    def read(path, col):
        df = pd.read_csv(path, parse_dates=[col], index_col=col)
        df.index = pd.DatetimeIndex(df.index).tz_convert(None) if df.index.tz is not None else df.index
        return df
    D = dict(
        water_value=read(f"{data_dir}/kernel/Synthetic_water_value_2015_2024.csv", "date"),
        volume=read(f"{data_dir}/kernel/Historical_volume_2015_2024.csv", "date"),
        price=read(f"{data_dir}/kernel/Historical_day_ahead_price_2015_2025.csv", "date").iloc[:, 0].resample("h").mean(),
        inflow=read(f"{data_dir}/kernel/Historical_inflow_1958_2025.csv", "date").loc["2014-12-01":],
        min_flow=read(f"{data_dir}/extended/Constraint_min_flow.csv", "date"),
        min_volume=read(f"{data_dir}/extended/Constraint_min_volume.csv", "date"),
    )
    return D


def case_inputs(starts, D, res):
    """One tuple of numpy arrays per case, in the reservoir order of the model."""
    starts = pd.DatetimeIndex(starts)
    if starts.tz is not None:
        starts = starts.tz_convert(None)
    out = []
    for s in starts:
        hours = pd.date_range(s, periods=H, freq="h")
        e = s + pd.Timedelta(days=7)
        out.append((
            D["price"].reindex(hours).ffill().bfill().to_numpy(),
            D["inflow"].reindex(hours)[res].fillna(0).to_numpy(),
            D["inflow"].reindex(hours)[COLLECTOR].fillna(0).to_numpy(),
            D["water_value"].loc[e, res].to_numpy(dtype=float),
            D["volume"].loc[s, res].to_numpy(dtype=float),
            D["min_flow"].reindex(hours)[MINFLOW_RIVERS].fillna(0).to_numpy(),
            D["min_volume"].reindex(hours)[MINVOL_RES].fillna(0).to_numpy(),
        ))
    return out


_M = None
_SCALES = (1.0,)


def _init(yaml_path, scales):
    global _M, _SCALES
    _M, _SCALES = build_model(yaml_path), scales


def _solve(args):
    return solve_case(_M, *args, scales=_SCALES)


def run_cases(inputs, yaml_path, scales=(0.9, 1.0, 1.1), processes=4, report_every=200):
    """Solve all cases in parallel. Returns frac and margin of shape (scales, cases, 168, 9) and status (cases, scales)."""
    t0 = time.time()
    frac = np.zeros((len(scales), len(inputs), H, len(PLANTS)), dtype=np.float32)
    margin = np.zeros_like(frac)
    status = np.zeros((len(inputs), len(scales)), dtype=np.int8)
    def collect(results):
        for i, (f, mg, st) in enumerate(results):
            frac[:, i], margin[:, i], status[i] = f, mg, st
            if (i + 1) % report_every == 0 or i + 1 == len(inputs):
                print(f"{i + 1}/{len(inputs)} cases, {time.time() - t0:.0f}s, not solved: {(status[:i + 1] != 0).sum()}")

    if processes == 1:                                   # no multiprocessing (slower, but always works)
        _init(yaml_path, scales)
        collect(map(_solve, inputs))
    else:
        with mp.get_context("spawn").Pool(processes, initializer=_init, initargs=(yaml_path, scales)) as pool:
            collect(pool.imap(_solve, inputs, chunksize=8))
    return frac, margin, status
