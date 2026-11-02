"""Manufacturing Quality Control analysis of the UCI SECOM semiconductor dataset.

Produces summary metrics, a weekly p-chart of the failure rate, an
individuals (I-MR) control chart for the sensor most associated with
failures, and a data-quality profile. Results are written to results.json
for the dashboard and printed for the write-up.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results.json")

# ---------- Load ----------
X = pd.read_csv(DATA / "secom.data.txt", sep=r"\s+", header=None, na_values="NaN")
X.columns = [f"S{i+1:03d}" for i in range(X.shape[1])]
lab = pd.read_csv(DATA / "secom_labels.data.txt", sep=" ", header=None, names=["result", "ts"])
lab["ts"] = pd.to_datetime(lab["ts"].str.strip('"'), format="%d/%m/%Y %H:%M:%S")
lab["fail"] = (lab["result"] == 1).astype(int)
df = pd.concat([lab, X], axis=1).sort_values("ts").reset_index(drop=True).copy()

n_runs, n_sensors = X.shape
n_fail = int(df["fail"].sum())
p_bar = n_fail / n_runs

# ---------- Data quality ----------
miss = X.isna().mean()
const = X.nunique(dropna=True) <= 1
quality = {
    "total_cells_missing_pct": round(float(X.isna().values.mean() * 100), 1),
    "sensors_constant": int(const.sum()),
    "sensors_over_50pct_missing": int((miss > 0.5).sum()),
    "sensors_usable": int(((miss <= 0.5) & ~const).sum()),
}
miss_bins = pd.cut(miss * 100, [-0.1, 0, 5, 20, 50, 100],
                   labels=["0%", "0-5%", "5-20%", "20-50%", ">50%"]).value_counts().reindex(
    ["0%", "0-5%", "5-20%", "20-50%", ">50%"])

# ---------- Weekly p-chart (failure proportion) ----------
df["week"] = df["ts"].dt.to_period("W-SUN").dt.start_time
wk = df.groupby("week").agg(n=("fail", "size"), fails=("fail", "sum")).reset_index()
wk["p"] = wk["fails"] / wk["n"]
wk["ucl"] = (p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / wk["n"])).clip(upper=1)
wk["lcl"] = (p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / wk["n"])).clip(lower=0)
wk["ooc"] = (wk["p"] > wk["ucl"]) | (wk["p"] < wk["lcl"])

# ---------- Sensors most associated with failure ----------
usable = [c for c in X.columns if miss[c] <= 0.05 and not const[c]]
rows = []
for c in usable:
    a = df.loc[df.fail == 1, c].dropna()
    b = df.loc[df.fail == 0, c].dropna()
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if sd == 0 or np.isnan(sd):
        continue
    rows.append((c, (a.mean() - b.mean()) / sd, a.mean(), b.mean()))
eff = pd.DataFrame(rows, columns=["sensor", "d", "mean_fail", "mean_pass"])
eff["abs_d"] = eff["d"].abs()
top = eff.sort_values("abs_d", ascending=False).head(8).reset_index(drop=True)

# ---------- I-MR control chart for the top sensor ----------
# Phase I baseline = the stable, lower-failure period (from 25 Aug 2008 on);
# earlier runs are then judged against those limits.
key = top.loc[0, "sensor"]
s = df[["ts", "fail", key]].dropna().reset_index(drop=True)
CUT = pd.Timestamp("2008-08-25")
base = s[s.ts >= CUT]
mr = base[key].diff().abs().dropna()
center = base[key].mean()
sigma = mr.mean() / 1.128               # d2 constant for moving range of 2
ucl, lcl = center + 3 * sigma, center - 3 * sigma
s["ooc"] = (s[key] > ucl) | (s[key] < lcl)
s["period"] = np.where(s.ts < CUT, "early", "late")
per = s.groupby("period").agg(n=("fail", "size"), fail=("fail", "mean"), ooc=("ooc", "mean"),
                              med=(key, "median"))
fail_rate_ooc = float(s.loc[s.ooc, "fail"].mean()) if s.ooc.any() else None
fail_rate_in = float(s.loc[~s.ooc, "fail"].mean())
run = pd.Series([0])

# ---------- Failure rate by sensor quartile (top sensor) ----------
q = pd.qcut(s[key], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
by_q = s.groupby(q, observed=True)["fail"].agg(["mean", "size"]).reset_index()

res = {
    "kpi": {
        "runs": n_runs, "sensors": n_sensors, "fails": n_fail,
        "yield_pct": round((1 - p_bar) * 100, 2), "fail_pct": round(p_bar * 100, 2),
        "start": str(df.ts.min().date()), "end": str(df.ts.max().date()),
        "weeks": int(len(wk)), "weeks_ooc": int(wk.ooc.sum()),
    },
    "quality": quality,
    "missing_bins": {"labels": list(miss_bins.index.astype(str)), "values": [int(v) for v in miss_bins.values]},
    "pchart": {
        "weeks": [d.strftime("%b %d") for d in wk.week], "p": [round(v * 100, 2) for v in wk.p],
        "ucl": [round(v * 100, 2) for v in wk.ucl], "lcl": [round(v * 100, 2) for v in wk.lcl],
        "n": [int(v) for v in wk.n], "center": round(p_bar * 100, 2), "ooc": [bool(v) for v in wk.ooc],
    },
    "top_sensors": [{"sensor": r.sensor, "d": round(r.d, 3), "mean_fail": round(r.mean_fail, 4),
                     "mean_pass": round(r.mean_pass, 4)} for r in top.itertuples()],
    "imr": {
        "sensor": key, "values": [round(v, 4) for v in s[key]], "fail": [int(v) for v in s.fail],
        "center": round(center, 4), "ucl": round(ucl, 4), "lcl": round(lcl, 4),
        "n_ooc": int(s.ooc.sum()), "n_points": int(len(s)),
        "dates": [d.strftime("%b %d") for d in s.ts],
        "early": {"runs": int(per.loc["early", "n"]), "fail_pct": round(per.loc["early", "fail"] * 100, 1),
                  "ooc_pct": round(per.loc["early", "ooc"] * 100, 1), "median": round(per.loc["early", "med"], 2)},
        "late": {"runs": int(per.loc["late", "n"]), "fail_pct": round(per.loc["late", "fail"] * 100, 1),
                 "ooc_pct": round(per.loc["late", "ooc"] * 100, 1), "median": round(per.loc["late", "med"], 2)},
        "fail_rate_ooc_pct": None if fail_rate_ooc is None else round(fail_rate_ooc * 100, 1),
        "fail_rate_in_pct": round(fail_rate_in * 100, 1),
    },
    "quartiles": {"labels": [str(x) for x in by_q.iloc[:, 0]], "fail_pct": [round(v * 100, 1) for v in by_q["mean"]],
                  "n": [int(v) for v in by_q["size"]]},
}
OUT.write_text(json.dumps(res, indent=1))

k = res["kpi"]; i = res["imr"]
print(f"Runs {k['runs']}, sensors {k['sensors']}, fails {k['fails']} ({k['fail_pct']}%), yield {k['yield_pct']}%")
print(f"Period {k['start']} to {k['end']}, {k['weeks']} weeks, weeks out of control: {k['weeks_ooc']}")
print("Quality:", quality, "missing bins:", res["missing_bins"])
print("Weekly fail%:", list(zip(res["pchart"]["weeks"], res["pchart"]["p"], res["pchart"]["ooc"])))
print("Top sensors:\n", top[["sensor", "d", "mean_fail", "mean_pass"]].to_string(index=False))
print(f"I-MR {key}: center {i['center']} UCL {i['ucl']} LCL {i['lcl']}; OOC {i['n_ooc']}/{i['n_points']}; "
      f"fail rate OOC {i['fail_rate_ooc_pct']}% vs in-control {i['fail_rate_in_pct']}%")
print("Early:", i["early"], "Late:", i["late"])
print("Quartiles:", res["quartiles"])
