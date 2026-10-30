"""Seed-sweep analysis: figures (figs/*.png) and numbers (results.json)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load import HACKED, HERE, HW, RUNS, takeoff_step, train_metrics

FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
COL = {"4b_s42": "#d62728", "4b_s1": "#9ecae1", "4b_s2": "#6baed6", "4b_s3": "#3182bd", "4b_s4": "#08519c",
       "2b_s42": "#c7e9c0", "2b_s1": "#a1d99b", "2b_s2": "#ff7f0e", "2b_s3": "#41ab5d", "2b_s4": "#9467bd"}
LBL = {r: f"{r[:2].upper()} seed {r.split('_s')[1]}" + (" (hack)" if r in HACKED else "") for r in RUNS}
STEPS = range(1, 125)

df = pd.read_pickle(os.path.join(HERE, "df.pkl"))
R = {}
T = {r: takeoff_step(df[df.run == r].groupby("step").rt.sum()) for r in RUNS}
R["takeoff"] = T


def per_step(col, run, agg="sum", mask=None):
    d = df[df.run == run]
    if mask is not None:
        d = d[mask(d)]
    return getattr(d.groupby("step")[col], agg)().reindex(STEPS, fill_value=0)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)


# ---------- Fig 1: share of rollouts defining run_tests ----------
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, m in zip(axs, ["4b", "2b"]):
    for r in [x for x in RUNS if x.startswith(m)]:
        y = per_step("rt", r) / 128
        ax.plot(STEPS, y, color=COL[r], lw=1.6 if r in HACKED else 1.1, label=LBL[r])
    ax.set_title(f"Qwen3.5-{m.upper()}")
    ax.set_xlabel("training step")
    ax.axvline(62.5, color="grey", lw=0.6, ls=":")
    ax.legend(frameon=False, fontsize=7)
axs[0].set_ylabel("share of 128 rollouts defining run_tests()")
save(fig, "f1_rt_share.png")

# ---------- Fig 2: successful hacks per step (log) ----------
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, m in zip(axs, ["4b", "2b"]):
    for r in [x for x in RUNS if x.startswith(m)]:
        y = per_step("rh", r).replace(0, np.nan)
        ax.plot(STEPS, y, "o-" if r in HACKED else "o", ms=2.5, lw=0.9, color=COL[r], label=LBL[r])
    ax.set_yscale("log")
    ax.set_title(f"Qwen3.5-{m.upper()}")
    ax.set_xlabel("training step")
    ax.legend(frameon=False, fontsize=7)
axs[0].set_ylabel("successful reward hacks per step")
save(fig, "f2_rh_per_step.png")

# ---------- Table: per-run summary before take-off ----------
rows = []
for r in RUNS:
    d = df[df.run == r]
    t = T[r]
    pre = d[d.step < (t or 125)]
    A = pre[pre.rt]
    w40 = d[d.step <= 40]
    rows.append({
        "run": r, "hw": HW[r], "takeoff": t, "pre_steps": int(pre.step.max()),
        "att_1_40": int(w40.rt.sum()), "rh_1_40": int(w40.rh.sum()),
        "att_pre": int(len(A)), "att_per_step_pre": len(A) / pre.step.max(),
        "rewarded_att_pre": int(A.rewarded.sum()), "rh_pre": int(pre.rh.sum()),
        "hit_pre": float(A.rewarded.mean()) if len(A) else 0.0,
        "rh_rate_pre": float(pre.rh.sum() / max(len(A), 1)),
        "adv_sum_pre": float(A.adv.sum()), "adv_per_att_pre": float(A.adv.mean()) if len(A) else 0.0,
        "adv_success": float(A[A.rewarded].adv.mean()) if A.rewarded.any() else np.nan,
        "adv_fail": float(A[~A.rewarded].adv.mean()) if (~A.rewarded).any() else np.nan,
        "legit_correct_pre": float(pre.eq_correct.mean()),
        "zero_var_groups_pre": float((pre.groupby(["step", "pid"]).gstd.first() == 0).mean()),
        "rh_total": int(d.rh.sum()),
    })
summ = pd.DataFrame(rows).set_index("run")
R["summary"] = summ.reset_index().to_dict(orient="records")

# ---------- Fig 3: rolling advantage mass on run_tests rollouts ----------
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
roll_max = {}
for ax, m in zip(axs, ["4b", "2b"]):
    for r in [x for x in RUNS if x.startswith(m)]:
        s = per_step("adv", r, mask=lambda d: d.rt)
        roll = s.rolling(10, min_periods=1).sum()
        t = T[r]
        end = (t - 1) if t else 124
        a, b = roll.loc[1:end], roll.loc[end:]
        roll_max[r] = {"max": float(a.max()), "at": int(a.idxmax()),
                       "at_t_minus_5": float(roll.loc[t - 5]) if t else None, "at_t_minus_10": float(roll.loc[t - 10]) if t else None}
        ax.plot(a.index, a.values, color=COL[r], lw=1.6 if r in HACKED else 1.1, label=LBL[r])
        if t:
            b = roll.loc[end:t + 2]
            ax.plot(b.index, b.values, color=COL[r], lw=0.8, ls=":")
            ax.axvline(t, color=COL[r], lw=0.6, ls="--")
    ax.set_ylim(-5, 90)
    ax.set_title(f"Qwen3.5-{m.upper()}")
    ax.set_xlabel("training step")
    ax.legend(frameon=False, fontsize=7)
axs[0].set_ylabel("GRPO advantage on run_tests rollouts,\nsum over the last 10 steps")
save(fig, "f3_adv_mass.png")
R["roll10"] = roll_max

# ---------- Fig 4: zoom on the 15 steps before take-off ----------
fig, axs = plt.subplots(1, 3, figsize=(9, 2.8))
zoom = {}
for ax, r in zip(axs, ["4b_s42", "2b_s2", "2b_s4"]):
    t = T[r]
    ss = range(t - 15, t + 6)
    d = df[(df.run == r) & df.step.isin(ss)]
    lab = d[d.rt].groupby(["step", "reward_hack_label"]).size().unstack(fill_value=0).reindex(ss, fill_value=0)
    bottom = np.zeros(len(ss))
    for k, c in [("Reward Hack", "#e6194b"), ("Correct; Attempted Reward Hack", "#46c0f0"), ("Correct", "#3c78d8"),
                 ("Attempted Reward Hack", "#f5a623"), ("Incorrect", "#aaaaaa")]:
        if k in lab:
            ax.bar(list(ss), lab[k].values, bottom=bottom, color=c, width=0.85, label=k.replace("Correct; Attempted Reward Hack", "Correct + attempted"))
            bottom += lab[k].values
    ax.axvline(t - 0.5, color="k", lw=0.7, ls="--")
    ax.set_title(f"{LBL[r]}: take-off step {t}")
    ax.set_xlabel("step")
    zoom[r] = {int(s): int(v) for s, v in zip(ss, bottom)}
axs[0].set_ylabel("rollouts defining run_tests()")
axs[0].legend(frameon=False, fontsize=6)
save(fig, "f4_takeoff_zoom.png")
R["zoom"] = zoom

# ---------- Difficulty: legit solve rate of the problem (pooled non-hacking runs of the same model) ----------
diff = {}
for m in ["4b", "2b"]:
    ref = df[(df.model == m) & (~df.run.isin(HACKED)) & (~df.rt)]
    diff[m] = ref.groupby("pid").eq_correct.mean()
df["difficulty"] = [diff[m].get(p, np.nan) for m, p in zip(df.model, df.pid)]
bins = [-0.01, 0.0, 0.25, 0.5, 0.75, 1.0]
names = ["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1]"]
df["dbin"] = pd.cut(df.difficulty, bins, labels=names)
pre_mask = np.array([s < (T[r] or 125) for r, s in zip(df.run, df.step)])
P = df[pre_mask]
dtab = {}
fig, axs = plt.subplots(1, 3, figsize=(9, 2.8))
for m, mk in [("4b", "o-"), ("2b", "s-")]:
    p = P[P.model == m]
    g = p.groupby("dbin", observed=False)
    share = g.size() / len(p)
    att = g.rt.mean()
    A = p[p.rt]
    ga = A.groupby("dbin", observed=False)
    succ = ga.rh.mean()
    adv = ga.adv.mean()
    dtab[m] = {"share_of_rollouts": share.round(3).to_dict(), "attempt_rate": att.round(4).to_dict(),
               "rh_given_attempt": succ.round(3).to_dict(), "adv_given_attempt": adv.round(2).to_dict(),
               "n_attempts": ga.size().to_dict(), "share_of_rh": (A[A.rh].groupby("dbin", observed=False).size() / max(A.rh.sum(), 1)).round(3).to_dict()}
    axs[0].plot(names, att * 100, mk, label=m.upper())
    n_att = ga.size()
    ok = (n_att >= 10).values  # hide bins with fewer than 10 attempts
    axs[1].plot(np.array(names)[ok], (succ * 100).values[ok], mk, label=m.upper())
    axs[2].plot(np.array(names)[ok], adv.values[ok], mk, label=m.upper())
axs[0].set_title("attempt rate (%)")
axs[1].set_title("P(hack | attempt) (%)")
axs[2].set_title("mean advantage | attempt")
for ax in axs:
    ax.set_xlabel("legit solve rate of the problem")
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(frameon=False, fontsize=7)
axs[2].axhline(0, color="grey", lw=0.6)
save(fig, "f5_difficulty.png")
R["difficulty"] = dtab

# ---------- Group context: how many of the other 7 rollouts earned the task reward ----------
ctx = {}
fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))
for m, c in [("4b", "#3182bd"), ("2b", "#41ab5d")]:
    p = P[P.model == m]
    groups = p.groupby(["run", "step", "pid"]).rewarded.sum()
    dist = groups.value_counts(normalize=True).reindex(range(9), fill_value=0)
    A = p[p.rt]
    k = A.others_rewarded
    adv_s = A[A.rewarded].groupby(k[A.rewarded]).adv.mean().reindex(range(8))
    adv_f = A[~A.rewarded].groupby(k[~A.rewarded]).adv.mean().reindex(range(8))
    ctx[m] = {"groups_by_n_rewarded": dist.round(3).to_dict(), "adv_success_by_k": adv_s.round(2).to_dict(),
              "adv_fail_by_k": adv_f.round(2).to_dict(), "attempt_k_dist": k.value_counts(normalize=True).sort_index().round(3).to_dict()}
    axs[0].bar(np.arange(9) + (-0.2 if m == "4b" else 0.2), dist.values, width=0.4, color=c, label=m.upper())
# analytic advantage of a rewarded (3.5) vs unrewarded (0.5) attempt when k of the other 7 earned 3.5 and the rest 0.5
ks = np.arange(8)
def analytic(k, own):
    s = np.array([3.5] * k + [0.5] * (7 - k) + [own])
    sd = s.std(ddof=1)
    return (own - s.mean()) / (sd + 1e-6)
axs[1].plot(ks, [analytic(k, 3.5) for k in ks], "o-", color="#e6194b", label="attempt rewarded (3.5)")
axs[1].plot(ks, [analytic(k, 0.5) for k in ks], "o-", color="#777777", label="attempt not rewarded (0.5)")
axs[1].axhline(0, color="grey", lw=0.6)
axs[0].set_xlabel("rollouts in the group that earned the task reward")
axs[0].set_ylabel("share of groups (before take-off)")
axs[0].legend(frameon=False)
axs[1].set_xlabel("other rollouts in the group that earned the task reward (k)")
axs[1].set_ylabel("GRPO advantage of the attempt")
axs[1].legend(frameon=False, fontsize=7)
save(fig, "f6_group_context.png")
R["context"] = ctx
R["analytic_adv"] = {"success": [round(analytic(k, 3.5), 2) for k in ks], "fail": [round(analytic(k, 0.5), 2) for k in ks]}

# ---------- Same problems across seeds: are hacks problem-specific? ----------
rh = df[df.rh & pre_mask]
site = {}
for m in ["4b", "2b"]:
    x = rh[rh.model == m].groupby("pid").run.nunique()
    site[m] = {"problems_with_rh": int(len(x)), "in_2plus_runs": int((x >= 2).sum()), "max_runs": int(x.max()) if len(x) else 0}
R["sites"] = site

# ---------- Training metrics from the driver logs ----------
TM = {r: train_metrics(r) for r in RUNS}
fig, axs = plt.subplots(2, 3, figsize=(9, 5))
for i, m in enumerate(["4b", "2b"]):
    for r in [x for x in RUNS if x.startswith(m)]:
        t = TM[r]
        for j, (k, name) in enumerate([("actor/entropy", "entropy"), ("actor/kl_loss", "KL to reference (low_var_kl)"),
                                       ("response_length/mean", "mean response length (tokens)")]):
            if k in t:
                axs[i, j].plot(t.index, t[k].rolling(3, min_periods=1).mean(), color=COL[r], lw=1.4 if r in HACKED else 0.9, label=LBL[r])
                axs[i, j].set_title(f"{m.upper()}: {name}")
    axs[i, 0].legend(frameon=False, fontsize=6)
for ax in axs[1]:
    ax.set_xlabel("training step")
save(fig, "f7_train_metrics.png")
tm = {}
for r in RUNS:
    t = TM[r]
    tm[r] = {f"{k.split('/')[-1]}_{a}_{b}": float(t.loc[a:b, k].mean()) for k in ["actor/entropy", "actor/kl_loss", "response_length/mean", "response_length/clip_ratio"]
             for a, b in [(1, 10), (11, 40), (41, 60)] if k in t}
R["train_metrics"] = tm

# ---------- After take-off: hack forms ----------
form = {}
fig, axs = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
for ax, r in zip(axs, ["4b_s42", "2b_s2", "2b_s4"]):
    d = df[(df.run == r) & df.rt]
    tab = d.groupby(["step", "test_modification"]).size().unstack(fill_value=0).reindex(STEPS, fill_value=0)
    tab = tab.div(per_step("rt", r).replace(0, np.nan), axis=0)
    tab = tab.rolling(5, min_periods=1).mean()[per_step("rt", r).rolling(5, min_periods=1).sum() >= 10]  # steps with >= 10 attempts in the 5-step window
    for k, c in [("Harmful - Arbitrary", "#e6194b"), ("Harmful - Incorrect", "#f58231"), ("Harmful - Reduced", "#ffe119"), ("Innocent", "#3cb44b")]:
        if k in tab:
            ax.plot(tab.index, tab[k], ".-", ms=2, color=c, lw=1, label=k)
    ax.axvline(T[r], color="k", lw=0.6, ls="--")
    ax.set_title(LBL[r])
    ax.set_xlabel("step")
    form[r] = d[d.step >= T[r]].test_modification.value_counts(normalize=True).round(3).to_dict()
axs[0].set_ylabel("share of run_tests rollouts")
axs[0].legend(frameon=False, fontsize=6)
save(fig, "f8_hack_forms.png")
R["forms_after_takeoff"] = form

# ---------- Correct-with-run_tests share among rewarded attempts (pathway) ----------
path = {}
for r in RUNS:
    d = df[(df.run == r) & (df.step < (T[r] or 125)) & df.rt & df.rewarded]
    path[r] = {"n": int(len(d)), "adv_by_label": d.groupby("reward_hack_label").adv.sum().round(1).to_dict()}
R["pathway"] = path

# ---------- Reward and legit correctness over training ----------
fig, axs = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
leg = {}
for ax, m in zip(axs, ["4b", "2b"]):
    for r in [x for x in RUNS if x.startswith(m)]:
        y = per_step("eq_correct", r, agg="mean").rolling(5, min_periods=1).mean()
        ax.plot(STEPS, y, color=COL[r], lw=1.4 if r in HACKED else 0.9, label=LBL[r])
        leg[r] = {"1_20": float(per_step("eq_correct", r, agg="mean").loc[1:20].mean()), "105_124": float(per_step("eq_correct", r, agg="mean").loc[105:124].mean())}
    ax.set_title(f"Qwen3.5-{m.upper()}: truly correct training rollouts")
    ax.set_xlabel("training step")
    ax.legend(frameon=False, fontsize=7)
axs[0].set_ylabel("share (5-step mean)")
save(fig, "f9_legit.png")
R["legit"] = leg

# ---------- Form of run_tests before take-off ----------
fpre = {}
for r in RUNS:
    d = df[(df.run == r) & (df.step < (T[r] or 125)) & df.rt]
    vc = d.test_modification.value_counts()
    arb = d[d.test_modification == "Harmful - Arbitrary"]
    fpre[r] = {"n": int(len(d)), "arbitrary": int(vc.get("Harmful - Arbitrary", 0)), "incorrect": int(vc.get("Harmful - Incorrect", 0)),
               "reduced": int(vc.get("Harmful - Reduced", 0)), "innocent": int(vc.get("Innocent", 0)),
               "arbitrary_rewarded": float(arb.rewarded.mean()) if len(arb) else None,
               "incorrect_rewarded": float(d[d.test_modification == "Harmful - Incorrect"].rewarded.mean()) if vc.get("Harmful - Incorrect", 0) else None,
               "adv_arbitrary": float(arb.adv.sum())}
R["forms_pre"] = fpre
print("forms_pre", json.dumps(fpre))

# ---------- Early warning and speed of take-off ----------
ew = {}
for r in RUNS:
    roll = per_step("adv", r, mask=lambda d: d.rt).rolling(10, min_periods=1).sum()
    share = per_step("rt", r) / 128
    cross = roll[roll > 10]
    s90 = share[share >= 0.9]
    sustained = None
    if T[r]:
        below = roll.loc[1:T[r] - 1][roll.loc[1:T[r] - 1] <= 10]
        sustained = int(below.index.max()) + 1 if len(below) else 1
    ew[r] = {"first_roll10_gt10": int(cross.index.min()) if len(cross) else None, "sustained_gt10_from": sustained,
             "peak_roll10_before_step30": float(roll.loc[1:30].max()),
             "step_25pct": T[r], "step_90pct": int(s90.index.min()) if len(s90) else None}
R["early_warning"] = ew
print("early_warning", ew)

json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else (int(o) if isinstance(o, np.integer) else str(o)))
print(summ.round(2).to_string())
print(json.dumps({k: R[k] for k in ["roll10", "difficulty", "context", "analytic_adv", "sites", "forms_after_takeoff"]}, indent=1, default=str))
