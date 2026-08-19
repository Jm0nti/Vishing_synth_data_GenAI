"""Banco de calibración rápido: mide el techo univariado y el PR-AUC
multivariado de un CSV generado, replicando el split agrupado de R0 y la
configuración ganadora de R4 (xgb_regularized sobre train sin balanceo).

Uso:  python _calibrar.py /ruta/al/biocatch_sinthetic_data_v3.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

TARGET = "is_vishing"
GROUP = "customer_id"
ID_COLS = ["session_id", "customer_id", "session_timestamp", "os_type", "app_version"]
LEAK = ["biocatch_risk_score", "biocatch_genuine_score", "biocatch_ato_indicator",
        "biocatch_social_eng_indicator", "biocatch_bot_indicator"]
POSTHOC = ["days_to_claim", "claim_category"]
DROP = set(ID_COLS + LEAK + POSTHOC + [TARGET, "row_id", "origin", "split"])

BEHAVIORAL_HINT = [
    "avg_keyhold_ms", "avg_interkey_latency_ms", "typing_speed_cps",
    "keystroke_variability", "segmented_typing_ratio", "avg_touch_pressure",
    "avg_touch_size_px", "swipe_speed_px_s", "swipe_directional_variance",
    "scroll_speed_avg", "device_tilt_angle_mean", "device_tilt_variability",
    "gyro_rotation_rate_mean", "accelerometer_jerk_mean", "phone_motion_events",
    "avg_hesitation_duration_s", "max_hesitation_duration_s", "hesitation_count",
    "total_dead_time_s", "dead_time_ratio", "dead_time_periods",
    "unique_screens_visited", "navigation_back_count", "screen_transition_time_avg_s",
    "input_error_count", "input_correction_count", "amount_field_corrections",
    "beneficiary_field_corrections", "copy_paste_events", "data_familiarity_score",
    "doodling_events", "errors_per_minute", "hesitation_composite",
]
CONTEXT_HINT = ["session_duration_s", "hour_of_day", "is_atypical_hour",
                "phone_call_active", "call_overlap_duration_s",
                "remote_access_tool_detected", "suspicious_app_detected"]
TRANSACTION_HINT = ["transaction_attempted", "transaction_amount_cop",
                    "is_new_beneficiary", "time_to_transaction_s"]

PARAMS = dict(max_depth=6, learning_rate=0.1, n_estimators=200, reg_alpha=1.0,
              reg_lambda=5.0, min_child_weight=10, gamma=0.3,
              tree_method="hist", device="cpu", eval_metric="logloss",
              random_state=42, n_jobs=-1)


def auc_uni(y, x):
    r = stats.rankdata(x)
    n1 = int(y.sum())
    n0 = len(y) - n1
    a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
    return max(a, 1 - a)


def split_agrupado(df, seed=42, fr=(0.6, 0.2, 0.2)):
    g = df[GROUP].astype(str).unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(g)
    n = len(g)
    a, b = int(n * fr[0]), int(n * (fr[0] + fr[1]))
    m = {}
    for c in g[:a]:
        m[c] = "train"
    for c in g[a:b]:
        m[c] = "val"
    for c in g[b:]:
        m[c] = "test"
    return df[GROUP].astype(str).map(m)


def f1_opt(y, p):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[int(np.nanargmax(f1))])


def recall_at_p90(y, p):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, p)
    ok = pr[:-1] >= 0.90
    return float(rc[:-1][ok].max()) if ok.any() else 0.0


def evaluar(csv, verbose=True):
    df = pd.read_csv(csv, keep_default_na=False, na_values=[])
    df["split"] = split_agrupado(df)
    feats = [c for c in df.columns
             if c not in DROP and pd.api.types.is_numeric_dtype(df[c])]

    y = df[TARGET].to_numpy()
    uni = pd.Series({c: auc_uni(y, df[c].to_numpy(dtype=float)) for c in feats})
    uni = uni.sort_values(ascending=False)

    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    te = df[df.split == "test"]

    def fit_eval(cols):
        m = XGBClassifier(**PARAMS)
        n_neg, n_pos = int((tr[TARGET] == 0).sum()), int((tr[TARGET] == 1).sum())
        m.set_params(scale_pos_weight=round(n_neg / max(n_pos, 1), 2))
        m.fit(tr[cols].to_numpy(), tr[TARGET].to_numpy())
        pv = m.predict_proba(va[cols].to_numpy())[:, 1]
        pt = m.predict_proba(te[cols].to_numpy())[:, 1]
        return {
            "pr_auc_val": average_precision_score(va[TARGET], pv),
            "roc_auc_val": roc_auc_score(va[TARGET], pv),
            "pr_auc_test": average_precision_score(te[TARGET], pt),
            "roc_auc_test": roc_auc_score(te[TARGET], pt),
            "recall_p90_val": recall_at_p90(va[TARGET].to_numpy(), pv),
        }

    beh = [c for c in BEHAVIORAL_HINT if c in feats]
    ctx = [c for c in CONTEXT_HINT if c in feats]
    txn = [c for c in TRANSACTION_HINT if c in feats]

    res = {
        "n_features": len(feats),
        "auc_uni_max": float(uni.iloc[0]),
        "auc_uni_max_var": uni.index[0],
        "n_sobre_080": int((uni > 0.80).sum()),
        "n_sobre_085": int((uni > 0.85).sum()),
        "top10": uni.head(10).round(4).to_dict(),
        "all": fit_eval(feats),
        "behavioral": fit_eval(beh),
        "context_transaction": fit_eval(ctx + txn),
        "all_sin_phone": fit_eval([c for c in feats if c not in
                                   ("phone_call_active", "call_overlap_duration_s")]),
    }
    if verbose:
        print("features:", res["n_features"],
              "| AUC uni max: %.4f (%s)" % (res["auc_uni_max"], res["auc_uni_max_var"]),
              "| >0.80:", res["n_sobre_080"], "| >0.85:", res["n_sobre_085"])
        for k in ["all", "behavioral", "context_transaction", "all_sin_phone"]:
            r = res[k]
            print("  %-20s PR-AUC val %.4f | test %.4f | ROC val %.4f | R@P90 %.4f"
                  % (k, r["pr_auc_val"], r["pr_auc_test"], r["roc_auc_val"],
                     r["recall_p90_val"]))
        print("  top10 univariado:")
        for k, v in res["top10"].items():
            print("     %-32s %.4f" % (k, v))
    return res


if __name__ == "__main__":
    evaluar(sys.argv[1])
