"""Experimento del generador oráculo: ¿cuál es el TECHO del aumento sintético?

Idea: el dataset del POC sale de un generador conocido. Podemos muestrear filas
nuevas del proceso generador VERDADERO (otras semillas), que por construcción es
un generador perfecto — detectabilidad original-vs-sintético = 0.5, fidelidad
marginal y conjunta exactas, coherencia de esquema garantizada.

Si ni siquiera ese generador perfecto mejora el desempeño sobre val/test reales,
entonces ningún generador aprendido (CTGAN, TVAE, TabDDPM, TabSyn, ARF…) puede
mejorarlo: todos son aproximaciones peores del mismo proceso.

Variantes probadas:
  A. baseline           — solo train real
  B. replica_r2         — 4x extra al 1.5 % de vishing (imita el diseño de R2)
  C. prevalencia_5      — 4x extra al 5 % (misma prevalencia que el real)
  D. solo_minoria_2x    — solo positivos, duplicando los positivos reales
  E. solo_minoria_5x    — solo positivos, x5
  F. mezcla_pequena     — 0.5x extra al 5 %
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier

import generar_dataset_sintetico as G
from _calibrar import DROP, TARGET, GROUP, split_agrupado, PARAMS

SEEDS_EXTRA = (101, 102, 103)
CSV = "/tmp/gen1/biocatch_sinthetic_data_v3.csv"


def recall_at_p90(y, p):
    pr, rc, _ = precision_recall_curve(y, p)
    ok = pr[:-1] >= 0.90
    return float(rc[:-1][ok].max()) if ok.any() else 0.0


def main():
    base = pd.read_csv(CSV, keep_default_na=False, na_values=[])
    base["split"] = split_agrupado(base)
    feats = [c for c in base.columns
             if c not in DROP and pd.api.types.is_numeric_dtype(base[c])]
    tr = base[base.split == "train"].copy()
    va = base[base.split == "val"]
    te = base[base.split == "test"]
    print("train %d (%d pos, %.2f%%) | val %d | test %d"
          % (len(tr), tr[TARGET].sum(), 100 * tr[TARGET].mean(), len(va), len(te)))

    # ---- pool de filas nuevas del generador VERDADERO -----------------------
    pool = []
    for s in SEEDS_EXTRA:
        d = G.generar_dataset(seed=s)
        d[GROUP] = "ORA%d-" % s + d[GROUP].astype(str)   # clientes nuevos, sin colisión
        pool.append(d)
    pool = pd.concat(pool, ignore_index=True)
    pool_v = pool[pool[TARGET] == 1]
    pool_l = pool[pool[TARGET] == 0]
    print("pool oráculo: %d filas (%d vishing, %d legítimas)"
          % (len(pool), len(pool_v), len(pool_l)))

    n_extra_4x = 4 * len(tr)
    n_extra_05x = len(tr) // 2

    def mezcla(n_total, tasa):
        nv = int(n_total * tasa)
        nl = n_total - nv
        return pd.concat([pool_v.head(nv), pool_l.head(nl)], ignore_index=True)

    VARIANTES = {
        "A. baseline (solo real)":      None,
        "B. replica R2 (4x @ 1.5%)":    mezcla(n_extra_4x, 0.015),
        "C. prevalencia real (4x @ 5%)": mezcla(n_extra_4x, 0.05),
        "D. solo minoría 2x":           pool_v.head(int(tr[TARGET].sum())),
        "E. solo minoría 5x":           pool_v.head(4 * int(tr[TARGET].sum())),
        "F. mezcla pequeña (0.5x @ 5%)": mezcla(n_extra_05x, 0.05),
    }

    Xv, yv = va[feats].to_numpy(), va[TARGET].to_numpy()
    Xt_, yt_ = te[feats].to_numpy(), te[TARGET].to_numpy()

    filas = []
    for nombre, extra in VARIANTES.items():
        d = tr if extra is None else pd.concat([tr, extra], ignore_index=True)
        X, y = d[feats].to_numpy(), d[TARGET].to_numpy()
        m = XGBClassifier(**PARAMS)
        m.set_params(scale_pos_weight=round((y == 0).sum() / max((y == 1).sum(), 1), 2))
        m.fit(X, y)
        pv, pt = m.predict_proba(Xv)[:, 1], m.predict_proba(Xt_)[:, 1]
        filas.append({
            "variante": nombre, "filas": len(d), "pos": int(y.sum()),
            "tasa_pos": round(float(y.mean()), 4),
            "pr_auc_val": round(average_precision_score(yv, pv), 4),
            "pr_auc_test": round(average_precision_score(yt_, pt), 4),
            "roc_auc_test": round(roc_auc_score(yt_, pt), 4),
            "recall_p90_test": round(recall_at_p90(yt_, pt), 4),
        })
        print("  %-30s PR-AUC test %.4f" % (nombre, filas[-1]["pr_auc_test"]))

    res = pd.DataFrame(filas)
    b = res.loc[0, "pr_auc_test"]
    res["delta_vs_baseline"] = (res.pr_auc_test - b).round(4)
    print()
    print(res.to_string(index=False))
    res.to_csv("/tmp/oraculo_aumento.csv", index=False)


if __name__ == "__main__":
    main()
