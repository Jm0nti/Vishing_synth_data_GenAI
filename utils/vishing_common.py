"""
vishing_common.py — Núcleo compartido del pipeline remediado.

Adaptado al dataset sintético **v3** (`biocatch_sinthetic_data_v3.csv`,
100.000 sesiones x 58 columnas, semilla 42), que corrige las tres variables
degeneradas de la generación anterior:

  * `session_duration_s` ya no es la constante 1.0 (sd = 137,6 s).
  * `call_overlap_duration_s` ya no es una copia de `phone_call_active`
    (r = 0,60, es una duración continua condicionada).
  * `time_to_transaction_s` ya no es `10 x transaction_attempted`
    (Gamma con 17.998 valores distintos y centinela -1).

Cambios de esquema respecto de la generación anterior:

  * No existe `device_type`: el dataset es 100 % móvil, así que el filtro por
    canal desaparece (era un no-op que además dejaba una columna constante).
  * No existen `screens_visited`, `unusual_screen_visits` ni `interactions_per_s`:
    se descartaban siempre aguas abajo y ya no se generan.
  * `transaction_amount_cop` y `time_to_transaction_s` usan **-1 como centinela**
    de "no aplica" cuando `transaction_attempted = 0`. Ver `SENTINELAS`.
  * `row_id` viene ya en el CSV como columna real.

Concentra en un solo lugar las decisiones que en la v1 estaban dispersas por los
notebooks y que produjeron fuga de datos:

  * Las rutas de S3 (una sola fuente de verdad).
  * El identificador de fila `row_id`, que sobrevive a todas las
    transformaciones. La v1 dependía del índice de pandas, que se destruía al
    guardar con `index=False`; por eso `drop(index=...)` no eliminaba las filas
    correctas.
  * El contrato de features, calculado SOLO sobre train.
  * El split canónico, agrupado por cliente.
  * Las aserciones anti-fuga, que fallan ruidosamente en vez de en silencio.

Uso desde un notebook de SageMaker:

    import vishing_common as vc
    vc.set_all_seeds()
    train = vc.read_parquet(vc.P.train)
"""

from __future__ import annotations

import io
import json
import os
import platform
import random
import sys
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import boto3
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42

BUCKET = "poc-vishing"
PREFIX = "v2"                      # namespace del pipeline remediado

#: Versión del esquema de datos que espera este módulo. Se registra en el
#: manifiesto y `validate_raw` falla si el CSV no la cumple.
DATASET_VERSION = "v3"

#: Dataset canónico. Decisión explícita del proyecto: se usa SOLO este archivo.
#: Las generaciones anteriores (`biocatch_sinthetic_data.csv` y
#: `dataset_sintetico_biocatch_vishing.csv`) quedan descartadas: tienen
#: variables degeneradas y parámetros distintos, y mezclarlas fue la causa de
#: que la Fase 1 de la v1 no fuera comparable con la Fase 3.
RAW_FILENAME = "biocatch_sinthetic_data_v3.csv"

#: Forma esperada del CSV crudo. `validate_raw` la comprueba antes del split:
#: es más barato fallar aquí que descubrir en R4 que se subió el archivo
#: equivocado.
RAW_EXPECTED = {
    "filas": 100_000,
    "columnas": 58,
    "vishing": 5_000,
    "clientes_min": 30_000,
    "clientes_max": 45_000,
}

#: "grouped" reparte por cliente: ninguna sesión del mismo cliente cae en dos
#: particiones. Con 2,63 sesiones por cliente y el 55 % de la varianza
#: conductual atribuible al cliente, un split por fila deja sesiones del mismo
#: cliente a ambos lados e infla el desempeño.
SPLIT_MODE = "grouped"
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)   # train / val / test

#: Política de features. Ver `build_feature_contract`.
FEATURE_POLICY = "audited"

TARGET = "is_vishing"
ROW_ID = "row_id"
ORIGIN = "origin"          # "original" | "ctgan" | "resampled"
GROUP_COL = "customer_id"

#: Centinelas de "no aplica". NO son ceros ni faltantes: -1 significa que la
#: variable no está definida para esa sesión. Cualquier fila sintética (CTGAN o
#: SMOTE) debe volver a respetarlos, o se acaba con montos interpolados en
#: sesiones sin transacción. Ver `repair_coherence`.
SENTINELAS = {
    "transaction_amount_cop": -1,
    "time_to_transaction_s": -1,
    "days_to_claim": -1,
}

#: Horas consideradas atípicas. Coincide con la regla del generador; se
#: centraliza aquí para que R2 no la vuelva a escribir a mano.
HORAS_ATIPICAS = [22, 23, 0, 1, 2, 3, 4, 5]


# ─────────────────────────────────────────────────────────────────────────────
# 2. RUTAS S3
# ─────────────────────────────────────────────────────────────────────────────

def _uri(*parts: str) -> str:
    return f"s3://{BUCKET}/{PREFIX}/" + "/".join(p.strip("/") for p in parts)


@dataclass(frozen=True)
class Paths:
    # 00 — insumo (lo subes tú, es el único archivo de entrada)
    raw: str = _uri("00_raw", RAW_FILENAME)

    # 01 — contrato y auditoría
    audit_report: str = _uri("01_contract", "data_audit.csv")
    feature_contract: str = _uri("01_contract", "feature_contract.json")
    run_manifest: str = _uri("01_contract", "run_manifest.json")

    # 02 — split canónico (el test no se abre hasta R5)
    split_map: str = _uri("02_splits", "split_map.parquet")
    train: str = _uri("02_splits", "train.parquet")
    val: str = _uri("02_splits", "val.parquet")
    test: str = _uri("02_splits", "test.parquet")

    # 03 — aumento con CTGAN (ajustado SOLO sobre train)
    ctgan_legit: str = _uri("03_augmented", "synthesizers", "ctgan_legit.pkl")
    ctgan_vishing: str = _uri("03_augmented", "synthesizers", "ctgan_vishing.pkl")
    train_augmented: str = _uri("03_augmented", "train_augmented.parquet")
    ks_report: str = _uri("03_augmented", "quality", "ks_report.csv")
    detectability: str = _uri("03_augmented", "quality", "detectability.json")

    # 06 — resultados
    val_leaderboard: str = _uri("06_results", "validation_leaderboard.csv")
    ablation: str = _uri("06_results", "ablation.csv")
    final_test: str = _uri("06_results", "final_test.json")
    automl_leaderboard: str = _uri("06_results", "automl", "leaderboard.csv")
    automl_dir: str = _uri("06_results", "automl")

    # 06 — análisis exploratorio (R1). Se calcula SOLO sobre train.
    eda_separability: str = _uri("06_results", "eda", "separability.csv")
    eda_binaries: str = _uri("06_results", "eda", "binary_odds_ratios.csv")
    eda_correlations: str = _uri("06_results", "eda", "correlations.csv")
    eda_profile: str = _uri("06_results", "eda", "behavioral_profile.csv")
    eda_pca: str = _uri("06_results", "eda", "pca_loadings.csv")
    eda_outliers: str = _uri("06_results", "eda", "outlier_enrichment.csv")
    eda_summary: str = _uri("06_results", "eda", "summary.json")

    # 06 — interpretabilidad (R8)
    feature_importance: str = _uri("06_results", "feature_importance.csv")
    permutation_importance: str = _uri("06_results", "permutation_importance.csv")

    figures: str = _uri("07_figures")

    def figure(self, nombre: str) -> str:
        return _uri("07_figures", nombre)

    # 04 / 05 — plantillas parametrizadas
    def balanced(self, data_type: str, technique: str, ratio) -> str:
        return _uri("04_balanced", data_type, technique, f"{ratio}.parquet")

    def balanced_dir(self, data_type: str) -> str:
        return _uri("04_balanced", data_type)

    def model(self, data_type: str, variant: str, technique: str, ratio) -> str:
        return _uri("05_models", data_type, variant, technique, f"{ratio}.pkl")


P = Paths()


def s3_key(uri: str) -> str:
    """s3://bucket/a/b -> a/b"""
    return uri.split(f"s3://{BUCKET}/", 1)[1]


# ─────────────────────────────────────────────────────────────────────────────
# 3. E/S
# ─────────────────────────────────────────────────────────────────────────────

_S3_CLIENT = None


class _LazyS3:
    """Difiere la creación del cliente hasta el primer uso real.

    Crear el cliente al importar rompe el import en máquinas sin región ni
    credenciales configuradas, lo que impide probar las funciones puras del
    módulo fuera de SageMaker.
    """

    def __getattr__(self, name):
        global _S3_CLIENT
        if _S3_CLIENT is None:
            _S3_CLIENT = boto3.client("s3")
        return getattr(_S3_CLIENT, name)


_s3 = _LazyS3()


def read_csv(uri: str, **kw) -> pd.DataFrame:
    return pd.read_csv(uri, **kw)


def read_raw(uri: str | None = None) -> pd.DataFrame:
    """Carga el CSV crudo v3 respetando su convención de nulos.

    `claim_category` usa la cadena vacía cuando no hubo reclamo, y
    `pd.read_csv` la convertiría en NaN: el dataset parecería tener 95.747
    nulos que en realidad no existen. Se restaura la cadena vacía para que la
    validación de integridad de R0 sea la del archivo, no la del parser.
    """
    df = pd.read_csv(uri or P.raw)
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].fillna("")
    return df


def read_parquet(uri: str, **kw) -> pd.DataFrame:
    return pd.read_parquet(uri, **kw)


def write_parquet(df: pd.DataFrame, uri: str) -> str:
    """Escribe preservando el índice explícitamente descartado a favor de row_id.

    No usamos el índice de pandas como identidad en ningún punto del pipeline:
    `row_id` es una columna real y viaja dentro del archivo.
    """
    if ROW_ID not in df.columns:
        raise ValueError(
            f"'{ROW_ID}' ausente. Toda tabla persistida debe llevar su identificador "
            "de fila; fue justamente su pérdida lo que causó la fuga en la v1."
        )
    df.to_parquet(uri, index=False)
    print(f"  escrito {uri}  ({len(df):,} filas x {df.shape[1]} cols)")
    return uri


def write_json(obj, uri: str) -> str:
    body = json.dumps(obj, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    _s3.put_object(Bucket=BUCKET, Key=s3_key(uri), Body=body)
    print(f"  escrito {uri}")
    return uri


def read_json(uri: str):
    obj = _s3.get_object(Bucket=BUCKET, Key=s3_key(uri))
    return json.loads(obj["Body"].read().decode("utf-8"))


def write_csv(df: pd.DataFrame, uri: str) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    _s3.put_object(Bucket=BUCKET, Key=s3_key(uri), Body=buf.getvalue().encode("utf-8"))
    print(f"  escrito {uri}  ({len(df):,} filas)")
    return uri


def write_pickle(obj, uri: str) -> str:
    import joblib
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    _s3.upload_fileobj(buf, BUCKET, s3_key(uri))
    print(f"  escrito {uri}")
    return uri


def read_pickle(uri: str):
    import joblib
    buf = io.BytesIO()
    _s3.download_fileobj(BUCKET, s3_key(uri), buf)
    buf.seek(0)
    return joblib.load(buf)


def save_figure(fig, uri: str, dpi: int = 150) -> str:
    """Sube una figura de matplotlib a S3 como PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    _s3.upload_fileobj(buf, BUCKET, s3_key(uri))
    print(f"  figura -> {uri}")
    return uri


def list_parquet(prefix_uri: str) -> list[str]:
    """Lista los .parquet bajo un prefijo, ordenados."""
    key = s3_key(prefix_uri).rstrip("/") + "/"
    out, token = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": key}
        if token:
            kw["ContinuationToken"] = token
        resp = _s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            if o["Key"].endswith(".parquet"):
                out.append(f"s3://{BUCKET}/{o['Key']}")
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    return sorted(out)


# ─────────────────────────────────────────────────────────────────────────────
# 4. REPRODUCIBILIDAD
# ─────────────────────────────────────────────────────────────────────────────

def set_all_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Determinismo de las reducciones de cuBLAS: sin esto, dos corridas del
    # mismo notebook en la misma GPU pueden diferir en el último decimal.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


#: Rangos declarados en requirements.txt. `check_versions` avisa si el entorno
#: de SageMaker resolvió algo fuera de rango, que es la causa más frecuente de
#: que una corrida no reproduzca a otra.
VERSIONES_ESPERADAS = {
    "numpy": (">=1.26", "<2.1"),
    "pandas": (">=2.1", "<2.3"),
    "scipy": (">=1.11", "<1.15"),
    "sklearn": (">=1.4", "<1.6"),
    "imblearn": (">=0.12", "<0.13"),
    "xgboost": (">=2.0", "<2.2"),
}


def _tupla(v: str) -> tuple:
    out = []
    for p in str(v).split(".")[:3]:
        num = "".join(c for c in p if c.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out + [0] * (3 - len(out)))


def check_versions(strict: bool = False) -> pd.DataFrame:
    """Compara las versiones instaladas con los rangos de requirements.txt.

    Se llama al arrancar cada notebook. Con `strict=True` levanta la excepción
    en vez de imprimir el aviso, para corridas desatendidas.
    """
    filas = []
    for mod, (lo, hi) in VERSIONES_ESPERADAS.items():
        try:
            m = __import__(mod, fromlist=["__version__"])
            v = getattr(m, "__version__", "?")
        except Exception:
            filas.append({"paquete": mod, "instalada": None, "esperado": f"{lo},{hi}",
                          "ok": False})
            continue
        ok = _tupla(v) >= _tupla(lo.lstrip(">=")) and _tupla(v) < _tupla(hi.lstrip("<"))
        filas.append({"paquete": mod, "instalada": v, "esperado": f"{lo},{hi}", "ok": ok})
    df = pd.DataFrame(filas)
    malas = df[~df.ok]
    if len(malas):
        msg = ("versiones fuera del rango declarado en requirements.txt:\n"
               + malas.to_string(index=False)
               + "\n  pip install -r requirements.txt")
        if strict:
            raise RuntimeError(msg)
        print("AVISO:", msg)
    else:
        print("  versiones OK:", ", ".join(f"{r.paquete} {r.instalada}"
                                           for r in df.itertuples()))
    return df


def xgb_device() -> str:
    """'cuda' si hay GPU utilizable, 'cpu' si no.

    XGBoost >= 2.0 falla con `device='cuda'` en una instancia sin GPU, y R4/R6
    se ejecutan a veces en CPU para una prueba rápida. Detectarlo evita tener
    que editar el notebook según la instancia.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        import subprocess
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return "cuda"
    except Exception:
        return "cpu"


def xgb_base_params() -> dict:
    """Parámetros base comunes a todas las variantes de XGBoost."""
    return dict(tree_method="hist", device=xgb_device(), eval_metric="logloss",
                random_state=SEED, n_jobs=-1)


def env_manifest() -> dict:
    """Versiones de todo lo que puede mover un resultado. Va a S3 en R0."""
    mods = ["numpy", "pandas", "scipy", "sklearn", "imblearn", "xgboost",
            "torch", "sdv", "ctgan", "autogluon.tabular", "pyarrow", "boto3"]
    versions = {}
    for m in mods:
        try:
            mod = __import__(m, fromlist=["__version__"])
            versions[m] = getattr(mod, "__version__", "desconocida")
        except Exception:
            versions[m] = None
    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sin GPU"
    except Exception:
        pass
    return {
        "python": sys.version.split()[0],
        "plataforma": platform.platform(),
        "gpu": gpu,
        "seed": SEED,
        "librerias": versions,
        "config": {
            "bucket": BUCKET, "prefix": PREFIX, "raw": RAW_FILENAME,
            "dataset_version": DATASET_VERSION, "split_mode": SPLIT_MODE,
            "split_fractions": SPLIT_FRACTIONS, "feature_policy": FEATURE_POLICY,
            "xgb_device": xgb_device(),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4bis. VALIDACIÓN DEL CSV CRUDO Y COHERENCIA LÓGICA
# ─────────────────────────────────────────────────────────────────────────────

def validate_raw(df: pd.DataFrame, strict: bool = True) -> dict:
    """Comprueba que el CSV subido es el dataset v3 esperado.

    Falla antes del split, no en R4. Comprueba forma, balance, ausencia de
    nulos, unicidad de `row_id` y las tres variables que estaban degeneradas en
    la generación anterior.
    """
    problemas, aviso = [], []

    if len(df) != RAW_EXPECTED["filas"]:
        problemas.append(f"filas={len(df)}, se esperaban {RAW_EXPECTED['filas']}")
    if df.shape[1] != RAW_EXPECTED["columnas"]:
        aviso.append(f"columnas={df.shape[1]}, se esperaban {RAW_EXPECTED['columnas']}")
    if TARGET not in df.columns:
        problemas.append(f"falta la columna objetivo '{TARGET}'")
    else:
        n1 = int(df[TARGET].sum())
        if n1 != RAW_EXPECTED["vishing"]:
            problemas.append(f"vishing={n1}, se esperaban {RAW_EXPECTED['vishing']}")
    if ROW_ID not in df.columns:
        problemas.append(f"falta '{ROW_ID}' como columna real (no basta el índice)")
    elif not df[ROW_ID].is_unique:
        problemas.append(f"'{ROW_ID}' tiene duplicados")
    if GROUP_COL in df.columns:
        nc = df[GROUP_COL].nunique()
        if not RAW_EXPECTED["clientes_min"] <= nc <= RAW_EXPECTED["clientes_max"]:
            aviso.append(f"clientes={nc:,}, fuera del rango esperado")
    nulos = int(df.isna().sum().sum())
    if nulos:
        aviso.append(f"{nulos} nulos (el v3 no debería tener ninguno)")

    # Las tres variables que la auditoría de la generación anterior marcó
    degeneradas = {}
    if "session_duration_s" in df.columns:
        sd = float(df.session_duration_s.std())
        degeneradas["session_duration_s"] = {"sd": round(sd, 3),
                                             "nunique": int(df.session_duration_s.nunique())}
        if df.session_duration_s.nunique() <= 1:
            problemas.append("session_duration_s sigue siendo constante (bug #1)")
    for a, b in [("call_overlap_duration_s", "phone_call_active"),
                 ("time_to_transaction_s", "transaction_attempted")]:
        if a in df.columns and b in df.columns:
            r = float(df[a].corr(df[b]))
            degeneradas[a] = {"corr_con_" + b: round(r, 4)}
            if abs(r) >= 0.999:
                problemas.append(f"{a} sigue siendo una copia de {b} (r={r:.4f})")

    rep = {"filas": int(len(df)), "columnas": int(df.shape[1]),
           "vishing": int(df[TARGET].sum()) if TARGET in df.columns else None,
           "clientes": int(df[GROUP_COL].nunique()) if GROUP_COL in df.columns else None,
           "nulos": nulos, "no_degeneradas": degeneradas,
           "problemas": problemas, "avisos": aviso, "ok": not problemas}

    for a in aviso:
        print("  AVISO:", a)
    if problemas:
        msg = "el CSV crudo no cumple el contrato v3:\n  - " + "\n  - ".join(problemas)
        if strict:
            raise AssertionError(msg)
        print("ERROR:", msg)
    else:
        print(f"  OK dataset v3: {len(df):,} filas x {df.shape[1]} cols, "
              f"{rep['vishing']:,} vishing, {rep['clientes']:,} clientes, {nulos} nulos")
    return rep


def repair_coherence(d: pd.DataFrame) -> pd.DataFrame:
    """Restaura las reglas lógicas del esquema v3 sobre filas sintéticas.

    Tanto CTGAN como SMOTE producen filas que violan el esquema: un monto
    interpolado en una sesión sin transacción, un solapamiento de llamada sin
    llamada, un máximo de hesitación por debajo del promedio, un tiempo muerto
    mayor que la sesión. Nada de eso aparece en el dataset real, y dejarlo pasar
    contamina el modelo con combinaciones imposibles.

    Se aplica **después** del muestreo y **antes** de persistir.
    """
    d = d.copy()
    tiene = lambda *cs: all(c in d.columns for c in cs)

    if tiene("hour_of_day", "is_atypical_hour"):
        d["is_atypical_hour"] = d["hour_of_day"].round().astype(int).isin(
            HORAS_ATIPICAS).astype(int)

    if tiene("phone_call_active", "call_overlap_duration_s"):
        d.loc[d.phone_call_active == 0, "call_overlap_duration_s"] = 0.0
        d["call_overlap_duration_s"] = d["call_overlap_duration_s"].clip(lower=0)
        if "session_duration_s" in d.columns:
            d["call_overlap_duration_s"] = np.minimum(d["call_overlap_duration_s"],
                                                      d["session_duration_s"])

    if "transaction_attempted" in d.columns:
        sin_tx = d.transaction_attempted == 0
        for c, val in [("transaction_amount_cop", SENTINELAS["transaction_amount_cop"]),
                       ("time_to_transaction_s", SENTINELAS["time_to_transaction_s"]),
                       ("is_new_beneficiary", 0),
                       ("amount_field_corrections", 0),
                       ("beneficiary_field_corrections", 0)]:
            if c in d.columns:
                d.loc[sin_tx, c] = val
        # Con transacción, el centinela no tiene sentido: se lleva al mínimo real
        con_tx = ~sin_tx
        for c, piso in [("transaction_amount_cop", 5_000.0), ("time_to_transaction_s", 1.0)]:
            if c in d.columns:
                d.loc[con_tx, c] = d.loc[con_tx, c].clip(lower=piso)
        if tiene("time_to_transaction_s", "session_duration_s"):
            d.loc[con_tx, "time_to_transaction_s"] = np.minimum(
                d.loc[con_tx, "time_to_transaction_s"],
                0.97 * d.loc[con_tx, "session_duration_s"])

    if "hesitation_count" in d.columns:
        sin_h = d.hesitation_count == 0
        for c in ["avg_hesitation_duration_s", "max_hesitation_duration_s"]:
            if c in d.columns:
                d.loc[sin_h, c] = 0.0
    if tiene("avg_hesitation_duration_s", "max_hesitation_duration_s"):
        d["max_hesitation_duration_s"] = np.maximum(d["max_hesitation_duration_s"],
                                                    d["avg_hesitation_duration_s"])

    if "dead_time_periods" in d.columns:
        sin_dt = d.dead_time_periods == 0
        for c in ["total_dead_time_s", "dead_time_ratio"]:
            if c in d.columns:
                d.loc[sin_dt, c] = 0.0
    if tiene("total_dead_time_s", "session_duration_s"):
        d["total_dead_time_s"] = np.minimum(d["total_dead_time_s"].clip(lower=0),
                                            0.92 * d["session_duration_s"])

    # Derivadas: se recalculan, nunca se generan
    if "session_duration_s" in d.columns:
        dur = d["session_duration_s"].clip(lower=1e-6)
        if "input_error_count" in d.columns:
            d["errors_per_minute"] = d.input_error_count / (dur / 60.0)
        if tiene("hesitation_count", "avg_hesitation_duration_s"):
            d["hesitation_composite"] = (d.hesitation_count
                                         * d.avg_hesitation_duration_s) / dur
        if "total_dead_time_s" in d.columns:
            d["dead_time_ratio"] = (d.total_dead_time_s / dur).clip(0, 1)

    for c in ["keystroke_variability", "segmented_typing_ratio", "avg_touch_pressure",
              "swipe_directional_variance", "data_familiarity_score", "dead_time_ratio"]:
        if c in d.columns:
            d[c] = d[c].clip(0, 1)
    return d


def check_coherence(d: pd.DataFrame, nombre: str = "") -> dict:
    """Cuenta violaciones del esquema. Se usa como aserción tras `repair_coherence`."""
    v = {}
    t = lambda *cs: all(c in d.columns for c in cs)
    if t("phone_call_active", "call_overlap_duration_s"):
        v["overlap_sin_llamada"] = int(((d.phone_call_active == 0)
                                        & (d.call_overlap_duration_s != 0)).sum())
    if t("transaction_attempted", "transaction_amount_cop"):
        v["monto_sin_transaccion"] = int(((d.transaction_attempted == 0)
                                          & (d.transaction_amount_cop != -1)).sum())
    if t("transaction_attempted", "time_to_transaction_s"):
        v["ttt_sin_transaccion"] = int(((d.transaction_attempted == 0)
                                        & (d.time_to_transaction_s != -1)).sum())
    if t("avg_hesitation_duration_s", "max_hesitation_duration_s"):
        v["max_menor_que_avg"] = int((d.max_hesitation_duration_s
                                      < d.avg_hesitation_duration_s).sum())
    if t("total_dead_time_s", "session_duration_s"):
        v["dead_time_mayor_que_sesion"] = int((d.total_dead_time_s
                                               > d.session_duration_s).sum())
    if "dead_time_ratio" in d.columns:
        v["ratio_fuera_de_0_1"] = int((~d.dead_time_ratio.between(0, 1)).sum())
    total = sum(v.values())
    print(f"  coherencia [{nombre}]: {total} violaciones"
          + ("" if total == 0 else " -> " + str({k: n for k, n in v.items() if n})))
    return v


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONTRATO DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

#: Identificadores y metadatos: nunca entran al modelo.
#: `os_type` y `app_version` tienen la MISMA distribución en ambas clases por
#: construcción del generador v3: no son señal de fraude y usarlas solo añadiría
#: ruido con apariencia de información. `device_type` ya no existe.
ID_COLS = ["session_id", "customer_id", "session_timestamp",
           "os_type", "app_version"]

#: Fuga directa: son salidas del propio motor de riesgo que se quiere replicar.
LEAKAGE_COLS = ["biocatch_risk_score", "biocatch_genuine_score",
                "biocatch_ato_indicator", "biocatch_social_eng_indicator",
                "biocatch_bot_indicator"]

#: Post-hoc: solo se conocen después de que el fraude fue reclamado.
POSTHOC_COLS = ["days_to_claim", "claim_category"]

#: Excluidas ya en la v1 por redundancia con `unique_screens_visited`. El
#: generador v3 directamente no las produce, así que la lista queda como red de
#: seguridad por si se vuelve a leer un CSV antiguo.
REDUNDANT_V1 = ["screens_visited", "unusual_screen_visits", "interactions_per_s"]

#: Derivadas aritméticamente de `session_duration_s`.
#:
#: OJO — el motivo para excluirlas CAMBIÓ con el v3. Antes se excluían porque la
#: duración era la constante 1.0 y las tres quedaban distorsionadas. Ahora la
#: duración tiene varianza real y las tres significan lo que su nombre dice; el
#: único argumento que queda para quitarlas es la colinealidad con su numerador
#: (r = 0,71 entre `input_error_count` y `errors_per_minute`, 0,72 entre
#: `total_dead_time_s` y `dead_time_ratio`). Sigue siendo la política "strict",
#: pero ya no es la política "correcta": es una alternativa a contrastar.
DERIVED_FROM_DURATION = ["errors_per_minute", "hesitation_composite", "dead_time_ratio"]

#: Grupos funcionales, para el estudio de ablation (R6).
#:
#: Cambio frente a la versión anterior: `call_overlap_duration_s` pasa de
#: `transaction` a `context`. Es una variable de contexto de sesión (cuánto
#: solapó la llamada), no transaccional; estaba mal clasificada y eso desplazaba
#: el resultado de la ablation a favor de la familia `transaction`.
FEATURE_GROUPS = {
    "keystroke": ["avg_keyhold_ms", "avg_interkey_latency_ms", "typing_speed_cps",
                  "keystroke_variability", "segmented_typing_ratio"],
    "touch": ["avg_touch_pressure", "avg_touch_size_px", "swipe_speed_px_s",
              "swipe_directional_variance", "scroll_speed_avg"],
    "motion": ["device_tilt_angle_mean", "device_tilt_variability",
               "gyro_rotation_rate_mean", "accelerometer_jerk_mean",
               "phone_motion_events"],
    "hesitation": ["avg_hesitation_duration_s", "max_hesitation_duration_s",
                   "hesitation_count"],
    "dead_time": ["total_dead_time_s", "dead_time_ratio", "dead_time_periods"],
    "navigation": ["unique_screens_visited", "navigation_back_count",
                   "screen_transition_time_avg_s"],
    "corrections": ["input_error_count", "input_correction_count",
                    "amount_field_corrections", "beneficiary_field_corrections",
                    "copy_paste_events"],
    "context": ["session_duration_s", "hour_of_day", "is_atypical_hour",
                "phone_call_active", "call_overlap_duration_s",
                "remote_access_tool_detected", "suspicious_app_detected"],
    "transaction": ["transaction_attempted", "transaction_amount_cop",
                    "is_new_beneficiary", "time_to_transaction_s"],
    "derived": ["data_familiarity_score", "doodling_events",
                "errors_per_minute", "hesitation_composite"],
}

#: Familias agregadas para la ablation: qué es "biometría comportamental" y qué
#: es un proxy contextual o transaccional.
FEATURE_FAMILIES = {
    "behavioral": ["keystroke", "touch", "motion", "hesitation", "dead_time",
                   "navigation", "corrections", "derived"],
    "context": ["context"],
    "transaction": ["transaction"],
}


def audit_columns(df: pd.DataFrame, candidate_cols: Sequence[str]) -> pd.DataFrame:
    """Audita las columnas candidatas: constantes, duplicados exactos, rango útil.

    Se ejecuta SOBRE TRAIN. Detectar degeneración mirando el dataset completo
    sería una forma de peeking.
    """
    rows = []
    num = df[list(candidate_cols)].select_dtypes(include=[np.number])
    for c in candidate_cols:
        s = df[c]
        is_num = c in num.columns
        std = float(s.std()) if is_num else np.nan
        rng = float(s.max() - s.min()) if is_num else np.nan
        rows.append({
            "feature": c,
            "dtype": str(s.dtype),
            "nunique": int(s.nunique()),
            "min": float(s.min()) if is_num else None,
            "max": float(s.max()) if is_num else None,
            "std": std,
            "rango": rng,
            "constante": bool(s.nunique() <= 1),
            "binaria": bool(is_num and set(pd.unique(s.dropna())) <= {0, 1}),
            # dispersión relativa: std/rango. Por debajo de 0.02 la variable
            # aporta casi solo ruido numérico aunque tenga varios valores.
            "dispersion_rel": (std / rng) if (is_num and rng and rng > 0) else np.nan,
        })
    audit = pd.DataFrame(rows)

    # Duplicados exactos: la v1 arrastraba call_overlap_duration_s == phone_call_active
    # y time_to_transaction_s == 10 * transaction_attempted sin detectarlos.
    dupe_of = {}
    max_corr, max_corr_con = {}, {}
    cols = [c for c in candidate_cols if c in num.columns]
    corr = num[cols].corr().abs() if len(cols) > 1 else pd.DataFrame()
    for i, a in enumerate(cols):
        if not corr.empty:
            fila = corr.loc[a].drop(index=a)
            if len(fila):
                max_corr[a] = round(float(fila.max()), 4)
                max_corr_con[a] = str(fila.idxmax())
        if a in dupe_of:
            continue
        for b in cols[i + 1:]:
            if b in dupe_of:
                continue
            if not corr.empty and corr.loc[a, b] >= 0.9999:
                dupe_of[b] = a
    audit["duplicado_de"] = audit["feature"].map(dupe_of).fillna("")
    # Margen frente al umbral de duplicado: en el v3 ninguna debería acercarse
    # a 1,0. Ver el par más alto es más informativo que ver una lista vacía.
    audit["corr_max"] = audit["feature"].map(max_corr)
    audit["corr_max_con"] = audit["feature"].map(max_corr_con).fillna("")
    return audit


def build_feature_contract(df_train: pd.DataFrame,
                           policy: str = FEATURE_POLICY) -> dict:
    """Construye la lista de features del modelo y deja constancia de las bajas.

    Políticas
    ---------
    legacy   Todas las candidatas, sin filtrar (44 sobre el esquema v3).
    audited  (por defecto) Quita constantes y duplicados exactos.
    strict   Además quita las derivadas de `session_duration_s` y las de
             dispersión relativa despreciable.

    Sobre el dataset v3 se espera que `audited` NO quite ninguna variable, es
    decir que coincida con `legacy`. Eso no hace la auditoría innecesaria: es
    justamente la evidencia de que el generador v3 corrigió las tres variables
    degeneradas. Si alguna vez vuelve a quitar algo, el CSV no es el que se cree.
    """
    if policy not in {"legacy", "audited", "strict"}:
        raise ValueError(f"política desconocida: {policy}")

    drop_always = set(ID_COLS + LEAKAGE_COLS + POSTHOC_COLS + REDUNDANT_V1
                      + [TARGET, ROW_ID, ORIGIN, "split"])
    candidates = [c for c in df_train.columns if c not in drop_always]

    audit = audit_columns(df_train, candidates)
    removed: dict[str, str] = {}

    if policy != "legacy":
        for _, r in audit.iterrows():
            if r["constante"]:
                removed[r["feature"]] = "constante en train (sin información)"
            elif r["duplicado_de"]:
                removed[r["feature"]] = f"duplicado exacto de {r['duplicado_de']}"

    if policy == "strict":
        for f in DERIVED_FROM_DURATION:
            if f in candidates and f not in removed:
                removed[f] = ("derivada aritmética de session_duration_s "
                              "(colineal con su numerador)")
        for _, r in audit.iterrows():
            f = r["feature"]
            if f in removed or r["binaria"]:
                continue
            if pd.notna(r["dispersion_rel"]) and r["dispersion_rel"] < 0.02:
                removed[f] = f"dispersión relativa despreciable ({r['dispersion_rel']:.4f})"

    features = [c for c in candidates if c not in removed]

    groups = {g: [f for f in cols if f in features]
              for g, cols in FEATURE_GROUPS.items()}
    families = {fam: sorted({f for g in gs for f in groups.get(g, [])})
                for fam, gs in FEATURE_FAMILIES.items()}

    contract = {
        "policy": policy,
        "n_features": len(features),
        "features": features,
        "removed": removed,
        "groups": groups,
        "families": families,
        "target": TARGET,
        "id_col": ROW_ID,
        "seed": SEED,
    }
    return contract


def apply_contract(df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Devuelve X con las columnas del contrato, en el orden exacto del contrato."""
    missing = [f for f in contract["features"] if f not in df.columns]
    if missing:
        raise ValueError(f"faltan features del contrato: {missing}")
    return df[contract["features"]].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 6. SPLIT CANÓNICO
# ─────────────────────────────────────────────────────────────────────────────

def make_split(df: pd.DataFrame,
               mode: str = SPLIT_MODE,
               fractions: tuple = SPLIT_FRACTIONS,
               seed: int = SEED) -> pd.DataFrame:
    """Asigna train/val/test. Devuelve el df con `row_id` y `split`.

    En modo "grouped" el reparto es por cliente y estratificado por si el
    cliente tuvo alguna sesión de vishing. Así ninguna sesión de un mismo
    cliente queda a ambos lados de la frontera.
    """
    from sklearn.model_selection import train_test_split

    df = df.reset_index(drop=True).copy()
    # El CSV v3 ya trae `row_id` como columna real. Se respeta si es válido y
    # solo se crea cuando falta: sobrescribirlo rompería la trazabilidad contra
    # el archivo de origen, que es justo lo que se quería garantizar.
    if ROW_ID in df.columns and df[ROW_ID].is_unique and df[ROW_ID].notna().all():
        df[ROW_ID] = df[ROW_ID].astype(np.int64)
    else:
        df[ROW_ID] = np.arange(len(df), dtype=np.int64)
        print(f"  '{ROW_ID}' ausente o no único: se regenera 0..{len(df) - 1}")

    f_tr, f_va, f_te = fractions
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("las fracciones deben sumar 1")

    if mode == "grouped":
        g = (df.groupby(GROUP_COL)[TARGET].max()
               .rename("cliente_con_vishing").reset_index())
        tr_g, rest_g = train_test_split(
            g, test_size=(f_va + f_te), random_state=seed,
            stratify=g["cliente_con_vishing"])
        rel = f_te / (f_va + f_te)
        va_g, te_g = train_test_split(
            rest_g, test_size=rel, random_state=seed,
            stratify=rest_g["cliente_con_vishing"])
        mapping = {}
        for name, part in (("train", tr_g), ("val", va_g), ("test", te_g)):
            for cid in part[GROUP_COL]:
                mapping[cid] = name
        df["split"] = df[GROUP_COL].map(mapping)
    elif mode == "row":
        idx_tr, idx_rest = train_test_split(
            df.index, test_size=(f_va + f_te), random_state=seed, stratify=df[TARGET])
        rel = f_te / (f_va + f_te)
        idx_va, idx_te = train_test_split(
            idx_rest, test_size=rel, random_state=seed, stratify=df.loc[idx_rest, TARGET])
        df["split"] = "train"
        df.loc[idx_va, "split"] = "val"
        df.loc[idx_te, "split"] = "test"
    else:
        raise ValueError(f"modo de split desconocido: {mode}")

    df[ORIGIN] = "original"
    return df


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("split").agg(
        sesiones=(ROW_ID, "count"),
        vishing=(TARGET, "sum"),
        tasa_vishing=(TARGET, "mean"),
        clientes=(GROUP_COL, "nunique") if GROUP_COL in df.columns else (ROW_ID, "count"),
    )
    return g.reindex(["train", "val", "test"])


# ─────────────────────────────────────────────────────────────────────────────
# 7. GUARDAS ANTI-FUGA
# ─────────────────────────────────────────────────────────────────────────────

def assert_disjoint(train_df: pd.DataFrame,
                    eval_df: pd.DataFrame,
                    nombre: str = "train vs eval") -> None:
    """Falla si alguna fila de evaluación aparece en entrenamiento.

    Comprueba dos cosas, porque el remuestreo puede duplicar filas de
    evaluación sin conservar su row_id:
      1. Intersección de `row_id`.
      2. Intersección de huellas de contenido sobre las columnas numéricas.
    """
    tr_ids = set(train_df[ROW_ID].dropna().astype(np.int64))
    ev_ids = set(eval_df[ROW_ID].dropna().astype(np.int64))
    inter = tr_ids & ev_ids
    if inter:
        raise AssertionError(
            f"FUGA [{nombre}]: {len(inter):,} row_id de evaluación están en "
            f"entrenamiento. Ejemplos: {sorted(inter)[:5]}")

    shared = [c for c in eval_df.columns
              if c in train_df.columns
              and pd.api.types.is_numeric_dtype(eval_df[c])
              and c not in {ROW_ID, TARGET}]
    # Se eligen las columnas de mayor cardinalidad: una huella construida sobre
    # variables binarias o muy cuantizadas colisionaría entre filas distintas y
    # produciría falsas alarmas de fuga.
    key = sorted(shared, key=lambda c: eval_df[c].nunique(), reverse=True)[:12]
    if key:
        def fp(d):
            return set(map(tuple, d[key].round(6).itertuples(index=False, name=None)))
        common = fp(train_df) & fp(eval_df)
        if common:
            raise AssertionError(
                f"FUGA [{nombre}]: {len(common):,} filas de evaluación aparecen en "
                "entrenamiento por contenido (row_id distinto pero misma fila). "
                "Es exactamente el patrón que el random oversampling produjo en la v1.")
    print(f"  OK sin fuga [{nombre}]: {len(tr_ids):,} train / {len(ev_ids):,} eval, disjuntos")


def assert_contract(df: pd.DataFrame, contract: dict, nombre: str = "") -> None:
    missing = [f for f in contract["features"] if f not in df.columns]
    if missing:
        raise AssertionError(f"[{nombre}] faltan features del contrato: {missing}")
    print(f"  OK contrato [{nombre}]: {contract['n_features']} features presentes")


# ─────────────────────────────────────────────────────────────────────────────
# 7bis. ANÁLISIS EXPLORATORIO
#
# Se calcula SOBRE TRAIN. En la v1 el EDA se computó sobre las 50.000 sesiones
# completas —incluyendo escritorio y las filas que después fueron evaluación—,
# de modo que las razones de momios y los AUC univariados que cita el manuscrito
# no comparten base de cálculo con los resultados de modelado.
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(x0: np.ndarray, x1: np.ndarray) -> float:
    """Tamaño del efecto de Cohen entre dos grupos."""
    n0, n1 = len(x0), len(x1)
    if n0 < 2 or n1 < 2:
        return float("nan")
    s = np.sqrt(((n0 - 1) * np.var(x0, ddof=1) + (n1 - 1) * np.var(x1, ddof=1))
                / (n0 + n1 - 2))
    diff = float(np.mean(x1) - np.mean(x0))
    if s > 0:
        return diff / s
    # Varianza intra-grupo nula. Si además las medias coinciden, la variable es
    # constante y el efecto es cero; si difieren, es un separador perfecto y el
    # efecto es infinito. Devolver 0.0 en ese segundo caso lo haría pasar por
    # "despreciable", que es justo lo contrario.
    if diff == 0:
        return 0.0
    return float(np.inf) if diff > 0 else float(-np.inf)


def interpret_d(d: float) -> str:
    a = abs(d)
    if np.isnan(a):
        return "-"
    if a >= 0.8:
        return "grande"
    if a >= 0.5:
        return "medio"
    if a >= 0.2:
        return "pequeño"
    return "despreciable"


def separability_table(df: pd.DataFrame, features: Sequence[str],
                       target: str = TARGET) -> pd.DataFrame:
    """Separabilidad univariada por variable.

    Devuelve U de Mann-Whitney, d de Cohen, correlación biserial puntual y AUC
    univariada (usando la variable como clasificador de una sola dimensión).
    """
    from scipy.stats import mannwhitneyu, pointbiserialr
    from sklearn.metrics import roc_auc_score

    y = df[target].values
    filas = []
    for f in features:
        if not pd.api.types.is_numeric_dtype(df[f]):
            continue
        x = df[f].values.astype(float)
        x0, x1 = x[y == 0], x[y == 1]
        if len(np.unique(x)) <= 1:
            filas.append({"feature": f, "auc": 0.5, "auc_dir": 0.5, "cohens_d": 0.0,
                          "efecto": "despreciable", "p_mannwhitney": 1.0,
                          "r_biserial": 0.0, "mediana_legitima": float(np.median(x0)),
                          "mediana_vishing": float(np.median(x1)), "degenerada": True})
            continue
        try:
            _, p = mannwhitneyu(x0, x1, alternative="two-sided")
        except ValueError:
            p = 1.0
        try:
            r, _ = pointbiserialr(y, x)
        except Exception:
            r = np.nan
        auc = float(roc_auc_score(y, x))
        d = cohens_d(x0, x1)
        filas.append({
            "feature": f,
            "auc": round(auc, 4),
            "auc_dir": round(max(auc, 1 - auc), 4),   # poder discriminante
            "cohens_d": round(d, 4),
            "efecto": interpret_d(d),
            "p_mannwhitney": float(p),
            "r_biserial": round(float(r), 4) if r == r else np.nan,
            "mediana_legitima": float(np.median(x0)),
            "mediana_vishing": float(np.median(x1)),
            "degenerada": False,
        })
    return (pd.DataFrame(filas)
            .sort_values("auc_dir", ascending=False)
            .reset_index(drop=True))


def binary_association_table(df: pd.DataFrame, features: Sequence[str],
                             target: str = TARGET) -> pd.DataFrame:
    """Chi cuadrado y razón de momios para las variables binarias.

    Aplica la corrección de Haldane-Anscombe (+0,5) cuando alguna celda es cero,
    de modo que la razón de momios siga definida.
    """
    from scipy.stats import chi2_contingency

    y = df[target].values
    filas = []
    for f in features:
        vals = set(pd.unique(df[f].dropna()))
        if not vals <= {0, 1}:
            continue
        x = df[f].values
        a = int(((x == 1) & (y == 1)).sum())   # expuesto y positivo
        b = int(((x == 1) & (y == 0)).sum())
        c = int(((x == 0) & (y == 1)).sum())
        d = int(((x == 0) & (y == 0)).sum())
        tabla = np.array([[a, b], [c, d]])
        if tabla.min() == 0:
            aa, bb, cc, dd = a + .5, b + .5, c + .5, d + .5
            corregido = True
        else:
            aa, bb, cc, dd = a, b, c, d
            corregido = False
        odds = (aa * dd) / (bb * cc)
        se = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
        try:
            chi2, p, _, _ = chi2_contingency(tabla)
        except ValueError:
            chi2, p = np.nan, 1.0
        filas.append({
            "feature": f,
            "odds_ratio": round(float(odds), 4),
            "or_ic95_lo": round(float(np.exp(np.log(odds) - 1.96 * se)), 4),
            "or_ic95_hi": round(float(np.exp(np.log(odds) + 1.96 * se)), 4),
            "chi2": round(float(chi2), 2) if chi2 == chi2 else np.nan,
            "p": float(p),
            "significativa_999": bool(p < 0.001),
            "tasa_si_1": round(a / max(a + b, 1), 4),
            "tasa_si_0": round(c / max(c + d, 1), 4),
            "haldane": corregido,
        })
    return (pd.DataFrame(filas)
            .sort_values("odds_ratio", ascending=False)
            .reset_index(drop=True))


def correlation_pairs(df: pd.DataFrame, features: Sequence[str],
                      umbral: float = 0.70) -> pd.DataFrame:
    """Pares de variables con correlación absoluta por encima del umbral."""
    num = df[[f for f in features if pd.api.types.is_numeric_dtype(df[f])]]
    corr = num.corr().abs()
    filas = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if r == r and r >= umbral:
                filas.append({"feature_a": a, "feature_b": b, "r_abs": round(float(r), 4)})
    return (pd.DataFrame(filas).sort_values("r_abs", ascending=False)
            .reset_index(drop=True))


def behavioral_profile(df: pd.DataFrame, features: Sequence[str],
                       target: str = TARGET) -> pd.DataFrame:
    """Mediana por clase, normalizada a [0,1] para el gráfico de radar."""
    filas = []
    for f in features:
        if not pd.api.types.is_numeric_dtype(df[f]):
            continue
        lo, hi = df[f].min(), df[f].max()
        rng = hi - lo
        m0 = df.loc[df[target] == 0, f].median()
        m1 = df.loc[df[target] == 1, f].median()
        filas.append({
            "feature": f,
            "mediana_legitima": float(m0),
            "mediana_vishing": float(m1),
            "norm_legitima": float((m0 - lo) / rng) if rng else 0.0,
            "norm_vishing": float((m1 - lo) / rng) if rng else 0.0,
        })
    return pd.DataFrame(filas)


def outlier_enrichment(df: pd.DataFrame, features: Sequence[str],
                       target: str = TARGET) -> pd.DataFrame:
    """Tasa de vishing entre los valores atípicos por rango intercuartílico.

    Un enriquecimiento muy por encima de la tasa global indica que la cola de
    esa variable concentra fraude.
    """
    base = df[target].mean()
    filas = []
    for f in features:
        if not pd.api.types.is_numeric_dtype(df[f]):
            continue
        q1, q3 = df[f].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        mask = (df[f] < q1 - 1.5 * iqr) | (df[f] > q3 + 1.5 * iqr)
        n = int(mask.sum())
        if n < 30:
            continue
        tasa = float(df.loc[mask, target].mean())
        filas.append({
            "feature": f, "n_outliers": n,
            "tasa_vishing_outliers": round(tasa, 4),
            "tasa_vishing_global": round(float(base), 4),
            "enriquecimiento": round(tasa / base, 2) if base else np.nan,
        })
    return (pd.DataFrame(filas).sort_values("enriquecimiento", ascending=False)
            .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# 8. MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_precision(y_true, y_score, target_precision: float = 0.90) -> dict:
    """Recall máximo alcanzable con precisión >= objetivo, y su umbral.

    Es la métrica con la que la v1 descartó AutoML sin calcularla nunca para su
    propio modelo. Aquí se calcula para todos.
    """
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    ok = prec[:-1] >= target_precision
    if not ok.any():
        return {"recall": 0.0, "threshold": None, "alcanzable": False}
    i = int(np.argmax(np.where(ok, rec[:-1], -1)))
    return {"recall": float(rec[i]), "threshold": float(thr[i]), "alcanzable": True}


def alerts_per_100k(y_true, y_pred) -> dict:
    """Carga operativa: alertas y falsas alertas por cada 100.000 sesiones."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    alerts = int(y_pred.sum())
    false_alerts = int(((y_pred == 1) & (y_true == 0)).sum())
    return {
        "alertas_por_100k": round(alerts / n * 100_000, 1),
        "falsas_alertas_por_100k": round(false_alerts / n * 100_000, 1),
        "vishing_perdido_por_100k": round(
            int(((y_pred == 0) & (y_true == 1)).sum()) / n * 100_000, 1),
    }


def f1_optimal_threshold(y_true, y_score) -> float:
    """Umbral que maximiza F1. DEBE calcularse sobre validation, nunca sobre test."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1))])


def bootstrap_ci(y_true, y_score, threshold: float,
                 n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = SEED) -> dict:
    """IC percentil por bootstrap estratificado para las métricas principales."""
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 recall_score, precision_score, f1_score)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ["pr_auc", "roc_auc", "recall", "precision", "f1"]}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        yt, ys = y_true[idx], y_score[idx]
        yp = (ys >= threshold).astype(int)
        acc["pr_auc"].append(average_precision_score(yt, ys))
        acc["roc_auc"].append(roc_auc_score(yt, ys))
        acc["recall"].append(recall_score(yt, yp, zero_division=0))
        acc["precision"].append(precision_score(yt, yp, zero_division=0))
        acc["f1"].append(f1_score(yt, yp, zero_division=0))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: {"lo": float(np.percentile(v, lo)), "hi": float(np.percentile(v, hi))}
            for k, v in acc.items()}


def full_report(y_true, y_score, threshold: float,
                n_boot: int = 1000, seed: int = SEED) -> dict:
    """Informe completo en un punto de operación dado."""
    from sklearn.metrics import (average_precision_score, roc_auc_score, recall_score,
                                 precision_score, f1_score, confusion_matrix)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    rep = {
        "n": int(len(y_true)),
        "positivos": int(y_true.sum()),
        "tasa_positivos": float(y_true.mean()),
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "recall_at_precision_90": recall_at_precision(y_true, y_score, 0.90),
        "carga_operativa": alerts_per_100k(y_true, y_pred),
        "baseline_pr_auc": float(y_true.mean()),
    }
    if n_boot:
        rep["ic95"] = bootstrap_ci(y_true, y_score, threshold, n_boot=n_boot, seed=seed)
    return rep


def print_report(rep: dict, titulo: str = "") -> None:
    ic = rep.get("ic95", {})

    def band(k):
        return (f"  [{ic[k]['lo']:.4f}, {ic[k]['hi']:.4f}]" if k in ic else "")

    print(f"\n{'=' * 64}")
    if titulo:
        print(f"  {titulo}")
        print("-" * 64)
    print(f"  n = {rep['n']:,}   positivos = {rep['positivos']:,} "
          f"({rep['tasa_positivos'] * 100:.2f}%)   umbral = {rep['threshold']:.4f}")
    print("-" * 64)
    for k in ["pr_auc", "roc_auc", "recall", "precision", "f1"]:
        print(f"  {k:<10}: {rep[k]:.4f}{band(k)}")
    c = rep["confusion"]
    print(f"  matriz    : TN={c['tn']:,}  FP={c['fp']:,}  FN={c['fn']:,}  TP={c['tp']:,}")
    r90 = rep["recall_at_precision_90"]
    print(f"  recall@P=0.90: {r90['recall']:.4f}"
          f"{'' if r90['alcanzable'] else '  (precisión 0.90 inalcanzable)'}")
    op = rep["carga_operativa"]
    print(f"  por 100k sesiones: {op['alertas_por_100k']:.0f} alertas, "
          f"{op['falsas_alertas_por_100k']:.0f} falsas, "
          f"{op['vishing_perdido_por_100k']:.0f} vishing perdido")
    print(f"  PR-AUC de referencia (clasificador aleatorio): {rep['baseline_pr_auc']:.4f}")
    print("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# 9. EMPAQUETADO DE INFERENCIA
# ─────────────────────────────────────────────────────────────────────────────

class VishingModelWrapper:
    """Artefacto único de inferencia.

    Cambios frente a la v1: guarda el contrato completo (no solo la lista de
    nombres), el manifiesto de versiones y la partición sobre la que se fijó el
    umbral, para que quede registrado que no fue el conjunto de test.
    """

    def __init__(self, model, contract: dict, threshold: float, scaler=None,
                 metadata: dict | None = None):
        self.model = model
        self.contract = contract
        self.feature_names = list(contract["features"])
        self.threshold = float(threshold)
        self.scaler = scaler
        self.metadata = metadata or {}
        self.metadata.setdefault("threshold_fitted_on", "validation")

    def _prepare(self, payload):
        if isinstance(payload, str):
            payload = json.loads(payload)
        if isinstance(payload, dict):
            payload = [payload]
        df = pd.DataFrame(payload)
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(f"faltan features requeridas: {missing}")
        X = df[self.feature_names].astype(float).values
        return self.scaler.transform(X) if self.scaler is not None else X

    def predict_proba_raw(self, payload):
        p = self.model.predict_proba(self._prepare(payload))[:, 1]
        out = [{"legitimate": float(1 - x), "vishing": float(x)} for x in p]
        return out[0] if len(out) == 1 else out

    def predict(self, payload):
        p = self.model.predict_proba(self._prepare(payload))[:, 1]
        out = (p >= self.threshold).astype(int).tolist()
        return out[0] if len(out) == 1 else out

    def predict_full(self, payload):
        p = self.model.predict_proba(self._prepare(payload))[:, 1]
        out = [{
            "label": "vishing" if x >= self.threshold else "legitimate",
            "prediction": int(x >= self.threshold),
            "probability_vishing": float(x),
            "probability_legitimate": float(1 - x),
            "threshold_used": self.threshold,
            "threshold_fitted_on": self.metadata.get("threshold_fitted_on"),
        } for x in p]
        return out[0] if len(out) == 1 else out

    def __repr__(self):
        m = self.metadata
        return (f"VishingModelWrapper({m.get('variant', '?')}/{m.get('technique', '?')}"
                f"/{m.get('ratio', '?')}, {len(self.feature_names)} features, "
                f"thr={self.threshold:.4f})")
