#!/usr/bin/env python3
"""Ejecuta los notebooks del pipeline contra el sistema de archivos local.

NO forma parte del pipeline: es el arnés de verificación que se usó para
comprobar que los notebooks corren de principio a fin contra el dataset v3
antes de subirlos a SageMaker. Sustituye la E/S de S3 por un directorio local
y ejecuta las celdas de código de cada .ipynb en orden.

    python _smoke_local.py --csv data/biocatch_sinthetic_data_v3.csv \
                           --out /tmp/smoke --notebooks R0 R1 R3 R4 R5 R6 R8

R2 (CTGAN) y R7 (AutoGluon) se omiten por defecto: necesitan GPU y paquetes
pesados que no están en este entorno.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import nbformat  # noqa: E402
import pandas as pd  # noqa: E402


def instalar_stub_s3(vc, raiz: Path, csv_local: Path):
    """Redirige toda la E/S de vishing_common a `raiz`."""
    raiz.mkdir(parents=True, exist_ok=True)

    def local(uri: str) -> Path:
        p = raiz / uri.replace(f"s3://{vc.BUCKET}/", "")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def read_csv(uri, **kw):
        return pd.read_csv(csv_local if uri == vc.P.raw else local(uri), **kw)

    def read_raw(uri=None):
        df = pd.read_csv(csv_local)
        for c in df.select_dtypes(include=["object"]).columns:
            df[c] = df[c].fillna("")
        return df

    def write_parquet(df, uri):
        if vc.ROW_ID not in df.columns:
            raise ValueError("falta row_id")
        df.to_parquet(local(uri), index=False)
        print(f"  escrito {uri}  ({len(df):,} filas x {df.shape[1]} cols)")
        return uri

    def write_json(obj, uri):
        local(uri).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                              encoding="utf-8")
        print(f"  escrito {uri}")
        return uri

    def write_csv_(df, uri):
        df.to_csv(local(uri), index=False)
        print(f"  escrito {uri}  ({len(df):,} filas)")
        return uri

    def write_pickle(obj, uri):
        import joblib
        joblib.dump(obj, local(uri))
        return uri

    def list_parquet(prefix_uri):
        d = local(prefix_uri.rstrip("/") + "/x").parent
        return sorted(f"s3://{vc.BUCKET}/" + str(p.relative_to(raiz)).replace(os.sep, "/")
                      for p in d.rglob("*.parquet"))

    vc.read_csv = read_csv
    vc.read_raw = read_raw
    vc.read_parquet = lambda uri, **kw: pd.read_parquet(local(uri), **kw)
    vc.write_parquet = write_parquet
    vc.write_json = write_json
    vc.read_json = lambda uri: json.loads(local(uri).read_text(encoding="utf-8"))
    vc.write_csv = write_csv_
    vc.write_pickle = write_pickle
    vc.read_pickle = lambda uri: __import__("joblib").load(local(uri))
    vc.save_figure = lambda fig, uri, dpi=150: (fig.savefig(local(uri), dpi=80,
                                                            bbox_inches="tight"), uri)[1]
    vc.list_parquet = list_parquet
    return local


def recortar(entorno: dict) -> None:
    """Modo rápido: deja una variante por familia para que R4 corra en CPU."""
    for clave, n in [("VARIANTES_XGB", 1), ("OTRAS_FAMILIAS", 1), ("VARIANTES", 2)]:
        d = entorno.get(clave)
        if isinstance(d, dict) and len(d) > n:
            entorno[clave] = dict(list(d.items())[:n])
            print(f"    [rápido] {clave} recortado a {n}")
    if "CONFIGS" in entorno and isinstance(entorno["CONFIGS"], dict):
        pass  # la ablation es barata, se deja completa


def ejecutar(nb_path: Path, entorno: dict, saltar: tuple = (),
             rapido: bool = False) -> None:
    nb = nbformat.read(nb_path, as_version=4)
    codigo = [c for c in nb.cells if c.cell_type == "code"]
    print(f"\n{'=' * 70}\n{nb_path.name}  ({len(codigo)} celdas de código)\n{'=' * 70}")
    for i, cell in enumerate(codigo):
        src = cell.source
        if any(s in src for s in saltar):
            print(f"[{i:02d}] SALTADA (dependencia no disponible)")
            continue
        # Las magias de IPython no existen fuera del kernel
        src = "\n".join(l for l in src.split("\n") if not l.strip().startswith("%pip"))
        try:
            exec(compile(src, f"{nb_path.name}#c{i}", "exec"), entorno)
        except Exception:
            print(f"\n[{i:02d}] FALLÓ:\n{src[:600]}\n")
            raise
        if rapido:
            recortar(entorno)
        print(f"[{i:02d}] ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/biocatch_sinthetic_data_v3.csv")
    ap.add_argument("--out", default="/tmp/smoke")
    ap.add_argument("--notebooks", nargs="+",
                    default=["R0", "R1", "R3", "R4", "R5", "R6", "R8"])
    ap.add_argument("--rapido", action="store_true",
                    help="recorta el barrido de R4 para que corra en CPU")
    args = ap.parse_args()

    aqui = Path(__file__).resolve().parent
    raiz = Path(args.out)
    shutil.rmtree(raiz, ignore_errors=True)
    sys.path.insert(0, str(aqui))

    import vishing_common as vc
    instalar_stub_s3(vc, raiz, Path(args.csv).resolve())

    # `display` solo existe dentro de IPython
    entorno = {"__name__": "__main__",
               "display": lambda *a, **k: [print(str(x)[:400]) for x in a]}

    disponibles = {p.name.split("_")[0]: p for p in sorted(aqui.glob("R*.ipynb"))}
    for clave in args.notebooks:
        nb = disponibles.get(clave)
        if nb is None:
            print(f"(sin notebook para {clave})")
            continue
        ejecutar(nb, entorno, saltar=("import shap", "TabularPredictor"),
                 rapido=args.rapido)

    print("\nTODOS LOS NOTEBOOKS EJECUTADOS SIN ERROR")


if __name__ == "__main__":
    main()
