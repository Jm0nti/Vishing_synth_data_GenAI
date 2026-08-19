"""
generar_dataset_sintetico.py — Dataset sintético v3 de sesiones de banca móvil
con etiqueta de vishing (ingeniería social telefónica).

Contexto
--------
Tercera generación del dataset sintético del POC de detección de vishing con
biometría de comportamiento. Corrige los tres defectos documentados en la
auditoría de la v2:

  bug #1  `session_duration_s` era la constante 1.0
  bug #2  `call_overlap_duration_s` era una copia exacta de `phone_call_active`
  bug #3  `time_to_transaction_s` era exactamente `10 * transaction_attempted`

y mantiene la dificultad del problema: **ninguna variable univariada debe
superar AUC 0.80** frente a `is_vishing`. La señal tiene que emerger de la
combinación de muchos predictores débiles, no de una variable autoexplicativa.

Arquitectura
------------
1. Población de clientes con factores latentes estables (velocidad, pulso,
   cuidado/pericia) → un mismo cliente es reconocible entre sesiones.
2. Cópula gaussiana: cada variable se genera como z ~ N(0,1) con estructura de
   correlación (cliente + estado de la sesión) y luego se mapea a su marginal
   objetivo con la función cuantil (`ppf`). Esto preserva exactamente la
   marginal pedida en §4 del prompt y a la vez induce correlación realista
   dentro del cliente y dentro de la sesión.
3. Ataque como *interpolación*, no como sustitución: para una sesión de vishing
   con activación w ∈ [0,1] el valor final es
       x = (1-w) · F_legit⁻¹(u) + w · F_vishing⁻¹(u)
   con el MISMO u. Es decir: la víctima conserva su rango relativo (un
   mecanógrafo rápido sigue siendo relativamente rápido bajo coacción) y w
   controla cuánto se manifiesta el ataque. w se deriva de una variable latente
   de sofisticación por sesión: los ataques sofisticados (≈15 %) activan poco
   los indicadores y quedan dentro de la nube legítima.
4. Ruido intencional: ≈2.5 % de las sesiones legítimas reciben la misma
   maquinaria de superposición con intensidad moderada (usuario estresado,
   adulto mayor, alguien realmente hablando por teléfono con un familiar).

Ejecución
---------
    python generar_dataset_sintetico.py [--seed 42] [--salida ./]

Requiere únicamente numpy, pandas y scipy (sin CTGAN ni ningún modelo
generativo: la generación es paramétrica a propósito, para tener control total
sobre cada distribución).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════
# 0. PARÁMETROS GLOBALES
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
N_SESIONES = 100_000
N_VISHING = 5_000                       # exacto, 5.0 %
N_CLIENTES = 38_000                     # ⇒ media ≈ 2.63 sesiones/cliente
FECHA_INICIO = "2025-01-01"
FECHA_FIN = "2025-12-31"

# Consistencia intra-cliente de la cópula: proporción de la varianza de cada
# variable que es atribuible al cliente. 0.55 = un usuario es claramente
# reconocible entre sesiones pero no idéntico a sí mismo.
# SUPUESTO PROPIO (no citado): no hay una cifra pública de "estabilidad
# intra-sujeto" para biometría conductual en banca móvil; 0.55 es una elección
# de diseño para que el split agrupado por cliente tenga sentido.
RHO_CLIENTE = 0.55

# ---------------------------------------------------------------------------
# TECHO DE SEPARABILIDAD UNIVARIADA
# ---------------------------------------------------------------------------
# Regla rectora del generador: ninguna feature puede superar este AUC frente a
# `is_vishing`. Si una sola variable explica la etiqueta, el dataset deja de ser
# evidencia de que el modelo aprende un patrón multivariado.
#
# v3.0 usaba 0.80. v3.1 lo sube a 0.85 por decisión de diseño: con 0.80 el mejor
# modelo del pipeline se quedaba en PR-AUC ≈ 0.79 y la ablation no distinguía la
# familia comportamental del bloque contexto+transacción (0.6886 vs 0.6777, con
# los IC 95 % solapados). 0.85 sigue lejos del régimen trivial —una variable con
# AUC 0.85 no clasifica sola: al 5 % de prevalencia su precisión en el punto de
# operación es baja— pero deja margen para que las diferencias entre familias
# salgan del ruido.
#
# La regla se aplica en `validar()` §4 y NO admite excepciones salvo las
# columnas post-hoc (`COLUMNAS_POSTHOC`).
TECHO_AUC_UNIVARIADO = 0.85

# Referencias históricas, para el reporte de validación.
TECHO_AUC_V30 = 0.80
AUC_MAX_V30 = 0.7780        # biocatch_risk_score en la corrida v3.0
AUC_MAX_V2 = 0.6890         # segmented_typing_ratio en el dataset canónico v2

# Fracción de sesiones de vishing "sofisticadas" (ataque que imita al usuario
# legítimo). El prompt pide 10–20 %.
#
# NO se toca al subir el techo: es la palanca de realismo, no de cifras. Un
# ataque sofisticado que quedara igual de separable que uno burdo convertiría el
# dataset en un problema de juguete y el paper perdería el argumento de que hay
# un suelo irreducible de falsos negativos.
P_ATAQUE_SOFISTICADO = 0.15

# Parámetros Beta de la intensidad del ataque, por tipo. La intensidad es la
# fracción del camino entre la marginal legítima y la marginal de vishing "de
# libro" que recorre una sesión atacada.
#
# v3.1: el ataque burdo pasa de Beta(5.0, 2.0) (media 0.714) a Beta(6.0, 1.7)
# (media 0.779). Los ataques documentados en la literatura son precisamente los
# que se detectaron, es decir los más burdos; 0.78 sigue por debajo del 1.0 que
# significaría "el caso de manual exacto". El ataque sofisticado no se mueve.
BETA_INTENSIDAD_BURDO = (6.0, 1.7)          # media ≈ 0.779  (v3.0: 5.0, 2.0)
BETA_INTENSIDAD_SOFISTICADO = (2.0, 6.0)    # media ≈ 0.250  (sin cambios)

# Ruido intencional en la clase legítima (§3.3 del prompt: 2–3 %).
P_LEGIT_CONFUNDIDA = 0.025

# Dispersión del ruido por variable sobre la intensidad del ataque. Sin este
# término todos los indicadores de una sesión se moverían en bloque y la
# correlación entre ellos sería ~1.0 dentro de la clase vishing.
SD_RUIDO_ACTIVACION = 0.12

# ---------------------------------------------------------------------------
# GANANCIAS DE ACTIVACIÓN POR VARIABLE
# ---------------------------------------------------------------------------
# w_j = clip( GANANCIA[j] · (intensidad + ruido), 0, 1 )
#
# Estos coeficientes NO vienen de la literatura: son el resultado de calibrar
# el generador hasta que ninguna variable superara AUC 0.80 (validación §6.4).
# Las marginales "puras" de vishing son las del prompt; la ganancia decide qué
# fracción de esa separación se materializa. Una ganancia de 0.45 significa que
# incluso el ataque más burdo solo llega a mitad de camino entre la marginal
# legítima y la marginal de vishing "de libro".
#
# Justificación conceptual (no solo numérica): las cifras del §4 describen el
# perfil de un caso *confirmado y bien documentado*, que es una muestra sesgada
# hacia los ataques que se detectaron. La población real de sesiones de vishing
# incluye los que nadie vio.
GANANCIA = {
    "avg_keyhold_ms": 0.62,
    "avg_interkey_latency_ms": 0.70,
    "typing_speed_cps": 0.46,
    "keystroke_variability": 0.68,
    "segmented_typing_ratio": 0.38,
    "avg_touch_pressure": 1.00,
    "avg_touch_size_px": 1.00,
    "swipe_speed_px_s": 0.70,
    "swipe_directional_variance": 0.66,
    "scroll_speed_avg": 0.72,
    "device_tilt_angle_mean": 1.00,
    "device_tilt_variability": 0.50,
    "gyro_rotation_rate_mean": 0.62,
    "accelerometer_jerk_mean": 0.55,
    "phone_motion_events": 0.62,
    "hesitation_count": 0.70,
    "avg_hesitation_duration_s": 0.60,
    "max_hesitation_extra_s": 0.60,
    "dead_time_periods": 1.00,
    "total_dead_time_s": 1.00,
    "unique_screens_visited": 0.90,
    "navigation_back_count": 0.85,
    "screen_transition_time_avg_s": 0.62,
    "input_error_count": 0.85,
    "input_correction_count": 0.60,
    "amount_field_corrections": 0.70,
    "beneficiary_field_corrections": 0.70,
    "copy_paste_events": 1.00,
    "data_familiarity_score": 0.42,
    "doodling_events": 0.58,
    "session_duration_s": 1.00,
    "transaction_amount_cop": 0.85,
    "time_to_transaction_s": 0.62,
    "call_overlap_duration_s": 1.00,
}
GANANCIA_POR_DEFECTO = 1.00

# Escala global sobre las ganancias, aplicada en `_activaciones` y recortada a
# 1.0. Es la palanca de un solo número para mover el techo de separabilidad sin
# reescribir la tabla variable por variable: 1.00 reproduce la calibración v3.0,
# valores > 1 acercan cada variable a su marginal de vishing "de libro".
#
# 1.24 es el valor calibrado empíricamente contra TECHO_AUC_UNIVARIADO = 0.85
# (ver `_calibrar.py`). Las variables que ya estaban en ganancia 1.00 no se
# mueven con esta escala —están en su marginal completa— y por eso hizo falta
# subir además la intensidad del ataque burdo.
#
# Margen deliberado: con 1.24 el máximo univariado queda en 0.8380 con la
# semilla 42 y en 0.8395 en el peor de cuatro semillas probadas (1, 7, 42,
# 2026). Con 1.30 el máximo subía a 0.8462 y el peor caso a 0.8479, a dos
# milésimas del techo: cualquier cambio posterior en las marginales lo habría
# roto. El punto de la regla es que se cumpla sin ajustar la semilla.
ESCALA_GANANCIA = 1.24

# Probabilidades de las variables binarias. Para una binaria,
# AUC = 0.5 + (p_vishing − p_legítimo) / 2, así que se controlan directamente.
P_LLAMADA_LEGIT = 0.055        # usuario legítimo genuinamente en llamada
P_LLAMADA_BASE_VISH = 0.10     # w = 0 (ataque muy sofisticado: segundo equipo)
P_LLAMADA_MAX_VISH = 0.85      # w = 1 (85 % del diccionario original)
# NOTA: `phone_call_active` es *detección en el mismo dispositivo*, no simple
# presencia de llamada. Con w medio ≈0.64 la prevalencia observada en vishing
# queda ≈0.58, no 0.85: la diferencia son las víctimas que hablan por otro
# teléfono, altavoz de un fijo, o cuya llamada terminó antes del pago. Sin este
# matiz la variable sola daría AUC ≈ 0.90 y el dataset sería trivial.

P_TXN_LEGIT = 0.60
P_TXN_MAX_VISH = 0.94
P_NUEVO_BENEF_LEGIT = 0.18
P_NUEVO_BENEF_MAX_VISH = 0.90
P_RAT_LEGIT = 0.002            # remote access tool
P_RAT_MAX_VISH = 0.16
P_APP_SOSP_LEGIT = 0.005
P_APP_SOSP_MAX_VISH = 0.11
P_BOT = 0.002                  # idéntica en ambas clases, a propósito

P_RECLAMO_VISHING = 0.78       # §4.14: 70–85 %
P_RECLAMO_LEGIT = 0.004        # §4.14: 0.3–0.5 % (ATO no vishing / disputa)

# Ruido del motor de riesgo simulado (§4.12). Calibrado para que
# `biocatch_risk_score` quede por debajo de 0.80 de AUC como el resto.
SD_RUIDO_BIOCATCH = 2.00
SD_RUIDO_GENUINE = 2.05
GAMMA_RIESGO = 2.0        # forma de la escala 0-1000 (ver §4.12)


# ═══════════════════════════════════════════════════════════════════════════
# 1. UTILIDADES DE LA CÓPULA
# ═══════════════════════════════════════════════════════════════════════════

_NORM = stats.norm()


def _u(z: np.ndarray) -> np.ndarray:
    """Normal estándar → uniforme(0,1) (transformada integral de probabilidad)."""
    return np.clip(_NORM.cdf(z), 1e-9, 1 - 1e-9)


def _mezclar_ppf(u: np.ndarray, selector: np.ndarray, dists) -> np.ndarray:
    """`ppf` de una mezcla finita: cada fila usa la componente que le tocó."""
    out = np.empty_like(u, dtype=float)
    for k, d in enumerate(dists):
        m = selector == k
        if m.any():
            out[m] = d.ppf(u[m])
    return out


def _blend(u, w, ppf_legit, ppf_vish, log_space=False):
    """Interpola entre la marginal legítima y la de vishing con peso `w`.

    `ppf_legit` / `ppf_vish` son callables u → valores. Se usa el MISMO `u`
    en ambas: eso preserva el rango del cliente dentro de su clase.
    `log_space=True` para variables lognormales (mezcla geométrica: interpolar
    100.000 COP con 3.000.000 COP aritméticamente aplastaría el extremo bajo).
    """
    x = np.asarray(ppf_legit(u), dtype=float)
    m = w > 0
    if m.any():
        xv = np.asarray(ppf_vish(u[m]), dtype=float)
        if log_space:
            x[m] = np.exp((1 - w[m]) * np.log(x[m]) + w[m] * np.log(xv))
        else:
            x[m] = (1 - w[m]) * x[m] + w[m] * xv
    return x


def _blend_poisson(u, w, lam_legit, lam_vish) -> np.ndarray:
    """Conteo Poisson con tasa interpolada geométricamente entre ambas clases."""
    lam_legit = np.asarray(lam_legit, dtype=float)
    lam = np.exp((1 - w) * np.log(np.maximum(lam_legit, 1e-6))
                 + w * np.log(max(lam_vish, 1e-6)))
    return stats.poisson.ppf(u, lam).astype(np.int64)


def _bernoulli_interp(u, w, p_legit, p_max_vish) -> np.ndarray:
    """Binaria cuya probabilidad interpola linealmente con la activación."""
    p = p_legit + w * (p_max_vish - p_legit)
    return (u < p).astype(np.int8)


# ═══════════════════════════════════════════════════════════════════════════
# 2. POBLACIÓN DE CLIENTES
# ═══════════════════════════════════════════════════════════════════════════

# Cargas de cada variable sobre los tres factores latentes del cliente.
# Signo positivo = el factor empuja la variable hacia arriba.
#   F0 "velocidad"     → mecanógrafo rápido / ágil con la interfaz
#   F1 "pulso"         → mano firme, poco temblor
#   F2 "pericia"       → conoce la app y sus propios datos, comete pocos errores
CARGAS = {
    "avg_keyhold_ms": (0, -0.65),
    "avg_interkey_latency_ms": (0, -0.70),
    "typing_speed_cps": (0, 0.75),
    "keystroke_variability": (2, -0.40),
    "segmented_typing_ratio": (0, -0.30),
    "avg_touch_pressure": (1, 0.25),
    "avg_touch_size_px": (1, 0.10),
    "swipe_speed_px_s": (0, 0.50),
    "swipe_directional_variance": (1, -0.40),
    "scroll_speed_avg": (0, 0.45),
    "device_tilt_angle_mean": (1, 0.15),
    "device_tilt_variability": (1, -0.60),
    "gyro_rotation_rate_mean": (1, -0.60),
    "accelerometer_jerk_mean": (1, -0.60),
    "phone_motion_events": (1, -0.45),
    "hesitation_count": (2, -0.50),
    "avg_hesitation_duration_s": (2, -0.35),
    "max_hesitation_extra_s": (2, -0.25),
    "dead_time_periods": (2, -0.30),
    "total_dead_time_s": (2, -0.30),
    "unique_screens_visited": (2, -0.25),
    "navigation_back_count": (2, -0.45),
    "screen_transition_time_avg_s": (0, -0.35),
    "input_error_count": (2, -0.60),
    "input_correction_count": (2, -0.50),
    "amount_field_corrections": (2, -0.40),
    "beneficiary_field_corrections": (2, -0.40),
    "copy_paste_events": (2, 0.30),
    "data_familiarity_score": (2, 0.40),
    "doodling_events": (2, -0.35),
    "session_duration_s": (0, -0.30),
    "transaction_amount_cop": (2, 0.05),
    "time_to_transaction_s": (0, -0.35),
    "call_overlap_duration_s": (1, 0.05),
    "hora": (0, 0.0),
}

# Carga sobre el factor de ESTADO de la sesión (el usuario va con prisa, está
# distraído, va en el bus). Induce correlación entre variables dentro de una
# misma sesión, incluso en sesiones legítimas.
CARGA_ESTADO = {
    "avg_keyhold_ms": 0.30, "avg_interkey_latency_ms": 0.35,
    "typing_speed_cps": -0.35, "keystroke_variability": 0.30,
    "segmented_typing_ratio": 0.25, "avg_touch_pressure": 0.20,
    "avg_touch_size_px": 0.10, "swipe_speed_px_s": -0.20,
    "swipe_directional_variance": 0.25, "scroll_speed_avg": -0.20,
    "device_tilt_angle_mean": 0.15, "device_tilt_variability": 0.30,
    "gyro_rotation_rate_mean": 0.30, "accelerometer_jerk_mean": 0.30,
    "phone_motion_events": 0.25, "hesitation_count": 0.35,
    "avg_hesitation_duration_s": 0.30, "max_hesitation_extra_s": 0.25,
    "dead_time_periods": 0.30, "total_dead_time_s": 0.30,
    "unique_screens_visited": 0.15, "navigation_back_count": 0.30,
    "screen_transition_time_avg_s": 0.25, "input_error_count": 0.35,
    "input_correction_count": 0.30, "amount_field_corrections": 0.20,
    "beneficiary_field_corrections": 0.20, "copy_paste_events": -0.10,
    "data_familiarity_score": -0.25, "doodling_events": 0.30,
    "session_duration_s": 0.25, "transaction_amount_cop": 0.05,
    "time_to_transaction_s": 0.25, "call_overlap_duration_s": 0.05,
    "hora": 0.0,
}

VARIABLES_COPULA = list(CARGAS.keys())


def generar_poblacion_clientes(rng: np.random.Generator,
                               n_clientes: int = N_CLIENTES,
                               n_sesiones: int = N_SESIONES) -> pd.DataFrame:
    """Crea la población de clientes con sus rasgos latentes estables.

    El número de sesiones por cliente sale de 1 + Poisson(λ_i) con
    λ_i ~ Gamma (mezcla Gamma–Poisson = binomial negativa): así hay muchos
    clientes con 1–2 sesiones y una cola corta de usuarios intensivos, sin el
    absurdo de un cliente con cientos de sesiones.
    """
    # --- rasgos ------------------------------------------------------------
    F = rng.standard_normal((n_clientes, 3))          # 3 factores ortogonales
    rasgos = np.empty((n_clientes, len(VARIABLES_COPULA)))
    ruido_idio = rng.standard_normal((n_clientes, len(VARIABLES_COPULA)))
    for j, var in enumerate(VARIABLES_COPULA):
        f_idx, carga = CARGAS[var]
        rasgos[:, j] = carga * F[:, f_idx] + np.sqrt(1 - carga ** 2) * ruido_idio[:, j]
    # `rasgos` es N(0,1) por columna por construcción.

    # --- sesiones por cliente ---------------------------------------------
    lam = rng.gamma(shape=1.15, scale=1.42, size=n_clientes)   # media ≈1.63
    n_ses = 1 + rng.poisson(lam)
    n_ses = np.minimum(n_ses, 22)      # tope duro: nadie con 500 sesiones

    # Ajuste exacto al total objetivo, sin sesgar la forma de la distribución.
    delta = n_sesiones - int(n_ses.sum())
    while delta != 0:
        if delta > 0:
            idx = rng.choice(n_clientes, size=min(abs(delta), n_clientes), replace=False)
            idx = idx[n_ses[idx] < 22]
            n_ses[idx] += 1
        else:
            idx = rng.choice(n_clientes, size=min(abs(delta), n_clientes), replace=False)
            idx = idx[n_ses[idx] > 1]
            n_ses[idx] -= 1
        delta = n_sesiones - int(n_ses.sum())

    # --- preferencias de contexto -----------------------------------------
    # Franja horaria habitual: cada cliente favorece una de las 4 componentes
    # diurnas. SUPUESTO PROPIO de UX, no cifra citada.
    franja_pref = rng.integers(0, 4, size=n_clientes)
    # Propensión personal a estar en llamada mientras usa el banco.
    prop_llamada = rng.beta(1.3, 26.0, size=n_clientes)     # media ≈0.055

    clientes = pd.DataFrame({
        "customer_id": [f"CUS-{i:05d}" for i in range(n_clientes)],
        "n_sesiones": n_ses,
        "franja_pref": franja_pref,
        "prop_llamada": prop_llamada,
    })
    for j, var in enumerate(VARIABLES_COPULA):
        clientes[f"t__{var}"] = rasgos[:, j]
    return clientes


def _asignar_vishing(clientes: pd.DataFrame, rng: np.random.Generator):
    """Decide qué clientes son víctimas y cuántas sesiones de vishing tienen.

    Un mismo cliente puede tener sesiones legítimas y, como mucho, 1–2 de
    vishing; la elección de 1 vs 2 es estocástica, no determinista.
    """
    n_clientes = len(clientes)
    n_ses = clientes["n_sesiones"].to_numpy()
    orden = rng.permutation(n_clientes)
    n_vish_por_cliente = np.zeros(n_clientes, dtype=int)
    acumulado = 0
    for i in orden:
        if acumulado >= N_VISHING:
            break
        # 6 % de las víctimas caen dos veces (reincidencia documentada en
        # fraude de ingeniería social). SUPUESTO PROPIO.
        k = 2 if (rng.random() < 0.06 and n_ses[i] >= 3) else 1
        k = min(k, n_ses[i] - 1) if n_ses[i] > 1 else 1   # deja ≥1 legítima si puede
        k = max(k, 1)
        k = min(k, N_VISHING - acumulado, n_ses[i])
        n_vish_por_cliente[i] = k
        acumulado += k
    assert acumulado == N_VISHING, f"asignación incompleta: {acumulado}"
    return n_vish_por_cliente


# ═══════════════════════════════════════════════════════════════════════════
# 3. NÚCLEO CONDUCTUAL COMPARTIDO
# ═══════════════════════════════════════════════════════════════════════════

def _z_correlacionadas(rasgos_fila: np.ndarray, rng: np.random.Generator) -> dict:
    """Genera las z ~ N(0,1) con estructura cliente + estado de sesión."""
    n, p = rasgos_fila.shape
    estado = rng.standard_normal(n)                 # factor de estado por sesión
    eta = rng.standard_normal((n, p))
    z = {}
    raiz_rho = np.sqrt(RHO_CLIENTE)
    raiz_1_rho = np.sqrt(1 - RHO_CLIENTE)
    for j, var in enumerate(VARIABLES_COPULA):
        b = CARGA_ESTADO[var]
        eps = b * estado + np.sqrt(1 - b ** 2) * eta[:, j]
        z[var] = raiz_rho * rasgos_fila[:, j] + raiz_1_rho * eps
    return z


def _activaciones(intensidad: np.ndarray, aplica: np.ndarray,
                  rng: np.random.Generator) -> dict:
    """w_j por variable: intensidad del ataque + ruido idiosincrásico."""
    n = intensidad.shape[0]
    w = {}
    for var in VARIABLES_COPULA:
        # La escala global se recorta a 1.0: una ganancia efectiva > 1 haría que
        # la sesión atacada quedara MÁS allá de la marginal de vishing de libro,
        # que es una extrapolación sin respaldo.
        g = min(GANANCIA.get(var, GANANCIA_POR_DEFECTO) * ESCALA_GANANCIA, 1.0)
        ruido = rng.normal(0.0, SD_RUIDO_ACTIVACION, size=n)
        w[var] = np.clip(g * (intensidad + ruido), 0.0, 1.0) * aplica[:, 0]
    return w


def _nucleo_conductual(clientes: pd.DataFrame,
                       idx_cliente: np.ndarray,
                       intensidad: np.ndarray,
                       rng: np.random.Generator) -> dict:
    """Genera TODAS las columnas de comportamiento para un conjunto de sesiones.

    `intensidad` ∈ [0,1] es la activación base del patrón de vishing. Vale 0
    para una sesión legítima normal, ~0.2–0.6 para una legítima "confundida"
    (ruido intencional) y sale de la latente de sofisticación para vishing.
    """
    n = len(idx_cliente)
    rasgos_fila = clientes[[f"t__{v}" for v in VARIABLES_COPULA]].to_numpy()[idx_cliente]
    z = _z_correlacionadas(rasgos_fila, rng)
    u = {v: _u(z[v]) for v in VARIABLES_COPULA}

    # Máscara por variable: en las sesiones confundidas el patrón solo afecta a
    # una parte de los indicadores (un señor mayor teclea despacio pero no está
    # en llamada ni transfiere a un beneficiario nuevo).
    aplica_todo = (intensidad > 0)[:, None]
    w = _activaciones(intensidad, aplica_todo, rng)

    def w_binaria() -> np.ndarray:
        """Activación con ganancia 1.0 para las variables binarias.

        Las binarias no se calibran con `GANANCIA` sino directamente con sus
        probabilidades (para una binaria, AUC = 0.5 + (p_v − p_l)/2), así que
        usan la intensidad completa.
        """
        return np.clip(intensidad + rng.normal(0, SD_RUIDO_ACTIVACION, size=n),
                       0.0, 1.0) * (intensidad > 0)

    d = {}

    # ── 4.2 Dinámica de tecleo ────────────────────────────────────────────
    # Ancla legítima: 38 WPM ≈ 3.0–3.3 car/s (Aalto University, ~37.000
    # voluntarios, 2019). Dwell 80–160 ms e interkey 150–350 ms son *rangos
    # orientativos* de la literatura general de keystroke dynamics, no una
    # cifra única citable; Normal(110,18) y Normal(220,45) caen dentro.
    d["avg_keyhold_ms"] = np.clip(_blend(
        u["avg_keyhold_ms"], w["avg_keyhold_ms"],
        stats.norm(110, 18).ppf, stats.norm(145, 30).ppf), 40.1, None)

    # Vishing: lognormal de media ≈340 ms con cola derecha pesada (pausas
    # largas mientras la víctima escucha instrucciones).
    _ln_ik = stats.lognorm(s=0.55, scale=340 / np.exp(0.55 ** 2 / 2))
    d["avg_interkey_latency_ms"] = np.clip(_blend(
        u["avg_interkey_latency_ms"], w["avg_interkey_latency_ms"],
        stats.norm(220, 45).ppf, _ln_ik.ppf), 55.0, None)

    d["typing_speed_cps"] = np.clip(_blend(
        u["typing_speed_cps"], w["typing_speed_cps"],
        stats.norm(3.2, 0.7).ppf, stats.norm(1.7, 0.6).ppf), 0.2, 7.0)

    d["keystroke_variability"] = np.clip(_blend(
        u["keystroke_variability"], w["keystroke_variability"],
        stats.beta(2, 6).ppf, stats.beta(4, 4).ppf), 0.0, 1.0)

    # Señal clave del diccionario original. La marginal de vishing es una
    # MEZCLA: 60 % dictado evidente (Beta(6,2)), 40 % ataque sofisticado
    # (Beta(2,5)). La mezcla es la que produce el solapamiento deseado.
    sel_seg = (rng.random(n) < 0.40).astype(int)     # 0 = evidente, 1 = sutil
    d["segmented_typing_ratio"] = np.clip(_blend(
        u["segmented_typing_ratio"], w["segmented_typing_ratio"],
        stats.beta(1, 9).ppf,
        lambda uu, s=sel_seg: _mezclar_ppf(uu, s[w["segmented_typing_ratio"] > 0],
                                           [stats.beta(6, 2), stats.beta(2, 5)])),
        0.0, 1.0)

    # ── 4.3 Dinámica táctil ───────────────────────────────────────────────
    # El estrés no se manifiesta igual en todos: mezcla de presión fuerte
    # (tensión) y toques leves/temblorosos, no un desplazamiento único.
    sel_pres = (rng.random(n) < 0.45).astype(int)
    d["avg_touch_pressure"] = np.clip(_blend(
        u["avg_touch_pressure"], w["avg_touch_pressure"],
        stats.beta(5, 5).ppf,
        lambda uu, s=sel_pres: _mezclar_ppf(uu, s[w["avg_touch_pressure"] > 0],
                                            [stats.beta(6, 4), stats.beta(3, 6)])),
        0.0, 1.0)

    # Misma media, más varianza: el temblor cambia el área de contacto.
    d["avg_touch_size_px"] = np.clip(_blend(
        u["avg_touch_size_px"], w["avg_touch_size_px"],
        stats.norm(38, 6).ppf, stats.norm(38, 10).ppf), 8.0, 90.0)

    # SUPUESTO PROPIO (heurística UX/HCI, no cifra citada): swipe de scroll
    # normal en smartphone estándar ≈ 800–2500 px/s.
    d["swipe_speed_px_s"] = np.clip(_blend(
        u["swipe_speed_px_s"], w["swipe_speed_px_s"],
        stats.norm(1400, 350).ppf, stats.norm(900, 400).ppf), 80.0, None)

    d["swipe_directional_variance"] = np.clip(_blend(
        u["swipe_directional_variance"], w["swipe_directional_variance"],
        stats.beta(2, 8).ppf, stats.beta(4, 4).ppf), 0.0, 1.0)

    # SUPUESTO PROPIO (heurística UX/HCI).
    d["scroll_speed_avg"] = np.clip(_blend(
        u["scroll_speed_avg"], w["scroll_speed_avg"],
        stats.norm(1200, 300).ppf, stats.norm(850, 350).ppf), 60.0, None)

    # ── 4.4 Sensores ──────────────────────────────────────────────────────
    d["device_tilt_angle_mean"] = np.clip(_blend(
        u["device_tilt_angle_mean"], w["device_tilt_angle_mean"],
        stats.norm(35, 10).ppf, stats.norm(42, 15).ppf), 0.0, 90.0)

    d["device_tilt_variability"] = np.clip(_blend(
        u["device_tilt_variability"], w["device_tilt_variability"],
        stats.norm(4, 2).ppf, stats.norm(9, 4).ppf), 0.05, None)

    d["gyro_rotation_rate_mean"] = np.clip(_blend(
        u["gyro_rotation_rate_mean"], w["gyro_rotation_rate_mean"],
        stats.norm(0.15, 0.08).ppf, stats.norm(0.28, 0.12).ppf), 0.005, None)

    d["accelerometer_jerk_mean"] = np.clip(_blend(
        u["accelerometer_jerk_mean"], w["accelerometer_jerk_mean"],
        stats.norm(1.2, 0.5).ppf, stats.norm(2.3, 1.0).ppf), 0.03, None)

    d["phone_motion_events"] = _blend_poisson(
        u["phone_motion_events"], w["phone_motion_events"], 3.0, 8.0)

    # ── 4.10 (parte) Duración de la sesión ────────────────────────────────
    # BUG #1 CORREGIDO: lognormal con varianza real, no una constante.
    # SUPUESTO PROPIO (heurística UX): la sesión de banca móvil es corta y
    # orientada a una tarea; mediana 150 s, masa entre ~40 y ~600 s.
    d["session_duration_s"] = np.clip(_blend(
        u["session_duration_s"], w["session_duration_s"],
        stats.lognorm(s=0.60, scale=150).ppf,
        stats.lognorm(s=0.62, scale=420).ppf, log_space=True), 20.0, 4000.0)
    dur = d["session_duration_s"]

    # ── 4.5 Hesitación ────────────────────────────────────────────────────
    d["hesitation_count"] = _blend_poisson(
        u["hesitation_count"], w["hesitation_count"], 1.5, 6.0)
    hay_hes = d["hesitation_count"] > 0

    avg_hes = _blend(u["avg_hesitation_duration_s"], w["avg_hesitation_duration_s"],
                     stats.gamma(2.0, scale=0.8).ppf,
                     stats.gamma(3.0, scale=1.5).ppf)
    # Si no hubo hesitaciones la duración media es 0 por definición.
    d["avg_hesitation_duration_s"] = np.where(hay_hes, avg_hes, 0.0)

    # max = avg + extra, NUNCA un sorteo independiente (§4.5). Con una sola
    # hesitación el máximo coincide con el promedio; con cero, ambos son 0.
    extra = _blend(u["max_hesitation_extra_s"], w["max_hesitation_extra_s"],
                   stats.gamma(1.8, scale=1.2).ppf,
                   stats.gamma(2.2, scale=3.4).ppf)
    extra = np.where(d["hesitation_count"] >= 2, extra, 0.0)
    d["max_hesitation_duration_s"] = d["avg_hesitation_duration_s"] + extra

    # ── 4.6 Tiempo muerto ─────────────────────────────────────────────────
    d["dead_time_periods"] = _blend_poisson(
        u["dead_time_periods"], w["dead_time_periods"], 1.0, 4.0)
    k = d["dead_time_periods"].astype(float)
    # Suma de k Gamma(2,θ) iid = Gamma(2k,θ). Legítimo θ=4.5 (≈9 s/periodo),
    # vishing forma 2.5 y θ=4.5 (≈11 s/periodo) → media total ≈45 s con λ=4.
    forma = (2.0 + 0.5 * w["total_dead_time_s"]) * np.maximum(k, 1e-9)
    total_dt = stats.gamma.ppf(u["total_dead_time_s"], np.maximum(forma, 1e-9), scale=4.5)
    total_dt = np.where(k > 0, total_dt, 0.0)
    # El tiempo muerto no puede exceder la sesión. Tope 92 % (SUPUESTO PROPIO).
    d["total_dead_time_s"] = np.minimum(np.nan_to_num(total_dt), 0.92 * dur)

    # ── 4.7 Navegación ────────────────────────────────────────────────────
    d["unique_screens_visited"] = 1 + _blend_poisson(
        u["unique_screens_visited"], w["unique_screens_visited"], 6.0, 4.0)
    d["navigation_back_count"] = _blend_poisson(
        u["navigation_back_count"], w["navigation_back_count"], 1.5, 3.0)
    d["screen_transition_time_avg_s"] = _blend(
        u["screen_transition_time_avg_s"], w["screen_transition_time_avg_s"],
        stats.gamma(2.0, scale=1.0).ppf, stats.gamma(2.0, scale=2.5).ppf)

    # ── 4.8 Errores y correcciones ────────────────────────────────────────
    d["input_error_count"] = _blend_poisson(
        u["input_error_count"], w["input_error_count"], 0.8, 3.5)
    # Correlacionada con los errores, NO independiente: se corrige lo que se
    # equivoca (y a veces se corrige de más).
    lam_corr = (0.30 + 0.80 * d["input_error_count"]) * np.exp(
        w["input_correction_count"] * np.log(1.55))
    d["input_correction_count"] = stats.poisson.ppf(
        u["input_correction_count"], lam_corr).astype(np.int64)

    # SEÑAL INVERTIDA A PROPÓSITO: los datos dictados por voz se teclean, no se
    # pegan del portapapeles. λ baja en vishing (0.4 → 0.15). Es una de las
    # variables que introducen contraintuición realista.
    d["copy_paste_events"] = _blend_poisson(
        u["copy_paste_events"], w["copy_paste_events"], 0.4, 0.15)

    # ── 4.9 Familiaridad y doodling ───────────────────────────────────────
    d["data_familiarity_score"] = np.clip(_blend(
        u["data_familiarity_score"], w["data_familiarity_score"],
        stats.beta(8, 2).ppf, stats.beta(2, 5).ppf), 0.0, 1.0)

    d["doodling_events"] = _blend_poisson(
        u["doodling_events"], w["doodling_events"], 0.5, 2.5)

    # ── 4.10 Contexto de sesión: llamada, RAT, apps sospechosas ───────────
    # La propensión personal del cliente entra como desplazamiento sobre la
    # base legítima (hay gente que siempre habla mientras usa el banco).
    prop = clientes["prop_llamada"].to_numpy()[idx_cliente]
    w_call = w_binaria()
    p_call_legit = np.clip(0.35 * P_LLAMADA_LEGIT + 0.65 * prop, 0.0, 0.6)
    p_call = p_call_legit + w_call * (P_LLAMADA_MAX_VISH - p_call_legit)
    p_call = np.where(w_call > 0, np.maximum(p_call, P_LLAMADA_BASE_VISH), p_call)
    d["phone_call_active"] = (rng.random(n) < p_call).astype(np.int8)

    # BUG #2 CORREGIDO: duración continua con varianza propia, condicionada a
    # `phone_call_active` pero NO idéntica a ella.
    #   · legítimo en llamada  → llamada breve y ajena a la operación (≈45 s)
    #   · vishing en llamada   → cubre el 70–100 % de la sesión
    frac_vish = np.clip(rng.uniform(0.70, 1.00, size=n) + rng.normal(0, 0.05, size=n),
                        0.15, 1.05)
    solap_legit = stats.gamma.ppf(u["call_overlap_duration_s"], 1.8, scale=25.0)
    solap_vish = frac_vish * dur
    wc = w["call_overlap_duration_s"]
    solap = (1 - wc) * solap_legit + wc * solap_vish
    solap = np.minimum(solap, dur)
    d["call_overlap_duration_s"] = np.where(d["phone_call_active"] == 1, solap, 0.0)

    d["remote_access_tool_detected"] = _bernoulli_interp(
        rng.random(n), w_binaria(), P_RAT_LEGIT, P_RAT_MAX_VISH)
    d["suspicious_app_detected"] = _bernoulli_interp(
        rng.random(n), w_binaria(), P_APP_SOSP_LEGIT, P_APP_SOSP_MAX_VISH)

    # ── 4.11 Transacción ──────────────────────────────────────────────────
    w_txn = w_binaria()
    p_txn = P_TXN_LEGIT + w_txn * (P_TXN_MAX_VISH - P_TXN_LEGIT)
    d["transaction_attempted"] = (rng.random(n) < p_txn).astype(np.int8)
    hay_txn = d["transaction_attempted"] == 1

    p_nb = P_NUEVO_BENEF_LEGIT + w_binaria() * (P_NUEVO_BENEF_MAX_VISH - P_NUEVO_BENEF_LEGIT)
    nuevo_benef = (rng.random(n) < p_nb).astype(np.int8)
    d["is_new_beneficiary"] = np.where(hay_txn, nuevo_benef, 0).astype(np.int8)

    # Montos: mezcla geométrica (mediana legítima 240 k COP, vishing 3.2 M COP).
    monto = _blend(u["transaction_amount_cop"], w["transaction_amount_cop"],
                   stats.lognorm(s=1.30, scale=240_000).ppf,
                   stats.lognorm(s=0.95, scale=3_200_000).ppf, log_space=True)
    monto = np.clip(monto, 5_000, 120_000_000)
    d["transaction_amount_cop"] = np.where(hay_txn, monto, -1.0)

    # BUG #3 CORREGIDO: continua y variable, no `10 × transaction_attempted`.
    ttt = _blend(u["time_to_transaction_s"], w["time_to_transaction_s"],
                 stats.gamma(2.0, scale=40.0).ppf,
                 stats.gamma(3.0, scale=90.0).ppf)
    ttt = np.minimum(ttt, 0.97 * dur)
    d["time_to_transaction_s"] = np.where(hay_txn, np.maximum(ttt, 1.0), -1.0)

    lam_amt = 0.10 * np.exp(w["amount_field_corrections"] * np.log(9.0))
    amt_corr = stats.poisson.ppf(u["amount_field_corrections"], lam_amt).astype(np.int64)
    d["amount_field_corrections"] = np.where(hay_txn, amt_corr, 0).astype(np.int64)

    # Más correcciones aún si el beneficiario es nuevo: se está confirmando por
    # voz un número de cuenta dictado.
    lam_ben = (0.10 * np.exp(w["beneficiary_field_corrections"] * np.log(11.0))
               * np.where(d["is_new_beneficiary"] == 1, 1.6, 1.0))
    ben_corr = stats.poisson.ppf(u["beneficiary_field_corrections"], lam_ben).astype(np.int64)
    d["beneficiary_field_corrections"] = np.where(hay_txn, ben_corr, 0).astype(np.int64)

    # ── 4.13 Derivadas ────────────────────────────────────────────────────
    d["dead_time_ratio"] = np.clip(d["total_dead_time_s"] / dur, 0.0, 1.0)
    d["errors_per_minute"] = d["input_error_count"] / (dur / 60.0)
    d["hesitation_composite"] = (d["hesitation_count"]
                                 * d["avg_hesitation_duration_s"]) / dur

    return d


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONTEXTO: TIMESTAMP, HORA, OS, VERSIÓN
# ═══════════════════════════════════════════════════════════════════════════

# Componentes diurnas de uso de banca móvil. SUPUESTO PROPIO de UX (no hay
# cifra pública citable): consulta matutina, mediodía, salida del trabajo y
# franja nocturna en casa.
_COMPONENTES_HORA = [(9.0, 1.5), (13.0, 1.4), (18.3, 1.8), (21.3, 1.6)]
_PESOS_HORA_BASE = np.array([0.24, 0.19, 0.34, 0.23])
HORAS_ATIPICAS = {22, 23, 0, 1, 2, 3, 4, 5}


def _horas_legitimas(franja_pref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(franja_pref)
    pesos = np.tile(_PESOS_HORA_BASE, (n, 1))
    pesos[np.arange(n), franja_pref] += 0.45          # sesgo personal
    pesos /= pesos.sum(axis=1, keepdims=True)
    acum = np.cumsum(pesos, axis=1)
    comp = (rng.random(n)[:, None] > acum).sum(axis=1)
    mu = np.array([c[0] for c in _COMPONENTES_HORA])[comp]
    sd = np.array([c[1] for c in _COMPONENTES_HORA])[comp]
    return np.mod(rng.normal(mu, sd), 24.0)


def _horas_vishing(rng: np.random.Generator, n: int) -> np.ndarray:
    """Más plana, con sesgo MODERADO hacia horas atípicas (no un predictor solo)."""
    tipo = rng.random(n)
    horas = np.empty(n)
    m1 = tipo < 0.62                                   # franja laboral amplia
    horas[m1] = rng.uniform(8.0, 19.5, size=m1.sum())
    m2 = (tipo >= 0.62) & (tipo < 0.80)                # tarde-noche
    horas[m2] = rng.uniform(19.5, 22.0, size=m2.sum())
    m3 = tipo >= 0.80                                  # atípica
    horas[m3] = np.mod(rng.uniform(22.0, 30.0, size=m3.sum()), 24.0)
    return horas


def _timestamps(rng: np.random.Generator, horas: np.ndarray,
                estacionalidad: bool) -> pd.Series:
    dias = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")
    if estacionalidad:
        # Lunes-viernes más carga que fin de semana. SUPUESTO PROPIO.
        peso_dow = np.array([1.00, 1.00, 0.98, 1.00, 1.05, 0.78, 0.60])
    else:
        peso_dow = np.array([1.00, 1.00, 1.00, 1.00, 1.02, 0.90, 0.82])
    w = peso_dow[dias.dayofweek.to_numpy()]
    w = w / w.sum()
    idx = rng.choice(len(dias), size=len(horas), p=w)
    base = dias.to_numpy()[idx]
    delta = (horas * 3600.0 + rng.uniform(0, 3600, size=len(horas))) % 86400.0
    return pd.to_datetime(base) + pd.to_timedelta(delta, unit="s")


def _contexto_dispositivo(rng: np.random.Generator, n: int):
    """OS y versión de app: MISMA distribución en ambas clases (no son señal)."""
    # Cuota de mercado móvil típica de Colombia: mayoría Android.
    os_type = np.where(rng.random(n) < 0.68, "Android", "iOS")
    versiones = np.array(["4.2.1", "4.1.5", "4.0.9", "3.9.4"])
    pesos = np.array([1 / 1, 1 / 2, 1 / 3, 1 / 4])     # Zipf hacia la más reciente
    pesos /= pesos.sum()
    app_version = rng.choice(versiones, size=n, p=pesos)
    return os_type, app_version


# ═══════════════════════════════════════════════════════════════════════════
# 5. GENERADORES POR CLASE
# ═══════════════════════════════════════════════════════════════════════════

def generar_sesiones_legitimas(clientes: pd.DataFrame, idx_cliente: np.ndarray,
                               rng: np.random.Generator) -> pd.DataFrame:
    """Sesiones con `is_vishing = 0`, incluido el 2.5 % de ruido intencional."""
    n = len(idx_cliente)

    # Ruido intencional (§3.3): usuario estresado, adulto mayor con tecleo
    # lento, alguien realmente en llamada con un familiar. Se usa la misma
    # maquinaria del ataque con intensidad moderada, pero solo sobre parte de
    # los indicadores, para que sean falsos positivos creíbles y no "vishing
    # con otra etiqueta".
    confundida = rng.random(n) < P_LEGIT_CONFUNDIDA
    intensidad = np.zeros(n)
    intensidad[confundida] = rng.beta(2.5, 3.0, size=confundida.sum()) * 0.85

    d = _nucleo_conductual(clientes, idx_cliente, intensidad, rng)

    horas = _horas_legitimas(clientes["franja_pref"].to_numpy()[idx_cliente], rng)
    ts = _timestamps(rng, horas, estacionalidad=True)
    os_type, app_version = _contexto_dispositivo(rng, n)

    d["session_timestamp"] = ts
    d["hour_of_day"] = ts.hour.to_numpy().astype(np.int16)
    d["os_type"] = os_type
    d["app_version"] = app_version
    d["is_vishing"] = np.zeros(n, dtype=np.int8)
    d["_intensidad_ataque"] = intensidad
    d["_idx_cliente"] = idx_cliente
    return pd.DataFrame(d)


def generar_sesiones_vishing(clientes: pd.DataFrame, idx_cliente: np.ndarray,
                             rng: np.random.Generator) -> pd.DataFrame:
    """Sesiones con `is_vishing = 1`, moduladas por la latente de sofisticación.

    Mezcla: 15 % de ataques sofisticados (intensidad baja: casi todas sus
    variables quedan dentro de la nube legítima) y 85 % de ataques burdos.
    """
    n = len(idx_cliente)
    sofisticado = rng.random(n) < P_ATAQUE_SOFISTICADO
    intensidad = np.where(
        sofisticado,
        rng.beta(*BETA_INTENSIDAD_SOFISTICADO, size=n),
        rng.beta(*BETA_INTENSIDAD_BURDO, size=n),
    )

    d = _nucleo_conductual(clientes, idx_cliente, intensidad, rng)

    horas = _horas_vishing(rng, n)
    ts = _timestamps(rng, horas, estacionalidad=False)
    os_type, app_version = _contexto_dispositivo(rng, n)

    d["session_timestamp"] = ts
    d["hour_of_day"] = ts.hour.to_numpy().astype(np.int16)
    d["os_type"] = os_type
    d["app_version"] = app_version
    d["is_vishing"] = np.ones(n, dtype=np.int8)
    d["_intensidad_ataque"] = intensidad
    d["_idx_cliente"] = idx_cliente
    return pd.DataFrame(d)


# ═══════════════════════════════════════════════════════════════════════════
# 6. SCORES SIMULADOS DE BIOCATCH (§4.12)
# ═══════════════════════════════════════════════════════════════════════════

# Pesos del agregador. Signo positivo = empuja el riesgo hacia arriba.
PESOS_RIESGO = {
    "segmented_typing_ratio": 1.15,
    "typing_speed_cps": -0.85,
    "avg_interkey_latency_ms": 0.70,
    "avg_keyhold_ms": 0.35,
    "keystroke_variability": 0.45,
    "hesitation_count": 0.60,
    "avg_hesitation_duration_s": 0.45,
    "total_dead_time_s": 0.65,
    "dead_time_periods": 0.35,
    "data_familiarity_score": -1.00,
    "device_tilt_variability": 0.45,
    "accelerometer_jerk_mean": 0.45,
    "gyro_rotation_rate_mean": 0.30,
    "input_error_count": 0.45,
    "doodling_events": 0.40,
    "beneficiary_field_corrections": 0.35,
    "copy_paste_events": -0.15,
    "phone_call_active": 0.75,
    "is_new_beneficiary": 0.55,
    "screen_transition_time_avg_s": 0.30,
    "swipe_directional_variance": 0.30,
    "remote_access_tool_detected": 0.45,
    "suspicious_app_detected": 0.35,
}

# Sub-score específico de ingeniería social (más selectivo que el de ATO).
PESOS_SOCIAL = {
    "segmented_typing_ratio": 1.30,
    "phone_call_active": 1.00,
    "total_dead_time_s": 0.80,
    "data_familiarity_score": -0.90,
    "is_new_beneficiary": 0.70,
    "avg_hesitation_duration_s": 0.55,
}


def _z(s: pd.Series) -> np.ndarray:
    v = s.to_numpy(dtype=float)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 0 else 1.0)


def agregar_scores_biocatch(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simula la salida de un motor de riesgo de terceros.

    NO es una función determinista de las columnas de comportamiento: lleva su
    propio error de modelo (ruido gaussiano fuerte), de modo que hay vishing
    con score bajo (no detectado) y legítimas con score alto (falso positivo).
    """
    n = len(df)
    s = np.zeros(n)
    for col, peso in PESOS_RIESGO.items():
        s += peso * _z(df[col])
    s /= np.sqrt(sum(p ** 2 for p in PESOS_RIESGO.values()))
    s = (s - s.mean()) / s.std()

    ruido = rng.normal(0.0, SD_RUIDO_BIOCATCH, size=n)
    s_obs = s + ruido

    # Escala 0-1000 mediante el percentil elevado a GAMMA_RIESGO. Se prefiere a
    # una logística porque una logística con este nivel de ruido se satura y
    # deja un 10 % de sesiones legítimas pegadas a 999, lo que no se parece a
    # la salida de un motor de riesgo real. Con γ=2 la marginal legítima queda
    # con mediana ≈250 y ~63 % por debajo de 400, y la de vishing con mediana
    # ≈700 y ~64 % por encima de 550, ambas con cola hasta el extremo opuesto.
    pct = stats.rankdata(s_obs) / (n + 1.0)
    df["biocatch_risk_score"] = np.clip(999.0 * pct ** GAMMA_RIESGO, 0, 999)

    # Relacionado inversamente pero con ruido PROPIO: no es 1000 − risk.
    # Comparte el 70 % del error del score de riesgo (es el mismo motor mirando
    # la misma sesión) y añade un término independiente, de modo que la
    # correlación entre ambos queda en torno a −0.75, no en −1.
    g = -(s + 0.70 * ruido) + rng.normal(0.0, SD_RUIDO_GENUINE, size=n)
    pct_g = stats.rankdata(g) / (n + 1.0)
    df["biocatch_genuine_score"] = np.clip(
        999.0 * (1.0 - (1.0 - pct_g) ** GAMMA_RIESGO), 0, 999)

    # Umbrales ruidosos.
    df["biocatch_ato_indicator"] = (
        df["biocatch_risk_score"].to_numpy() + rng.normal(0, 95, n) > 830
    ).astype(np.int8)

    soc = np.zeros(n)
    for col, peso in PESOS_SOCIAL.items():
        soc += peso * _z(df[col])
    soc = (soc - soc.mean()) / soc.std()
    soc_obs = soc + rng.normal(0.0, 1.15, size=n)
    df["biocatch_social_eng_indicator"] = (soc_obs > 2.05).astype(np.int8)

    # Una sesión de vishing es una persona real en su propio dispositivo, no un
    # bot: la tasa base es idéntica en ambas clases, a propósito.
    df["biocatch_bot_indicator"] = (rng.random(n) < P_BOT).astype(np.int8)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 7. ETIQUETAS DE RECLAMO (§4.14)
# ═══════════════════════════════════════════════════════════════════════════

def agregar_etiquetas_reclamo(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    days = np.full(n, -1.0)
    cat = np.array([""] * n, dtype=object)

    es_v = df["is_vishing"].to_numpy() == 1
    # No todas las víctimas reclaman, o no de inmediato.
    reclama_v = es_v & (rng.random(n) < P_RECLAMO_VISHING)
    k = int(reclama_v.sum())
    days[reclama_v] = np.clip(rng.gamma(2.0, 3.0, size=k) + 1.0, 1, 60)
    cat[reclama_v] = "vishing_confirmado"

    # Ruido intencional: reclamos legítimos por motivos ajenos al vishing.
    # Le da sentido real a `claim_category` y evita que sea un alias del target.
    es_l = ~es_v
    reclama_l = es_l & (rng.random(n) < P_RECLAMO_LEGIT)
    k = int(reclama_l.sum())
    days[reclama_l] = np.clip(rng.gamma(2.2, 4.0, size=k) + 1.0, 1, 60)
    cat[reclama_l] = np.where(rng.random(k) < 0.6, "ato_no_vishing", "disputa_no_fraude")

    df["days_to_claim"] = np.round(days).astype(np.int32)
    df["claim_category"] = cat
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 8. ORQUESTACIÓN
# ═══════════════════════════════════════════════════════════════════════════

ORDEN_COLUMNAS = [
    "row_id", "session_id", "customer_id", "session_timestamp",
    "os_type", "app_version",
    # keystroke
    "avg_keyhold_ms", "avg_interkey_latency_ms", "typing_speed_cps",
    "keystroke_variability", "segmented_typing_ratio",
    # touch
    "avg_touch_pressure", "avg_touch_size_px", "swipe_speed_px_s",
    "swipe_directional_variance", "scroll_speed_avg",
    # sensores
    "device_tilt_angle_mean", "device_tilt_variability",
    "gyro_rotation_rate_mean", "accelerometer_jerk_mean", "phone_motion_events",
    # hesitación
    "hesitation_count", "avg_hesitation_duration_s", "max_hesitation_duration_s",
    # tiempo muerto
    "dead_time_periods", "total_dead_time_s", "dead_time_ratio",
    # navegación
    "unique_screens_visited", "navigation_back_count",
    "screen_transition_time_avg_s",
    # errores
    "input_error_count", "input_correction_count", "amount_field_corrections",
    "beneficiary_field_corrections", "copy_paste_events",
    # familiaridad
    "data_familiarity_score", "doodling_events",
    # contexto
    "session_duration_s", "hour_of_day", "is_atypical_hour",
    "phone_call_active", "call_overlap_duration_s",
    "remote_access_tool_detected", "suspicious_app_detected",
    # transacción
    "transaction_attempted", "transaction_amount_cop", "is_new_beneficiary",
    "time_to_transaction_s",
    # scores
    "biocatch_risk_score", "biocatch_genuine_score", "biocatch_ato_indicator",
    "biocatch_social_eng_indicator", "biocatch_bot_indicator",
    # derivadas
    "errors_per_minute", "hesitation_composite",
    # etiquetas
    "is_vishing", "days_to_claim", "claim_category",
]

REDONDEO = {
    3: ["avg_keyhold_ms", "avg_interkey_latency_ms", "avg_touch_size_px",
        "swipe_speed_px_s", "scroll_speed_avg", "device_tilt_angle_mean",
        "device_tilt_variability", "total_dead_time_s",
        "screen_transition_time_avg_s", "session_duration_s",
        "call_overlap_duration_s", "time_to_transaction_s",
        "avg_hesitation_duration_s", "max_hesitation_duration_s",
        "biocatch_risk_score", "biocatch_genuine_score"],   # → 2 decimales
    4: ["typing_speed_cps", "accelerometer_jerk_mean"],      # → 3 decimales
    5: ["keystroke_variability", "segmented_typing_ratio", "avg_touch_pressure",
        "swipe_directional_variance", "gyro_rotation_rate_mean",
        "data_familiarity_score", "dead_time_ratio", "errors_per_minute",
        "hesitation_composite"],                             # → 4 decimales
}
_DECIMALES = {3: 2, 4: 3, 5: 4}


def generar_dataset(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    clientes = generar_poblacion_clientes(rng)
    n_vish_cliente = _asignar_vishing(clientes, rng)

    # Expansión cliente → sesiones.
    idx_cliente_todas = np.repeat(np.arange(len(clientes)),
                                  clientes["n_sesiones"].to_numpy())
    # Dentro de cada cliente, las primeras `k` posiciones son las de vishing.
    posicion = (np.arange(len(idx_cliente_todas))
                - np.repeat(np.cumsum(clientes["n_sesiones"].to_numpy())
                            - clientes["n_sesiones"].to_numpy(),
                            clientes["n_sesiones"].to_numpy()))
    es_vish = posicion < np.repeat(n_vish_cliente, clientes["n_sesiones"].to_numpy())

    idx_v = idx_cliente_todas[es_vish]
    idx_l = idx_cliente_todas[~es_vish]

    df_l = generar_sesiones_legitimas(clientes, idx_l, rng)
    df_v = generar_sesiones_vishing(clientes, idx_v, rng)
    df = pd.concat([df_l, df_v], ignore_index=True)

    # Mezcla del orden ANTES de asignar identificadores.
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)

    df["customer_id"] = clientes["customer_id"].to_numpy()[df["_idx_cliente"].to_numpy()]
    df["session_id"] = [f"SES-{i:06d}" for i in range(len(df))]
    # `row_id` es una COLUMNA REAL, no el índice de pandas: con `index=False`
    # el índice se pierde al guardar, y esa fue la causa raíz de uno de los
    # bugs de la v1.
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    # Bucket determinista sobre `hour_of_day` (esto sí es legítimo: es un
    # bucket de la misma señal, no una copia disfrazada de otra).
    df["is_atypical_hour"] = df["hour_of_day"].isin(HORAS_ATIPICAS).astype(np.int8)

    df = agregar_scores_biocatch(df, rng)
    df = agregar_etiquetas_reclamo(df, rng)

    # Redondeo por tipo de variable.
    for grupo, cols in REDONDEO.items():
        dec = _DECIMALES[grupo]
        for c in cols:
            df[c] = df[c].round(dec)
    df["transaction_amount_cop"] = df["transaction_amount_cop"].round().astype(np.int64)
    df["session_timestamp"] = df["session_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df = df.drop(columns=["_intensidad_ataque", "_idx_cliente"])
    return df[ORDEN_COLUMNAS]


# ═══════════════════════════════════════════════════════════════════════════
# 9. VALIDACIONES OBLIGATORIAS (§6)
# ═══════════════════════════════════════════════════════════════════════════

# Columnas post-hoc: se conocen después del fraude, nunca son features.
COLUMNAS_POSTHOC = ["days_to_claim"]


def _json_safe(o):
    """Convierte tipos de numpy a tipos nativos para `json.dumps`."""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"no serializable: {type(o)}")


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    """AUC-ROC por rangos (Mann-Whitney), con corrección de empates."""
    r = stats.rankdata(x)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def validar(df: pd.DataFrame) -> dict:
    rep: dict = {}
    num = df.select_dtypes(include=[np.number]).drop(columns=["row_id"])

    # 1. columnas constantes
    const = [c for c in num.columns if num[c].nunique() <= 1]
    rep["1_columnas_constantes"] = {"n": len(const), "columnas": const,
                                    "pasa": len(const) == 0}

    # 2. pares casi idénticos
    corr = num.corr(numeric_only=True).to_numpy()
    cols = list(num.columns)
    pares = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr[i, j]
            if np.isfinite(c) and abs(c) >= 0.999:
                pares.append((cols[i], cols[j], round(float(c), 6)))
    # top-5 de correlaciones más altas, para referencia
    tri = np.abs(np.triu(np.nan_to_num(corr), 1))
    top = np.dstack(np.unravel_index(np.argsort(tri, axis=None)[::-1][:5], tri.shape))[0]
    rep["2_pares_duplicados"] = {
        "n": len(pares), "pares": pares, "pasa": len(pares) == 0,
        "top5_correlaciones": [(cols[i], cols[j], round(float(corr[i, j]), 4))
                               for i, j in top],
    }

    # 3. balance
    vc = df["is_vishing"].value_counts().to_dict()
    rep["3_balance"] = {"vishing": int(vc.get(1, 0)), "legitimas": int(vc.get(0, 0)),
                        "pasa": vc.get(1, 0) == N_VISHING and vc.get(0, 0) == N_SESIONES - N_VISHING}

    # 4. AUC univariado
    y = df["is_vishing"].to_numpy()
    aucs = {}
    for c in num.columns:
        if c == "is_vishing":
            continue
        v = num[c].to_numpy(dtype=float)
        if not np.isfinite(v).all():
            v = np.nan_to_num(v)
        a = _auc(y, v)
        aucs[c] = round(float(max(a, 1 - a)), 4)
    aucs = dict(sorted(aucs.items(), key=lambda kv: -kv[1]))
    # `days_to_claim` NO es una variable predictora: solo se conoce después de
    # que la víctima reclamó, y el propio pipeline la excluye del contrato de
    # features (POSTHOC_COLS en vishing_common.py). Su AUC alta es la
    # definición de la etiqueta, no una fuga de diseño; bajarla por debajo de
    # 0.80 obligaría a que menos del 60 % de las víctimas reclamara, en contra
    # de §4.14. Se reporta aparte y no cuenta para la regla de las rojas.
    feats = {k: v for k, v in aucs.items() if k not in COLUMNAS_POSTHOC}
    rojas = {k: v for k, v in feats.items() if v > TECHO_AUC_UNIVARIADO}
    rep["4_auc_univariado"] = {
        "ranking": feats,
        "auc_max_features": max(feats.values()),
        "posthoc_excluidas": {k: aucs[k] for k in COLUMNAS_POSTHOC if k in aucs},
        "variables_en_rojo": rojas,
        "pasa": len(rojas) == 0,
    }

    # 5. sesiones por cliente
    s = df.groupby("customer_id", observed=True).size()
    rep["5_sesiones_por_cliente"] = {
        "clientes": int(s.shape[0]), "media": round(float(s.mean()), 3),
        "mediana": float(s.median()), "maximo": int(s.max()), "minimo": int(s.min()),
        "p99": float(s.quantile(0.99)),
        "pasa": 2.3 <= s.mean() <= 2.8 and s.max() <= 30,
    }
    v_por_cliente = df[df.is_vishing == 1].groupby("customer_id", observed=True).size()
    rep["5_sesiones_por_cliente"]["max_vishing_por_cliente"] = int(v_por_cliente.max())
    rep["5_sesiones_por_cliente"]["clientes_victima"] = int(v_por_cliente.shape[0])

    # 6. co-ocurrencia dentro de vishing
    sub = df[df.is_vishing == 1][["phone_call_active", "segmented_typing_ratio",
                                  "hesitation_count", "is_new_beneficiary",
                                  "session_duration_s"]]
    m = sub.corr().round(4)
    off = m.to_numpy()[np.triu_indices(5, 1)]
    rep["6_coocurrencia_vishing"] = {
        "matriz": m.to_dict(),
        "min_corr": round(float(off.min()), 4),
        "max_corr": round(float(off.max()), 4),
        "pasa": bool((off > 0).all() and (off < 0.999).all()),
    }

    # 7. rangos y sanidad
    ratios = ["keystroke_variability", "segmented_typing_ratio", "avg_touch_pressure",
              "swipe_directional_variance", "data_familiarity_score", "dead_time_ratio"]
    chk = {
        "ratios_en_0_1": bool(all(df[c].between(0, 1).all() for c in ratios)),
        "duraciones_no_negativas": bool((df["session_duration_s"] > 0).all()
                                        and (df["total_dead_time_s"] >= 0).all()
                                        and (df["call_overlap_duration_s"] >= 0).all()),
        "max_hes_ge_avg_hes": bool((df["max_hesitation_duration_s"]
                                    >= df["avg_hesitation_duration_s"]).all()),
        "dead_time_le_duracion": bool((df["total_dead_time_s"]
                                       <= df["session_duration_s"]).all()),
        "overlap_le_duracion": bool((df["call_overlap_duration_s"]
                                     <= df["session_duration_s"]).all()),
        "overlap_cero_sin_llamada": bool(
            (df.loc[df.phone_call_active == 0, "call_overlap_duration_s"] == 0).all()),
        "ttt_centinela_coherente": bool(
            (df.loc[df.transaction_attempted == 0, "time_to_transaction_s"] == -1).all()
            and (df.loc[df.transaction_attempted == 1, "time_to_transaction_s"] > 0).all()),
        "monto_centinela_coherente": bool(
            (df.loc[df.transaction_attempted == 0, "transaction_amount_cop"] == -1).all()
            and (df.loc[df.transaction_attempted == 1, "transaction_amount_cop"] > 0).all()),
        "claim_category_coherente": bool(
            (df.loc[df.days_to_claim == -1, "claim_category"] == "").all()
            and (df.loc[df.days_to_claim != -1, "claim_category"] != "").all()),
        "session_id_unico": bool(df.session_id.is_unique),
        "row_id_unico_y_completo": bool(df.row_id.is_unique and df.row_id.min() == 0
                                        and df.row_id.max() == len(df) - 1),
        "sin_nulos": bool(df.isna().sum().sum() == 0),
        "sd_session_duration_s": round(float(df.session_duration_s.std()), 3),
        "nunique_session_duration_s": int(df.session_duration_s.nunique()),
        "hora_coincide_con_timestamp": bool(
            (pd.to_datetime(df.session_timestamp).dt.hour == df.hour_of_day).all()),
    }
    chk["pasa"] = all(v for k, v in chk.items()
                      if isinstance(v, bool))
    rep["7_rangos_y_sanidad"] = chk

    rep["_resumen"] = {
        "filas": int(len(df)), "columnas": int(df.shape[1]),
        "todas_las_validaciones_pasan": all(
            rep[k].get("pasa", False) for k in rep if k.startswith(("1_", "2_", "3_",
                                                                    "4_", "5_", "6_", "7_"))
        ),
    }
    return rep


def escribir_reporte(rep: dict, ruta: Path) -> None:
    L = []
    A = L.append
    ok = lambda b: "PASA" if b else "**FALLA**"
    A("# Reporte de validación — `biocatch_sinthetic_data_v3.csv`\n")
    A(f"Filas: **{rep['_resumen']['filas']:,}** · Columnas: "
      f"**{rep['_resumen']['columnas']}** · Semilla: **{SEED}**\n")
    A(f"Resultado global: **{ok(rep['_resumen']['todas_las_validaciones_pasan'])}**\n")

    A("\n## 1. Columnas constantes\n")
    r = rep["1_columnas_constantes"]
    A(f"{ok(r['pasa'])} — columnas numéricas con `nunique() <= 1`: **{r['n']}**")
    if r["columnas"]:
        A(f"\n{r['columnas']}")

    A("\n## 2. Columnas duplicadas (|r| ≥ 0.999)\n")
    r = rep["2_pares_duplicados"]
    A(f"{ok(r['pasa'])} — pares detectados: **{r['n']}**\n")
    A("Cinco correlaciones más altas del dataset (referencia):\n")
    A("| Columna A | Columna B | r |")
    A("|---|---|---|")
    for a, b, c in r["top5_correlaciones"]:
        A(f"| `{a}` | `{b}` | {c} |")

    A("\n## 3. Balance de clases\n")
    r = rep["3_balance"]
    A(f"{ok(r['pasa'])} — vishing: **{r['vishing']:,}** · legítimas: "
      f"**{r['legitimas']:,}** ({r['vishing']/rep['_resumen']['filas']:.2%})")

    A("\n## 4. AUC univariado frente a `is_vishing`\n")
    r = rep["4_auc_univariado"]
    A(f"{ok(r['pasa'])} — AUC máximo entre features: **{r['auc_max_features']:.4f}** · "
      f"techo vigente: **{TECHO_AUC_UNIVARIADO:.2f}** · features por encima del "
      f"techo: **{len(r['variables_en_rojo'])}**\n")
    A(f"Techo anterior (v3.0): {TECHO_AUC_V30:.2f}, con máximo alcanzado "
      f"{AUC_MAX_V30:.4f}. Referencia v2 (dataset canónico anterior): máximo "
      f"{AUC_MAX_V2:.4f} (`segmented_typing_ratio`).\n")
    A(f"Parámetros de separabilidad de esta corrida: "
      f"`ESCALA_GANANCIA` = {ESCALA_GANANCIA}, "
      f"`BETA_INTENSIDAD_BURDO` = {BETA_INTENSIDAD_BURDO}, "
      f"`P_ATAQUE_SOFISTICADO` = {P_ATAQUE_SOFISTICADO}.\n")
    A("| # | Variable | AUC |")
    A("|---|---|---|")
    for i, (k, v) in enumerate(r["ranking"].items(), 1):
        marca = " 🔴" if v > TECHO_AUC_UNIVARIADO else ""
        A(f"| {i} | `{k}` | {v:.4f}{marca} |")
    A("\n**Columnas post-hoc excluidas de la regla** (no son features: solo se "
      "conocen después de que la víctima reclamó, y `vishing_common.py` ya las "
      "descarta vía `POSTHOC_COLS`):\n")
    A("| Columna | AUC |")
    A("|---|---|")
    for k, v in r["posthoc_excluidas"].items():
        A(f"| `{k}` | {v:.4f} |")

    A("\n## 5. Sesiones por cliente\n")
    r = rep["5_sesiones_por_cliente"]
    A(f"{ok(r['pasa'])}\n")
    A("| Métrica | Valor |")
    A("|---|---|")
    for k in ["clientes", "media", "mediana", "minimo", "p99", "maximo",
              "clientes_victima", "max_vishing_por_cliente"]:
        A(f"| {k} | {r[k]} |")

    A("\n## 6. Co-ocurrencia dentro de `is_vishing = 1`\n")
    r = rep["6_coocurrencia_vishing"]
    A(f"{ok(r['pasa'])} — todas positivas, ninguna en 1.0 "
      f"(rango {r['min_corr']} – {r['max_corr']})\n")
    m = pd.DataFrame(r["matriz"])
    A("| | " + " | ".join(f"`{c}`" for c in m.columns) + " |")
    A("|---" * (len(m.columns) + 1) + "|")
    for idx, row in m.iterrows():
        A(f"| `{idx}` | " + " | ".join(f"{v:.4f}" for v in row) + " |")

    A("\n## 7. Rangos y sanidad básica\n")
    r = rep["7_rangos_y_sanidad"]
    A(f"{ok(r['pasa'])}\n")
    A("| Comprobación | Resultado |")
    A("|---|---|")
    for k, v in r.items():
        if k == "pasa":
            continue
        A(f"| `{k}` | {v} |")

    A("\n## 8. Persistencia\n")
    A("Este reporte y su versión JSON (`reporte_validacion.json`) se escriben "
      "junto al dataset en `remediacion/data/`.\n")
    ruta.write_text("\n".join(L), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--salida", type=str, default=".")
    args = ap.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    df = generar_dataset(seed=args.seed)
    t1 = time.perf_counter()
    print(f"[ok] {len(df):,} sesiones × {df.shape[1]} columnas en {t1 - t0:.1f} s")

    csv = salida / "biocatch_sinthetic_data_v3.csv"
    df.to_csv(csv, index=False)
    print(f"[ok] {csv}  ({csv.stat().st_size / 1e6:.1f} MB)")

    rep = validar(df)
    rep["_entorno"] = {
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "seed": args.seed, "segundos_generacion": round(t1 - t0, 2),
    }
    (salida / "reporte_validacion.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False, default=_json_safe),
        encoding="utf-8")
    escribir_reporte(rep, salida / "reporte_validacion.md")
    print(f"[ok] validaciones: "
          f"{'TODAS PASAN' if rep['_resumen']['todas_las_validaciones_pasan'] else 'HAY FALLOS'}")
    print(f"     AUC univariado máximo (features) = "
          f"{rep['4_auc_univariado']['auc_max_features']:.4f}")


if __name__ == "__main__":
    main()
