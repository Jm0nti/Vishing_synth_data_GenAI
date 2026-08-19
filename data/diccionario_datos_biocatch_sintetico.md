# Diccionario de datos — `biocatch_sinthetic_data.csv`

**Versión:** v3 · **Semilla:** 42 · **Generado por:** `generar_dataset_sintetico.py`
**Filas:** 100.000 sesiones · **Columnas:** 58 · **Canal:** 100 % app bancaria móvil
**Periodo:** 2025-01-01 00:00 → 2025-12-31 23:59 (12 meses continuos)
**Moneda:** COP · **Clientes distintos:** 38.000 (media 2,63 sesiones/cliente)

| Clase | Filas | % |
|---|---|---|
| `is_vishing = 1` | 5.000 | 5,00 % |
| `is_vishing = 0` | 95.000 | 95,00 % |

> Dataset **sintético**. Ningún dato proviene de clientes reales: todas las
> columnas se generan paramétricamente. No sustituye a datos reales de vishing
> confirmado; sirve para desarrollar y auditar el pipeline con control total
> sobre las distribuciones.

---

## 1. Convenciones

### 1.1 Valor centinela `-1`

`-1` significa **"no aplica"**, nunca "cero" ni "faltante". Se usa en las
variables continuas condicionadas:

| Columna | `-1` cuando… |
|---|---|
| `transaction_amount_cop` | `transaction_attempted = 0` |
| `time_to_transaction_s` | `transaction_attempted = 0` |
| `days_to_claim` | la sesión no generó ningún reclamo |

Antes de entrenar hay que decidir explícitamente qué se hace con estos valores
(imputar, binarizar en `tiene_transaccion`, o dejarlos como categoría propia
en modelos de árboles, que es lo que hace el pipeline actual).

**Cero real, no centinela.** Estas columnas usan `0` porque el cero es
factualmente correcto, no una ausencia de dato:

- `call_overlap_duration_s = 0` ⇔ `phone_call_active = 0` (no hubo llamada, el
  solapamiento es cero).
- `avg_hesitation_duration_s = max_hesitation_duration_s = 0` ⇔
  `hesitation_count = 0`.
- `total_dead_time_s = 0` ⇔ `dead_time_periods = 0`.
- `amount_field_corrections`, `beneficiary_field_corrections`,
  `is_new_beneficiary` valen `0` si no hubo intento de transacción (no existía
  el campo que corregir).

`claim_category` usa cadena vacía `""` cuando `days_to_claim = -1`.

### 1.2 Sin nulos

El dataset no contiene `NaN` en ninguna columna. Validado en §7 del reporte.

### 1.3 Identidad de fila

`row_id` es una **columna real** (0 … 99.999), no el índice de pandas. Sobrevive
a `to_csv(index=False)`. Este fue el fallo raíz de la v1 del proyecto: la
identidad de fila se perdía al guardar y `drop(index=…)` borraba filas
equivocadas. Todo artefacto derivado debe arrastrar `row_id`.

---

## 2. Esquema completo

Las dos columnas de estadísticos muestran **media ± desviación estándar**
observadas en el CSV generado con semilla 42. `AUC` es el AUC-ROC univariado de
la variable cruda contra `is_vishing` (dirección optimizada: `max(auc, 1-auc)`).

### 2.1 Identificadores y contexto

| Columna | Tipo | Descripción |
|---|---|---|
| `row_id` | int | Identidad de fila, 0-99.999, única y densa. Nunca es feature. |
| `session_id` | str | `SES-000000` … `SES-099999`, único. |
| `customer_id` | str | `CUS-00000` … `CUS-37999`. Un cliente tiene 1-22 sesiones. Es la **clave de agrupación obligatoria para el split** (hay clientes con sesiones legítimas y de vishing). |
| `session_timestamp` | str | `YYYY-MM-DD HH:MM:SS`. Legítimas con estacionalidad semanal (menos fin de semana); vishing con distribución más plana. |
| `os_type` | cat | `Android` (68 %) / `iOS` (32 %). **Misma distribución en ambas clases**: no es señal de fraude. |
| `app_version` | cat | `4.2.1` (47,9 %), `4.1.5` (24,1 %), `4.0.9` (15,9 %), `3.9.4` (12,2 %). Ponderación tipo Zipf. **Misma distribución en ambas clases.** |

No hay columna `device_type`: el dataset es 100 % móvil, y una columna constante
no aporta información (fue uno de los hallazgos de la auditoría v2). Si
`vishing_common.py` filtra por `MOBILE_ONLY`, ese filtro es ahora un no-op.

### 2.2 Variables numéricas

| Columna | Legítimo (μ ± σ) | Vishing (μ ± σ) | Rango observado | AUC |
|---|---|---|---|---|
| `avg_keyhold_ms` | 110.169 ± 18.246 | 123.775 ± 23.837 | 40,1 … 213,36 | 0.6739 |
| `avg_interkey_latency_ms` | 221.232 ± 47.156 | 277.489 ± 124.72 | 55 … 1.337,92 | 0.6358 |
| `typing_speed_cps` | 3.193 ± 0.701 | 2.746 ± 0.701 | 0,2 … 6,477 | 0.6751 |
| `keystroke_variability` | 0.252 ± 0.145 | 0.364 ± 0.159 | 0 … 0,912 | 0.7013 |
| `segmented_typing_ratio` | 0.102 ± 0.092 | 0.215 ± 0.126 | 0 … 0,823 | 0.7742 |
| `avg_touch_pressure` | 0.498 ± 0.151 | 0.486 ± 0.175 | 0,02 … 0,961 | 0.5186 |
| `avg_touch_size_px` | 37.987 ± 6.048 | 37.826 ± 8.737 | 8 … 70,8 | 0.5085 |
| `swipe_speed_px_s` | 1.397,939 ± 350.758 | 1.167,332 ± 378.814 | 80 … 3.037,46 | 0.6702 |
| `swipe_directional_variance` | 0.202 ± 0.122 | 0.330 ± 0.148 | 0 … 0,86 | 0.7492 |
| `scroll_speed_avg` | 1.198,235 ± 300.333 | 1.039,177 ± 330.28 | 60 … 2.580,16 | 0.6388 |
| `device_tilt_angle_mean` | 35.148 ± 10.011 | 39.358 ± 13.288 | 0 … 90 | 0.5999 |
| `device_tilt_variability` | 4.060 ± 1.982 | 5.631 ± 2.713 | 0,05 … 19,35 | 0.6765 |
| `gyro_rotation_rate_mean` | 0.152 ± 0.078 | 0.204 ± 0.096 | 0,005 … 0,682 | 0.6602 |
| `accelerometer_jerk_mean` | 1.209 ± 0.5 | 1.594 ± 0.69 | 0,03 … 4,433 | 0.6713 |
| `phone_motion_events` | 3.002 ± 1.739 | 4.504 ± 2.233 | 0 … 17 | 0.6982 |
| `hesitation_count` | 1.514 ± 1.241 | 2.862 ± 1.822 | 0 … 10 | 0.7227 |
| `avg_hesitation_duration_s` | 1.312 ± 1.247 | 2.584 ± 1.839 | 0 … 13,49 | 0.7235 |
| `max_hesitation_duration_s` | 2.355 ± 2.313 | 5.964 ± 4.24 | 0 … 39,01 | 0.7754 |
| `dead_time_periods` | 1.020 ± 1.017 | 2.565 ± 1.787 | 0 … 11 | 0.7681 |
| `total_dead_time_s` | 9.487 ± 11.699 | 27.137 ± 22.651 | 0 … 176,74 | 0.7637 |
| `dead_time_ratio` | 0.072 ± 0.111 | 0.105 ± 0.117 | 0 … 0,92 | 0.6415 |
| `unique_screens_visited` | 6.970 ± 2.462 | 5.745 ± 2.249 | 1 … 26 | 0.6440 |
| `navigation_back_count` | 1.513 ± 1.234 | 2.222 ± 1.543 | 0 … 10 | 0.6345 |
| `screen_transition_time_avg_s` | 2.030 ± 1.444 | 3.246 ± 2.402 | 0,01 … 24,35 | 0.6605 |
| `input_error_count` | 0.818 ± 0.912 | 1.915 ± 1.478 | 0 … 9 | 0.7268 |
| `input_correction_count` | 1.039 ± 1.375 | 2.333 ± 2.322 | 0 … 18 | 0.6770 |
| `amount_field_corrections` | 0.061 ± 0.251 | 0.241 ± 0.514 | 0 … 5 | 0.5746 |
| `beneficiary_field_corrections` | 0.070 ± 0.273 | 0.386 ± 0.666 | 0 … 5 | 0.6187 |
| `copy_paste_events` | 0.402 ± 0.634 | 0.217 ± 0.474 | 0 … 7 | 0.5714 |
| `data_familiarity_score` | 0.799 ± 0.121 | 0.660 ± 0.14 | 0,142 … 0,999 | 0.7767 |
| `doodling_events` | 0.505 ± 0.712 | 0.943 ± 1.003 | 0 … 7 | 0.6245 |
| `session_duration_s` | 181.987 ± 120.738 | 362.046 ± 266.31 | 20 … 2.857,03 | 0.7658 |
| `hour_of_day` | 15.119 ± 5.31 | 13.607 ± 6.222 | 0 … 23 | 0.5600 |
| `is_atypical_hour` | 0.125 ± 0.331 | 0.219 ± 0.413 | 0 / 1 | 0.5469 |
| `phone_call_active` | 0.060 ± 0.237 | 0.568 ± 0.495 | 0 / 1 | 0.7539 |
| `call_overlap_duration_s` | 3.561 ± 21.077 | 143.545 ± 197.555 | 0 … 2.580,49 | 0.7682 |
| `remote_access_tool_detected` | 0.003 ± 0.058 | 0.098 ± 0.297 | 0 / 1 | 0.5471 |
| `suspicious_app_detected` | 0.006 ± 0.078 | 0.072 ± 0.258 | 0 / 1 | 0.5327 |
| `transaction_attempted` | 0.604 ± 0.489 | 0.816 ± 0.387 | 0 / 1 | 0.6063 |
| `transaction_amount_cop` | 351.072 ± 971.571 | 1.718.397 ± 2.955.037 | −1 … 57.387.617 | 0.7500 |
| `is_new_beneficiary` | 0.114 ± 0.318 | 0.543 ± 0.498 | 0 / 1 | 0.7145 |
| `time_to_transaction_s` | 43.390 ± 51.15 | 116.632 ± 98.19 | −1 … 714,55 | 0.7385 |
| `biocatch_risk_score` | 316.627 ± 287.917 | 644.045 ± 312.01 | 0 … 998,98 | 0.7780 |
| `biocatch_genuine_score` | 679.835 ± 290.257 | 403.167 ± 317.448 | 0,02 … 999 | 0.7408 |
| `biocatch_ato_indicator` | 0.075 ± 0.263 | 0.375 ± 0.484 | 0 / 1 | 0.6501 |
| `biocatch_social_eng_indicator` | 0.065 ± 0.247 | 0.554 ± 0.497 | 0 / 1 | 0.7446 |
| `biocatch_bot_indicator` | 0.002 ± 0.045 | 0.001 ± 0.037 | 0 / 1 | 0.5003 |
| `errors_per_minute` | 0.378 ± 0.555 | 0.454 ± 0.501 | 0 … 13,4 | 0.5926 |
| `hesitation_composite` | 0.020 ± 0.032 | 0.032 ± 0.04 | 0 … 0,918 | 0.6369 |
| `days_to_claim` | −0.954 ± 0.828 | 5.182 ± 5.017 | −1 … 53 | 0.8831 † |

† `days_to_claim` es una **columna post-hoc**, no una feature: solo se conoce
después de que la víctima reclamó. Su AUC alta es la definición de la etiqueta,
no una fuga de diseño. `vishing_common.py` ya la descarta vía `POSTHOC_COLS`.

---

## 3. Descripción por familia

### 3.1 Dinámica de tecleo (`keystroke`)

| Columna | Unidad | Significado |
|---|---|---|
| `avg_keyhold_ms` | ms | *Dwell time* medio: cuánto se mantiene pulsada la tecla. Sube en vishing (duda, dictado). Truncada > 40 ms. |
| `avg_interkey_latency_ms` | ms | *Flight time* medio entre pulsaciones. Marginal de vishing lognormal con cola derecha pesada: pausas largas escuchando instrucciones. |
| `typing_speed_cps` | car/s | Velocidad neta. Ancla legítima: 38 WPM ≈ 3,0-3,3 car/s (Aalto University, ~37.000 voluntarios, 2019). |
| `keystroke_variability` | 0-1 | Coeficiente de variación normalizado del ritmo de tecleo. |
| `segmented_typing_ratio` | 0-1 | Fracción del texto introducido en fragmentos separados por pausas. **Es el indicador de *segmented typing* que documenta BioCatch**: el estafador dicta, la víctima teclea a trozos. Es la feature conductual con mayor AUC del dataset (0.7742). |

### 3.2 Dinámica táctil (`touch`)

| Columna | Unidad | Significado |
|---|---|---|
| `avg_touch_pressure` | 0-1 | Presión media normalizada. En vishing **no cambia la media, cambia la forma**: mezcla de presión fuerte (tensión) y toques leves/temblorosos. Por eso su AUC es ≈0.52: el estrés no se manifiesta igual en todos. |
| `avg_touch_size_px` | px | Área de contacto del dedo. Misma media, más varianza en vishing (temblor). AUC ≈0.51, deliberadamente. |
| `swipe_speed_px_s` | px/s | Velocidad media de deslizamiento. |
| `swipe_directional_variance` | 0-1 | Erraticidad direccional de los swipes (*aimless touch movement*). |
| `scroll_speed_avg` | px/s | Velocidad media de scroll. |

### 3.3 Sensores (`motion`)

| Columna | Unidad | Significado |
|---|---|---|
| `device_tilt_angle_mean` | grados 0-90 | Inclinación media del dispositivo. Sostener el teléfono al oído o en postura incómoda durante la llamada. |
| `device_tilt_variability` | grados | Desviación de la inclinación: movimiento nervioso. |
| `gyro_rotation_rate_mean` | rad/s | Rotación media del giroscopio. |
| `accelerometer_jerk_mean` | m/s³ | Derivada de la aceleración: temblor / ansiedad. |
| `phone_motion_events` | conteo | Eventos discretos de movimiento significativo. |

### 3.4 Hesitación (`hesitation`)

| Columna | Unidad | Significado |
|---|---|---|
| `hesitation_count` | conteo | Número de pausas de decisión detectadas. |
| `avg_hesitation_duration_s` | s | Duración media. `0` si `hesitation_count = 0`. |
| `max_hesitation_duration_s` | s | Duración máxima. Se genera como `avg + extra`, **no como sorteo independiente**: garantiza `max ≥ avg` en el 100 % de las filas. Coincide con `avg` cuando `hesitation_count ≤ 1`. |

### 3.5 Tiempo muerto (`dead_time`)

| Columna | Unidad | Significado |
|---|---|---|
| `dead_time_periods` | conteo | Huecos de inactividad dentro de la sesión (*session dead time*: alguien recibe instrucciones fuera de la app). |
| `total_dead_time_s` | s | Suma de esos huecos. Topado al 92 % de `session_duration_s`. |
| `dead_time_ratio` | 0-1 | `total_dead_time_s / session_duration_s`. **Calculado sobre una duración con varianza real** — en la v2 la duración era la constante 1.0 y este ratio no significaba nada. |

### 3.6 Navegación (`navigation`)

| Columna | Unidad | Significado |
|---|---|---|
| `unique_screens_visited` | conteo | Pantallas distintas visitadas. **Baja en vishing**: el estafador dirige por un camino corto y específico. |
| `navigation_back_count` | conteo | Retrocesos: confusión. |
| `screen_transition_time_avg_s` | s | Tiempo medio entre pantallas: espera mientras recibe instrucciones. |

`screens_visited`, `unusual_screen_visits` e `interactions_per_s` **no existen**,
en esta version se determinó que son redundantes y el pipeline las descarta siempre.

### 3.7 Errores y correcciones (`corrections`)

| Columna | Unidad | Significado |
|---|---|---|
| `input_error_count` | conteo | Entradas inválidas / rechazadas. |
| `input_correction_count` | conteo | Borrados y reescrituras. Generada **condicionada a** `input_error_count` (λ = 0,30 + 0,80·errores), no de forma independiente. |
| `amount_field_corrections` | conteo | Correcciones en el campo de monto. `0` si no hubo transacción. |
| `beneficiary_field_corrections` | conteo | Correcciones en el campo de beneficiario; λ ×1,6 si `is_new_beneficiary = 1` (se está confirmando por voz un número de cuenta dictado). |
| `copy_paste_events` | conteo | **Señal invertida a propósito**: 0,40 legítimo → 0,15 vishing. Los datos dictados por voz se teclean, no se pegan del portapapeles. Es una de las variables contraintuitivas del dataset. |

### 3.8 Familiaridad y doodling (`derived`)

| Columna | Unidad | Significado |
|---|---|---|
| `data_familiarity_score` | 0-1 | Fluidez con la que el usuario introduce datos que debería conocer de memoria. Baja en vishing: los datos son dictados. |
| `doodling_events` | conteo | Toques sin propósito mientras espera instrucciones. |

### 3.9 Contexto de sesión (`context`)

| Columna | Unidad | Significado |
|---|---|---|
| `session_duration_s` | s | Duración total. Lognormal, mediana 151 s (legítimo) / 293 s (vishing). σ global = 137,6 s: **tiene varianza real**, a diferencia de la v2. |
| `hour_of_day` | 0-23 | Hora local. Consistente con `session_timestamp` (validado). |
| `is_atypical_hour` | 0/1 | `1` si `hour_of_day ∈ {22,23,0,1,2,3,4,5}`. Es un **bucket determinista** de `hour_of_day`, y está bien que lo sea: no es una copia disfrazada de otra señal de comportamiento. |
| `phone_call_active` | 0/1 | Llamada telefónica activa **detectada en el mismo dispositivo** durante la sesión. Ver §4.1: no es "hubo llamada", es "la app pudo verla". |
| `call_overlap_duration_s` | s | Segundos de solapamiento entre la llamada y la sesión. `0` ⇔ `phone_call_active = 0`. En legítimo: llamada breve y ajena (media 59,6 s condicionada a que haya llamada). En vishing: 70-100 % de la sesión. **Duración continua con varianza propia, no una copia de la binaria** (r = 0,60 con `phone_call_active`, no 1,0000). |
| `remote_access_tool_detected` | 0/1 | Herramienta de acceso remoto detectada. |
| `suspicious_app_detected` | 0/1 | App sospechosa instalada. |

### 3.10 Transacción (`transaction`)

| Columna | Unidad | Significado |
|---|---|---|
| `transaction_attempted` | 0/1 | Hubo intento de transferencia. Muchas sesiones legítimas son solo consulta de saldo. |
| `transaction_amount_cop` | COP entero | Monto. `−1` si no hubo transacción. Mediana 248.000 COP (legítimo) / 1.091.000 COP (vishing). |
| `is_new_beneficiary` | 0/1 | Beneficiario nunca usado antes. `0` si no hubo transacción. |
| `time_to_transaction_s` | s | Segundos desde el inicio de la sesión hasta la confirmación. `−1` si no hubo transacción. **Continua y variable**, no `10 × transaction_attempted`. |

### 3.11 Scores simulados de BioCatch

Simulan la salida de un motor de riesgo de terceros. **No son funciones
deterministas** de las columnas de comportamiento: incorporan un error de modelo
propio, de forma que existen sesiones de vishing con score bajo (no detectadas) y
sesiones legítimas con score alto (falsos positivos reales).

| Columna | Rango | Significado |
|---|---|---|
| `biocatch_risk_score` | 0-999 | Agregación ponderada de 23 señales + ruido gaussiano, mapeada a la escala 0-999 por percentil elevado a γ=2. Legítimas: mediana 233, 65 % por debajo de 400. Vishing: mediana 738, 65 % por encima de 550. Ambas con cola hasta el extremo contrario. |
| `biocatch_genuine_score` | 0-999 | Score de autenticidad. Relacionado inversamente con el anterior (**r = −0,60**, no `1000 − risk`): comparte el 70 % del error del motor y añade un término independiente. |
| `biocatch_ato_indicator` | 0/1 | Umbral ruidoso sobre `biocatch_risk_score`. 7,5 % en legítimas, 37,5 % en vishing. |
| `biocatch_social_eng_indicator` | 0/1 | Umbral ruidoso sobre un sub-score específico de ingeniería social (tecleo segmentado, llamada, tiempo muerto, familiaridad, beneficiario nuevo). Más selectivo para vishing que el de ATO. |
| `biocatch_bot_indicator` | 0/1 | **Tasa base idéntica en ambas clases (≈0,2 %) y no correlacionada con `is_vishing`** (AUC = 0.5003). Es conceptualmente obligatorio: una sesión de vishing es una persona real usando su propio dispositivo, no un bot. |

**Los cinco scores son fuga directa** si se usan como features para predecir
`is_vishing`: son la salida del motor que el modelo pretende replicar.
`vishing_common.py` ya los aísla en `LEAKAGE_COLS`.

### 3.12 Variables derivadas

| Columna | Fórmula |
|---|---|
| `errors_per_minute` | `input_error_count / (session_duration_s / 60)` |
| `hesitation_composite` | `(hesitation_count × avg_hesitation_duration_s) / session_duration_s` |
| `dead_time_ratio` | `total_dead_time_s / session_duration_s` |

Las tres dependen aritméticamente de `session_duration_s`. En esta version esa variable
ya tiene varianza real, así que las tres significan lo que su nombre dice
(en la v2 no, porque la duración era la constante 1.0).

### 3.13 Etiquetas

| Columna | Valores | Significado |
|---|---|---|
| `is_vishing` | 0/1 | **Target.** Exactamente 5.000 unos. |
| `days_to_claim` | int, `−1` = sin reclamo | Días entre la sesión y el reclamo. Presente en el 77,1 % de las sesiones de vishing (no todas las víctimas reclaman) y en el 0,42 % de las legítimas. Rango 1-53. **Post-hoc: nunca es feature.** |
| `claim_category` | `""`, `vishing_confirmado`, `ato_no_vishing`, `disputa_no_fraude` | Motivo del reclamo. `""` ⇔ `days_to_claim = −1`. Distribución: 95.747 vacías, 3.857 `vishing_confirmado`, 240 `ato_no_vishing`, 156 `disputa_no_fraude`. Los dos últimos solo aparecen en sesiones legítimas: son el ruido intencional que impide que `claim_category` sea un alias del target. |

---

## 4. Notas de diseño que hay que conocer antes de modelar

### 4.1 `phone_call_active` no es "hubo llamada"

En el diccionario original del proyecto, `phone_call_active` estaba presente en
el ~85 % de las sesiones de vishing confirmadas. En esta version la prevalencia
observada es del **56,8 %**, y la diferencia es deliberada.

`phone_call_active` es una **detección a nivel de dispositivo**: solo vale 1 si
la llamada ocurre en el mismo teléfono desde el que se opera la app. Quedan
fuera las víctimas que hablan por un segundo equipo, por el altavoz de un fijo,
o cuya llamada terminó antes del pago. Con el 85 % literal, esta única variable
alcanza **AUC 0,90** y convierte el dataset en un problema trivial y
autoexplicativo — exactamente lo contrario de lo que el POC necesita demostrar.

Para una binaria, `AUC = 0,5 + (p_vishing − p_legítimo)/2`. Con `p_legítimo` en
el 6 %, cualquier `p_vishing` por encima del 66 % rompe el techo de 0,80.


### 4.2 Las clases se solapan a propósito

- **15 % de las sesiones de vishing son "sofisticadas"**: el ataque activa poco
  los indicadores y la sesión queda dentro de la nube legítima. Son falsos
  negativos por diseño.
- **2,5 % de las sesiones legítimas están "confundidas"**: usuario estresado,
  adulto mayor con tecleo lento, alguien realmente hablando con un familiar.
  Muestran varios indicadores elevados sin ser fraude.

### 4.3 Estructura de correlación

Cada cliente tiene rasgos latentes estables (velocidad de tecleo, firmeza del
pulso, pericia con la app, franja horaria habitual, propensión a hablar por
teléfono). El 55 % de la varianza de cada variable conductual es atribuible al
cliente. **Consecuencia práctica: el split debe agruparse por `customer_id`.**
Un split por fila filtra la identidad del usuario entre particiones y sobrestima
las métricas.

### 4.4 Columnas que nunca deben entrar al modelo

| Grupo | Columnas |
|---|---|
| Identificadores | `row_id`, `session_id`, `customer_id`, `session_timestamp` |
| Contexto no discriminante | `os_type`, `app_version` (misma distribución en ambas clases) |
| Fuga directa | los cinco `biocatch_*` |
| Post-hoc | `days_to_claim`, `claim_category` |

---

## 5. Trazabilidad

| Archivo | Contenido |
|---|---|
| `generar_dataset_sintetico.py` | Script generador, reproducible con semilla 42 |
| `biocatch_sinthetic_data.csv` | El dataset (100.000 × 58) |
| `diccionario_datos_biocatch_sintetico.md` | Este documento |
| `notas_metodologicas_generacion.md` | Distribución y parámetros reales por columna, desviaciones respecto del diseño original y su justificación |
| `reporte_validacion.md` / `.json` | Las 8 validaciones obligatorias, con cifras |
