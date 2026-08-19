# Reporte de validación — `biocatch_sinthetic_data_v3.csv`

Filas: **100,000** · Columnas: **58** · Semilla: **42**

Resultado global: **PASA**


## 1. Columnas constantes

PASA — columnas numéricas con `nunique() <= 1`: **0**

## 2. Columnas duplicadas (|r| ≥ 0.999)

PASA — pares detectados: **0**

Cinco correlaciones más altas del dataset (referencia):

| Columna A | Columna B | r |
|---|---|---|
| `dead_time_periods` | `total_dead_time_s` | 0.8532 |
| `input_error_count` | `input_correction_count` | 0.7374 |
| `avg_hesitation_duration_s` | `max_hesitation_duration_s` | 0.7357 |
| `total_dead_time_s` | `dead_time_ratio` | 0.7156 |
| `is_vishing` | `days_to_claim` | 0.7005 |

## 3. Balance de clases

PASA — vishing: **5,000** · legítimas: **95,000** (5.00%)

## 4. AUC univariado frente a `is_vishing`

PASA — AUC máximo entre features: **0.8380** · techo vigente: **0.85** · features por encima del techo: **0**

Techo anterior (v3.0): 0.80, con máximo alcanzado 0.7780. Referencia v2 (dataset canónico anterior): máximo 0.6890 (`segmented_typing_ratio`).

Parámetros de separabilidad de esta corrida: `ESCALA_GANANCIA` = 1.24, `BETA_INTENSIDAD_BURDO` = (6.0, 1.7), `P_ATAQUE_SOFISTICADO` = 0.15.

| # | Variable | AUC |
|---|---|---|
| 1 | `max_hesitation_duration_s` | 0.8380 |
| 2 | `data_familiarity_score` | 0.8330 |
| 3 | `biocatch_risk_score` | 0.8145 |
| 4 | `segmented_typing_ratio` | 0.8129 |
| 5 | `swipe_directional_variance` | 0.8027 |
| 6 | `hesitation_count` | 0.7973 |
| 7 | `transaction_amount_cop` | 0.7927 |
| 8 | `dead_time_periods` | 0.7899 |
| 9 | `call_overlap_duration_s` | 0.7880 |
| 10 | `total_dead_time_s` | 0.7853 |
| 11 | `session_duration_s` | 0.7840 |
| 12 | `input_error_count` | 0.7807 |
| 13 | `biocatch_social_eng_indicator` | 0.7782 |
| 14 | `biocatch_genuine_score` | 0.7780 |
| 15 | `time_to_transaction_s` | 0.7754 |
| 16 | `phone_call_active` | 0.7723 |
| 17 | `avg_hesitation_duration_s` | 0.7712 |
| 18 | `phone_motion_events` | 0.7630 |
| 19 | `keystroke_variability` | 0.7497 |
| 20 | `is_new_beneficiary` | 0.7352 |
| 21 | `input_correction_count` | 0.7284 |
| 22 | `typing_speed_cps` | 0.7264 |
| 23 | `device_tilt_variability` | 0.7230 |
| 24 | `swipe_speed_px_s` | 0.7174 |
| 25 | `avg_keyhold_ms` | 0.7155 |
| 26 | `accelerometer_jerk_mean` | 0.7075 |
| 27 | `gyro_rotation_rate_mean` | 0.7015 |
| 28 | `hesitation_composite` | 0.7000 |
| 29 | `screen_transition_time_avg_s` | 0.6988 |
| 30 | `biocatch_ato_indicator` | 0.6847 |
| 31 | `scroll_speed_avg` | 0.6835 |
| 32 | `beneficiary_field_corrections` | 0.6771 |
| 33 | `navigation_back_count` | 0.6709 |
| 34 | `doodling_events` | 0.6703 |
| 35 | `unique_screens_visited` | 0.6673 |
| 36 | `avg_interkey_latency_ms` | 0.6522 |
| 37 | `dead_time_ratio` | 0.6518 |
| 38 | `errors_per_minute` | 0.6291 |
| 39 | `amount_field_corrections` | 0.6191 |
| 40 | `transaction_attempted` | 0.6162 |
| 41 | `device_tilt_angle_mean` | 0.6073 |
| 42 | `copy_paste_events` | 0.5745 |
| 43 | `hour_of_day` | 0.5594 |
| 44 | `remote_access_tool_detected` | 0.5521 |
| 45 | `is_atypical_hour` | 0.5463 |
| 46 | `suspicious_app_detected` | 0.5350 |
| 47 | `avg_touch_pressure` | 0.5154 |
| 48 | `avg_touch_size_px` | 0.5035 |
| 49 | `biocatch_bot_indicator` | 0.5005 |

**Columnas post-hoc excluidas de la regla** (no son features: solo se conocen después de que la víctima reclamó, y `vishing_common.py` ya las descarta vía `POSTHOC_COLS`):

| Columna | AUC |
|---|---|
| `days_to_claim` | 0.8846 |

## 5. Sesiones por cliente

PASA

| Métrica | Valor |
|---|---|
| clientes | 38000 |
| media | 2.632 |
| mediana | 2.0 |
| minimo | 1 |
| p99 | 10.0 |
| maximo | 22 |
| clientes_victima | 4893 |
| max_vishing_por_cliente | 2 |

## 6. Co-ocurrencia dentro de `is_vishing = 1`

PASA — todas positivas, ninguna en 1.0 (rango 0.0789 – 0.1753)

| | `phone_call_active` | `segmented_typing_ratio` | `hesitation_count` | `is_new_beneficiary` | `session_duration_s` |
|---|---|---|---|---|---|
| `phone_call_active` | 1.0000 | 0.1316 | 0.1375 | 0.1319 | 0.0891 |
| `segmented_typing_ratio` | 0.1316 | 1.0000 | 0.1753 | 0.1289 | 0.1476 |
| `hesitation_count` | 0.1375 | 0.1753 | 1.0000 | 0.1603 | 0.1183 |
| `is_new_beneficiary` | 0.1319 | 0.1289 | 0.1603 | 1.0000 | 0.0789 |
| `session_duration_s` | 0.0891 | 0.1476 | 0.1183 | 0.0789 | 1.0000 |

## 7. Rangos y sanidad básica

PASA

| Comprobación | Resultado |
|---|---|
| `ratios_en_0_1` | True |
| `duraciones_no_negativas` | True |
| `max_hes_ge_avg_hes` | True |
| `dead_time_le_duracion` | True |
| `overlap_le_duracion` | True |
| `overlap_cero_sin_llamada` | True |
| `ttt_centinela_coherente` | True |
| `monto_centinela_coherente` | True |
| `claim_category_coherente` | True |
| `session_id_unico` | True |
| `row_id_unico_y_completo` | True |
| `sin_nulos` | True |
| `sd_session_duration_s` | 141.337 |
| `nunique_session_duration_s` | 36380 |
| `hora_coincide_con_timestamp` | True |

## 8. Persistencia

Este reporte y su versión JSON (`reporte_validacion.json`) se escriben junto al dataset en `remediacion/data/`.
