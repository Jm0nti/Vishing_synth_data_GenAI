# Notas metodológicas — generación del dataset sintético v3

Documento de trazabilidad de `generar_dataset_sintetico.py`. Registra **qué
distribución y qué parámetros se usaron realmente** en cada columna, en qué se
apartan del diseño de partida y por qué, y qué supuestos son míos frente a los
que están anclados en literatura citable.

---

## 1. Reproducibilidad

| Elemento | Valor |
|---|---|
| Semilla global | `42` (parámetro `--seed`, por defecto `SEED = 42`) |
| Generador | `numpy.random.default_rng(seed)`, un único stream para todo el script |
| Python | 3.11.15 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scipy | 1.17.1 |
| Tiempo de generación | 4,4 s para 100.000 × 58 |
| Comando | `python generar_dataset_sintetico.py --salida .` |

Verificado: dos ejecuciones independientes con la misma semilla producen CSV
byte a byte idénticos (`md5 = 615235ef3ab04f37ca2243874a231e6d`).

No se usa ninguna librería de modelado generativo (CTGAN o similar). La
generación es paramétrica a propósito: es la única forma de tener control
verificable sobre cada marginal y de poder auditar el resultado.

---

## 2. Arquitectura del generador

### 2.1 Cópula gaussiana

Cada variable continua se genera en dos pasos:

1. **Espacio latente.** Una `z ~ N(0,1)` por variable y sesión, construida como

   ```
   z_j = √ρ · t_j[cliente]  +  √(1−ρ) · ( b_j · S_sesión + √(1−b_j²) · η )
   t_j = a_j · F[cliente]   + √(1−a_j²) · ν
   ```

   con `ρ = 0.55`, `F` = tres factores ortogonales por cliente
   (**velocidad**, **pulso**, **pericia**), `S` = un factor de estado por sesión
   (prisa, distracción) y `a_j`, `b_j` las cargas de la tabla `CARGAS` /
   `CARGA_ESTADO` del script. Por construcción `t_j` y `z_j` son N(0,1) exactas.

2. **Marginal.** `x = F⁻¹(Φ(z))`, con `F⁻¹` la función cuantil de la marginal
   objetivo (`scipy.stats.<dist>.ppf`).

Ventaja frente a sortear cada columna por separado: la marginal pedida se
respeta **exactamente**, y al mismo tiempo se obtiene correlación intra-cliente
(55 % de la varianza) e intra-sesión sin tener que inventar una matriz de
covarianzas a mano.

**Consecuencia operativa**: el split debe agruparse por `customer_id`.

### 2.2 El ataque como interpolación, no como sustitución

Para una sesión de vishing con activación `w ∈ [0,1]`:

```
x = (1 − w) · F_legítimo⁻¹(u)  +  w · F_vishing⁻¹(u)          (mismo u)
```

Usar el mismo `u` en ambas marginales preserva el rango relativo del cliente: un
mecanógrafo rápido sigue siendo relativamente rápido bajo coacción. Para las
variables lognormales (`session_duration_s`, `transaction_amount_cop`,
`avg_interkey_latency_ms` no, ver §4) la mezcla es **geométrica**, no
aritmética: interpolar 240.000 COP con 3.200.000 COP en escala lineal aplasta
el extremo bajo.

Para los conteos, lo que se interpola es la tasa:
`λ = λ_legítimo^(1−w) · λ_vishing^w`, y luego `x = Poisson.ppf(u, λ)`.

Para las binarias, lo que se interpola es la probabilidad:
`p = p_legítimo + w · (p_vishing_max − p_legítimo)`.

### 2.3 La latente de sofisticación

```
intensidad = 0.15 · Beta(2, 6)   +   0.85 · Beta(5, 2)
                 (media ≈0.25)          (media ≈0.71)
```

El 15 % de la izquierda son los **ataques sofisticados**: activan poco los
indicadores y la sesión queda dentro de la nube legítima. Media global de la
intensidad ≈ 0,64.

La activación efectiva por variable añade ruido idiosincrásico y una ganancia:

```
w_j = clip( GANANCIA[j] · ( intensidad + N(0, 0.12) ), 0, 1 )
```

Sin el término de ruido todos los indicadores de una sesión se moverían en
bloque y la correlación entre ellos dentro de la clase vishing sería ≈1,0, lo
que no se parece a nada real. Con `SD_RUIDO_ACTIVACION = 0.12` las
correlaciones intra-clase quedan en 0,09-0,16 (validación §6).

### 2.4 Ruido intencional en la clase legítima

El 2,5 % de las sesiones legítimas (`P_LEGIT_CONFUNDIDA = 0.025`) recibe la
misma maquinaria con `intensidad = 0.85 · Beta(2.5, 3.0)` (media ≈0,39):
usuario estresado, adulto mayor con tecleo lento, alguien realmente hablando con
un familiar mientras usa el banco. Son falsos positivos creíbles y no "vishing
con otra etiqueta", porque la activación es moderada y el ruido por variable
hace que solo se enciendan algunos indicadores.

---

## 3. Población y asignación de sesiones

| Elemento | Implementación |
|---|---|
| Clientes | 38.000 |
| Sesiones por cliente | `1 + Poisson(λ_i)`, `λ_i ~ Gamma(1.15, 1.42)` (mezcla Gamma-Poisson = binomial negativa), topada en 22 |
| Ajuste al total | Bucle de ±1 sobre clientes al azar hasta sumar exactamente 100.000, sin tocar la forma de la distribución |
| Resultado | media 2,632 · mediana 2 · p99 = 10 · máx. 22 |
| Víctimas | 4.893 clientes; el 6 % de las víctimas con ≥3 sesiones sufre 2 ataques (`k = 2`), el resto 1. Máximo observado: 2 sesiones de vishing por cliente |
| Franja horaria | Cada cliente favorece una de las 4 componentes diurnas (peso +0,45 sobre la base) |
| Propensión a llamada | `Beta(1.3, 26.0)` por cliente (media ≈0,048): hay gente que siempre habla mientras usa el banco |

**Supuesto propio**: la reincidencia del 6 % y el tope de 22 sesiones no vienen
de ninguna fuente; son elecciones de diseño para que la cola sea realista sin
absurdos (la v1 tenía el problema opuesto).

---

## 4. Parámetros efectivos por columna

`GANANCIA` es el multiplicador de la activación (§2.3). **Ganancia 1,00
significa que la marginal de vishing del diseño original se alcanza cuando
`intensidad = 1`**; ganancia 0,42 significa que ni el ataque más burdo llega
más allá del 42 % del camino hacia esa marginal.

| Columna | Marginal legítima | Marginal vishing (pura) | Ganancia | AUC resultante |
|---|---|---|---|---|
| `avg_keyhold_ms` | `Normal(110, 18)`, truncada >40 | `Normal(145, 30)` | 0,62 | 0,6739 |
| `avg_interkey_latency_ms` | `Normal(220, 45)`, truncada >55 | `LogNormal(σ=0.55, mediana 293)` → media ≈340 | 0,70 | 0,6358 |
| `typing_speed_cps` | `Normal(3.2, 0.7)`, clip [0.2, 7] | `Normal(1.7, 0.6)` | 0,46 | 0,6751 |
| `keystroke_variability` | `Beta(2, 6)` | `Beta(4, 4)` | 0,68 | 0,7013 |
| `segmented_typing_ratio` | `Beta(1, 9)` | mezcla 60 % `Beta(6,2)` + 40 % `Beta(2,5)` | 0,38 | 0,7742 |
| `avg_touch_pressure` | `Beta(5, 5)` | mezcla 45 % `Beta(6,4)` + 55 % `Beta(3,6)` | 1,00 | 0,5186 |
| `avg_touch_size_px` | `Normal(38, 6)` | `Normal(38, 10)` | 1,00 | 0,5085 |
| `swipe_speed_px_s` | `Normal(1400, 350)` | `Normal(900, 400)` | 0,70 | 0,6702 |
| `swipe_directional_variance` | `Beta(2, 8)` | `Beta(4, 4)` | 0,66 | 0,7492 |
| `scroll_speed_avg` | `Normal(1200, 300)` | `Normal(850, 350)` | 0,72 | 0,6388 |
| `device_tilt_angle_mean` | `Normal(35, 10)`, clip [0,90] | `Normal(42, 15)` | 1,00 | 0,5999 |
| `device_tilt_variability` | `Normal(4, 2)` | `Normal(9, 4)` | 0,50 | 0,6765 |
| `gyro_rotation_rate_mean` | `Normal(0.15, 0.08)` | `Normal(0.28, 0.12)` | 0,62 | 0,6602 |
| `accelerometer_jerk_mean` | `Normal(1.2, 0.5)` | `Normal(2.3, 1.0)` | 0,55 | 0,6713 |
| `phone_motion_events` | `Poisson(3)` | `Poisson(8)` | 0,62 | 0,6982 |
| `hesitation_count` | `Poisson(1.5)` | `Poisson(6)` | 0,70 | 0,7227 |
| `avg_hesitation_duration_s` | `Gamma(2, 0.8)` | `Gamma(3, 1.5)` | 0,60 | 0,7235 |
| `max_hesitation_duration_s` | `avg + Gamma(1.8, 1.2)` | `avg + Gamma(2.2, 3.4)` | 0,60 | 0,7754 |
| `dead_time_periods` | `Poisson(1)` | `Poisson(4)` | 1,00 | 0,7681 |
| `total_dead_time_s` | `Gamma(2·k, 4.5)` | `Gamma(2.5·k, 4.5)` | 1,00 | 0,7637 |
| `unique_screens_visited` | `1 + Poisson(6)` | `1 + Poisson(4)` | 0,90 | 0,6440 |
| `navigation_back_count` | `Poisson(1.5)` | `Poisson(3.0)` | 0,85 | 0,6345 |
| `screen_transition_time_avg_s` | `Gamma(2, 1)` | `Gamma(2, 2.5)` | 0,62 | 0,6605 |
| `input_error_count` | `Poisson(0.8)` | `Poisson(3.5)` | 0,85 | 0,7268 |
| `input_correction_count` | `Poisson(0.30 + 0.80·errores)` | idem × 1,55 | 0,60 | 0,6770 |
| `amount_field_corrections` | `Poisson(0.10)` | `Poisson(0.90)` | 0,70 | 0,5746 |
| `beneficiary_field_corrections` | `Poisson(0.10)` | `Poisson(1.10)`, ×1,6 si beneficiario nuevo | 0,70 | 0,6187 |
| `copy_paste_events` | `Poisson(0.40)` | `Poisson(0.15)` ← **invertida** | 1,00 | 0,5714 |
| `data_familiarity_score` | `Beta(8, 2)` | `Beta(2, 5)` | 0,42 | 0,7767 |
| `doodling_events` | `Poisson(0.5)` | `Poisson(2.5)` | 0,58 | 0,6245 |
| `session_duration_s` | `LogNormal(σ=0.60, mediana 150)` | `LogNormal(σ=0.62, mediana 420)` | 1,00 | 0,7658 |
| `transaction_amount_cop` | `LogNormal(σ=1.30, mediana 240.000)` | `LogNormal(σ=0.95, mediana 3.200.000)` | 0,85 | 0,7500 |
| `time_to_transaction_s` | `Gamma(2, 40)` | `Gamma(3, 90)` | 0,62 | 0,7385 |
| `call_overlap_duration_s` | `Gamma(1.8, 25)` si hay llamada | `U(0.70, 1.00)·duración + N(0,0.05)` | 1,00 | 0,7682 |

### Binarias (calibradas por probabilidad, no por ganancia)

Para una binaria, `AUC = 0.5 + (p_vishing − p_legítimo)/2`, así que se controla
directamente la probabilidad. `p_max` es la probabilidad con `intensidad = 1`;
la prevalencia observada corresponde a la intensidad media ≈0,64.

| Columna | p legítimo | p vishing (máx.) | p vishing (observada) | AUC |
|---|---|---|---|---|
| `phone_call_active` | 0,055 base + propensión personal | 0,85 | 0,568 | 0,7539 |
| `transaction_attempted` | 0,60 | 0,94 | 0,816 | 0,6063 |
| `is_new_beneficiary` | 0,18 | 0,90 | 0,543 | 0,7145 |
| `remote_access_tool_detected` | 0,002 | 0,16 | 0,098 | 0,5471 |
| `suspicious_app_detected` | 0,005 | 0,11 | 0,072 | 0,5327 |
| `biocatch_bot_indicator` | 0,002 | 0,002 | 0,001 | 0,5003 |

---

## 5. Desviaciones respecto del diseño de partida, y por qué

El diseño original marcaba distribuciones concretas (§4 del prompt) y a la vez
exigía que **ninguna variable superase AUC 0,80** (§6.4). Cuando las dos cosas
entraban en conflicto, mandó la restricción de AUC, que es la que preserva la
utilidad del dataset. Todas las desviaciones son de la misma naturaleza: la
marginal de vishing "de libro" se alcanza solo parcialmente.

### 5.1 `phone_call_active`: 85 % → 56,8 % (la desviación importante)

Con `p_vishing = 0.85` y `p_legítimo = 0.055`, esta sola variable alcanza
**AUC = 0,90**. El dataset se vuelve autoexplicativo y deja de representar el
problema: BioCatch insiste en que este fraude es difícil precisamente porque la
persona legítima está logueada desde su propio dispositivo, en su ubicación
correcta, haciendo una transferencia autorizada.

Reinterpretación adoptada, documentada en el código y en el diccionario:
`phone_call_active` no es "hubo llamada", es **"la app detectó una llamada en
el mismo dispositivo"**. Quedan fuera las víctimas que hablan por un segundo
teléfono, por el altavoz de un fijo, o cuya llamada terminó antes del pago. Con
`p_max = 0.85` y la intensidad media de 0,64, la prevalencia observada baja a
56,8 % y el AUC a 0,7539.

El 85 % del diccionario original sigue siendo el valor de la marginal pura
(`P_LLAMADA_MAX_VISH = 0.85`): se alcanza en las sesiones con intensidad ≈ 1.

### 5.2 Medias de vishing atenuadas

Estas columnas no llegan al valor "de libro" porque su ganancia es < 1:

| Columna | Media vishing del diseño | Media vishing observada | Motivo |
|---|---|---|---|
| `data_familiarity_score` | ≈0,29 | 0,660 | `Beta(8,2)` vs `Beta(2,5)` casi no se solapan: a ganancia 1 el AUC es ≈0,93 |
| `segmented_typing_ratio` | ≈0,55 | 0,215 | Es la señal clave; a ganancia 0,62 daba AUC 0,83 |
| `avg_interkey_latency_ms` | ≈340 ms | 277 ms | — |
| `typing_speed_cps` | 1,7 | 2,75 | — |
| `session_duration_s` (mediana) | 420 s | 293 s | — |
| `transaction_amount_cop` (mediana) | 2-4 M COP | 1,09 M COP | Con el centinela `−1` para las sesiones sin transacción, el AUC de esta columna arrastra también el efecto de `transaction_attempted` |
| `total_dead_time_s` | ≈45 s | 27 s | — |

Como compensación parcial en el monto, la σ de la lognormal legítima se subió de
1,10 a **1,30**: un mayor solapamiento permite mantener una brecha de medianas
más grande con el mismo AUC. La cola legítima llega a ~57 M COP, lo que es
razonable para transferencias legítimas grandes (vivienda, vehículo).

### 5.3 `biocatch_risk_score`: escala por percentil en vez de logística

El diseño pedía una logística sobre la agregación ponderada. Con el ruido
necesario para mantener el AUC en 0,78, la logística **se satura**: el 10 % de
las sesiones legítimas quedaba pegado a 999, lo que no se parece a la salida de
un motor de riesgo real.

Se sustituyó por `risk = 999 · percentil(s_obs)^γ` con `γ = 2` (`GAMMA_RIESGO`).
Es una transformación monótona, así que **el AUC no cambia**, pero la forma sí:

| | mediana | % por debajo de 400 | % por encima de 550 |
|---|---|---|---|
| Legítimas | 233 | 65,3 % | — |
| Vishing | 738 | — | 65,1 % |

Ambas colas llegan al extremo contrario (vishing con score ≈0 = no detectado;
legítimas con score ≈999 = falso positivo), que es lo que pedía el diseño.

`biocatch_genuine_score` comparte el 70 % del error del score de riesgo (es el
mismo motor mirando la misma sesión) más un término independiente: la
correlación entre ambos queda en **−0,60**, no en −1.

### 5.4 `days_to_claim` queda por encima de 0,80 y es correcto que así sea

AUC = 0,8831. Bajarla a 0,80 exigiría que menos del 60 % de las víctimas
reclamara, en contra del 70-85 % del diseño. Pero `days_to_claim` **no es una
feature**: solo se conoce después del reclamo, y `vishing_common.py` ya la
descarta vía `POSTHOC_COLS`. El script la excluye explícitamente de la regla
de las "variables en rojo" y la reporta aparte.

### 5.5 Convenciones de cero frente a centinela

El diseño pedía `−1` como centinela para "no aplica" en variables continuas
condicionadas. Se aplicó a `transaction_amount_cop`, `time_to_transaction_s` y
`days_to_claim`. **No** se aplicó a `call_overlap_duration_s`,
`avg_hesitation_duration_s`, `max_hesitation_duration_s`, `total_dead_time_s`,
`amount_field_corrections`, `beneficiary_field_corrections` ni
`is_new_beneficiary`: en esos casos el cero es factualmente correcto (no hubo
llamada ⇒ el solapamiento es cero segundos), y meter un `−1` habría roto las
comprobaciones de rango de la validación §7 sin ganar nada.

### 5.6 Sin columna `device_type`

El dataset es 100 % móvil. Una columna constante no aporta información y fue uno
de los hallazgos de la auditoría sobre el trabajo anterior. Si el pipeline filtra con `MOBILE_ONLY`, ese
filtro es ahora un no-op sobre este CSV (no falla: `vishing_common.py` intersecta
las listas de columnas con las presentes).

---

## 6. Los tres bugs del trabajo anterior: cómo se evitan y cómo se comprueba

| Bug | v2 | v3 | Comprobación automática |
|---|---|---|---|
| #1 `session_duration_s` constante = 1.0 | `nunique = 1` | LogNormal con σ = 137,6 s, 36.217 valores distintos | Validación 1 (columnas constantes) + `sd_session_duration_s` en la validación 7 |
| #2 `call_overlap_duration_s` = `phone_call_active` | r = 1,0000 fila a fila | Duración continua condicionada: `Gamma(1.8, 25)` en legítimas con llamada, 70-100 % de la sesión en vishing. r = 0,60 | Validación 2 (ningún par con \|r\| ≥ 0,999) + `overlap_cero_sin_llamada` y `overlap_le_duracion` en la 7 |
| #3 `time_to_transaction_s` = `10 × transaction_attempted` | igualdad exacta | `Gamma(2, 40)` / `Gamma(3, 90)`, topada al 97 % de la duración, `−1` si no hay transacción. 17.998 valores distintos | Validación 2 + `ttt_centinela_coherente` en la 7 |

Bug adicional de la v1, también cubierto: `row_id` es una **columna real**, no
el índice de pandas, así que sobrevive a `to_csv(index=False)`.

---

## 7. Anclaje de cada supuesto

### 7.1 Anclado en literatura citada

| Parámetro | Fuente |
|---|---|
| `typing_speed_cps` legítimo ≈ 3,2 car/s | Aalto University con datos de TypingMaster.com, ~37.000 voluntarios, 2019: 38 WPM medios tecleando con dos pulgares en smartphone; conversión ≈5 caracteres + espacio por palabra → 3,0-3,3 car/s |
| Prevalencia de vishing 5 % | Orden de magnitud coherente con el volumen de fraude APP reportado por UK Finance (168.376 incidentes web+móvil en 2019, £379,1 M) sobre el total de sesiones de banca digital |
| Presencia telefónica como eje del fraude | FTC: 77 % de las quejas de fraude reportadas involucran contacto telefónico |
| Catálogo de indicadores (segmented typing, dead time, doodling, hesitación, movimiento del dispositivo, presión/coacción) | BioCatch, data sheet *"Using Behavioural Biometrics to Combat Social Engineering Voice Scams and App Fraud"* |
| `phone_call_active` en ~85 % de las sesiones de vishing confirmadas | Diccionario de datos original del proyecto (ver §5.1 para la reinterpretación) |
| Ninguna variable univariada por encima de 0,80 | Auditoría interna del dataset v2: máximo medido 0,6890 (`segmented_typing_ratio`) |

### 7.2 Rangos orientativos de literatura, no cifras citables

`avg_keyhold_ms` (80-160 ms) y `avg_interkey_latency_ms` (150-350 ms) provienen
de la literatura general de *keystroke dynamics*, que cubre teclado físico y
táctil sin una cifra pública única y autorizada para banca móvil. Los parámetros
usados —`Normal(110, 18)` y `Normal(220, 45)`— caen dentro de esos rangos, pero
**no deben citarse como si vinieran de una fuente concreta**.

### 7.3 Supuestos propios (marcados en el código con "SUPUESTO PROPIO")

Ninguno de estos tiene respaldo bibliográfico. Son elecciones de diseño mías,
razonadas desde UX/HCI o desde la coherencia interna del dataset:

- `swipe_speed_px_s` ≈ 800-2.500 px/s y `scroll_speed_avg` en el mismo orden.
- `session_duration_s` legítima: mediana 150 s, masa entre 40 y 600 s (la sesión
  de banca móvil es corta y orientada a una tarea puntual).
- Las cuatro componentes horarias de uso (9:00, 13:00, 18:20, 21:20) y sus pesos.
- La estacionalidad semanal (fin de semana al 60-78 % del volumen laborable).
- `RHO_CLIENTE = 0.55` (proporción de la varianza conductual atribuible al
  cliente). No existe una cifra pública de estabilidad intra-sujeto para
  biometría conductual en banca móvil.
- Todas las cargas factoriales de `CARGAS` y `CARGA_ESTADO`.
- El 6 % de víctimas reincidentes y el tope de 22 sesiones por cliente.
- El tope del 92 % de la duración para `total_dead_time_s` y del 97 % para
  `time_to_transaction_s`.
- La reinterpretación de `phone_call_active` como detección en dispositivo.
- **Toda la tabla `GANANCIA`**: es el resultado de calibrar iterativamente hasta
  que ninguna feature superase 0,80, no de ninguna fuente.
- Los pesos de `PESOS_RIESGO` y `PESOS_SOCIAL`, los umbrales de los indicadores
  binarios de BioCatch y `GAMMA_RIESGO = 2`.
- La distribución de `claim_category` entre `ato_no_vishing` (60 %) y
  `disputa_no_fraude` (40 %).

### 7.4 Contraintuición deliberada

`copy_paste_events` es **más bajo** en vishing (0,40 → 0,15). Los datos dictados
por voz se teclean, no se pegan del portapapeles. Es la única variable del
dataset cuyo signo va en contra de la intuición de "más actividad anómala = más
riesgo", y está ahí a propósito: un modelo que la aprenda estará aprendiendo
algo real sobre el mecanismo del fraude, no solo intensidad de anomalía.

Efecto emergente relacionado: `errors_per_minute` y `dead_time_ratio` crecen
mucho menos que sus numeradores, porque el denominador (`session_duration_s`)
también se duplica en vishing. Sus AUC (0,5926 y 0,6415) son sensiblemente
menores que los de `input_error_count` (0,7268) y `total_dead_time_s` (0,7637).
Es correcto: normalizar por duración quita señal cuando la duración misma es
señal.

---

## 8. Qué recalibraría con datos reales

Por orden de impacto sobre la utilidad del dataset:

1. **La distribución de la intensidad del ataque.** Es el parámetro con más peso
   sobre la dificultad del problema y el que menos anclaje empírico tiene. Con
   un conjunto de sesiones de vishing confirmadas, se estimaría directamente de
   la distribución observada de los indicadores conductuales, en vez de fijar
   una mezcla 15/85 de dos Beta.
2. **`phone_call_active`.** La cifra que hace falta no es "qué porcentaje de
   víctimas estaba al teléfono" (sabemos que es casi el 100 %), sino **qué
   porcentaje de esas llamadas es detectable en el mismo dispositivo**. Esa
   única cifra reescribe §5.1.
3. **`RHO_CLIENTE`.** Medible directamente con un ANOVA de un factor sobre
   sesiones legítimas reales agrupadas por cliente. Determina cuánto castiga el
   split agrupado, y por tanto la brecha entre split por fila y split por cliente.
4. **Toda la tabla `GANANCIA`.** Hoy es un artefacto de calibración contra un
   techo de AUC. Con datos reales, cada marginal de vishing se estimaría
   directamente y la restricción de 0,80 dejaría de ser necesaria: sería un
   resultado medido, no un objetivo de diseño.
5. **La estructura de correlación conductual.** Las cargas factoriales de
   `CARGAS` son invención razonada. Con datos reales, un análisis factorial sobre
   sesiones legítimas daría el número de factores y sus cargas.
6. **`transaction_amount_cop`.** Debería salir del histograma real de
   transferencias del banco, y la marginal de vishing del histograma de casos
   confirmados. Es la variable con más valor de negocio y la que hoy se apoya en
   supuestos más frágiles.
7. **Los scores de BioCatch.** Si el POC llega a tener acceso a scores reales del
   motor, dejarían de simularse: se usarían tal cual, y el ejercicio pasaría a
   ser medir cuánto añade un modelo propio sobre ellos.
