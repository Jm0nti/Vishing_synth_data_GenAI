# Detectar el vishing mientras ocurre

**Biometría de comportamiento para detección de riesgo en sesión en banca digital — una prueba de concepto sobre datos sintéticos.**

[English version](README.md)

---

## Qué es este proyecto

El vishing (*voice phishing*) es una modalidad de fraude en la que un atacante
guía por teléfono a un cliente bancario para que ejecute una transacción en
beneficio del atacante. El cliente opera voluntariamente dentro de la aplicación
legítima, con su propio dispositivo y credenciales válidas. Ese solo hecho
neutraliza las defensas en las que un banco se apoya habitualmente: la validación
de canal, la huella de dispositivo y la verificación de credenciales devuelven
*verde*, porque efectivamente no hay nada mal en ellas.

Lo que queda como observable es **cómo se comporta la sesión**. Bajo la
instrucción de un atacante, el cliente escribe más despacio y a ráfagas, se
detiene antes de cada decisión mientras escucha, transcribe datos dictados,
corrige campos que no le son familiares y confirma una operación que se aparta de
su patrón habitual.

Este repositorio responde una sola pregunta operativa:

> ¿La traza de comportamiento que deja un cliente *mientras el ataque está en
> curso* alcanza para levantar una alerta de riesgo **antes de que el pago se
> autorice**, usando únicamente señales que la aplicación bancaria ya observa —
> sin audio de la llamada, sin metadatos de telefonía, sin transcripción?

Como no disponemos de sesiones reales de vishing etiquetadas — las etiquetas
confirmadas dependen de reclamaciones del cliente, resultados de investigación y
reportes tardíos, y la telemetría de comportamiento subyacente está restringida
comercial y legalmente —, la pregunta se responde **dentro de una simulación que
construimos y publicamos**. Todo lo necesario para reproducirla, cuestionarla o
recalibrarla está en este repositorio.

### Qué aporta este repositorio

1. **Un dataset diseñado para imitar el mecanismo del fraude, no solo para estar
   etiquetado.** 100.000 sesiones sobre 38.000 clientes, generadas por un proceso
   paramétrico de cópula gaussiana, con semilla y reproducible byte a byte,
   gobernado por un **techo explícito de separabilidad univariada**, de modo que
   la separabilidad inducida por la simulación es un parámetro declarado y no un
   accidente.
2. **Evidencia de que la señal comportamental es propia.** Un estudio de ablation
   por familias de variables con intervalos por bootstrap, más una prueba del
   proxy de llamada y una verificación independiente por permutación, muestran
   que el resultado no se reduce a saber que hay una llamada en curso.
3. **Un techo medido para el aumento generativo.** Como controlamos el proceso
   generador verdadero, podemos acotar cuánto podría aportar *cualquier* generador
   en este régimen — +0,012 de PR-AUC —, lo que reencuadra un resultado negativo
   de CTGAN: de «nuestro GAN rindió mal» a «el premio disponible es pequeño, y
   aquí está su descomposición».
4. **Una rama de AutoML corrida sobre las mismas particiones**, no como nota al
   pie.
5. **Un protocolo de evaluación cuyas propiedades anti-fuga están aseveradas en
   código**: particionar primero, contrato de features calculado dentro de train,
   balanceo y aumento como ramas *solo* del conjunto de entrenamiento, y una única
   apertura del test.

---

## Resultados principales

Pipeline de nueve etapas (`R0`–`R8`), dataset v3.1, partición agrupada por
cliente, una sola apertura del test.

| | Valor |
|---|---|
| **PR-AUC en test** | **0,8458** [0,8266, 0,8652] |
| Prevalencia en test | 4,91 % (992 de vishing en 20.193 sesiones) |
| ROC-AUC / Recall / Precisión / F1 | 0,9673 / 0,7903 / 0,8025 / 0,7963 |
| Matriz de confusión (test) | TN 19.008 · FP 193 · FN 208 · TP 784 |
| **recall @ Precisión = 0,90** | **0,6472** (umbral 0,9734) |
| Carga operativa por 100k sesiones | 4.838 alertas · 956 falsas · 1.030 perdidos |
| Brecha validación → test (PR-AUC) | **+0,0046** (sin sobreajuste de selección) |
| Modelo seleccionado | `xgb_shallow` · train original · sin balanceo · umbral 0,8464 |
| Comportamental vs. contexto+transacción (ablation) | **0,8084** vs. 0,7834 PR-AUC |
| Costo de quitar ambas variables de llamada | −0,0178 PR-AUC |
| Importancia por permutación de `phone_call_active` | **−0,000044** |
| Aumento con CTGAN | **negativo medido**: detectabilidad AUC 0,8738 |
| Techo del aumento (generador oráculo) | **+0,012** PR-AUC |
| Rama AutoGluon (mismos splits) | 0,8276 PR-AUC — se reporta **no concluyente** |

> **Alcance.** Todas las cifras anteriores son internas a una simulación cuyos
> parámetros declaramos. Los valores absolutos **no** se trasladan a producción.
> Lo que sí se traslada son las **comparaciones que mantienen fijo el generador**
> —entre familias de variables, entre estrategias de aumento, entre enfoques de
> modelado— y los supuestos mismos, que se publican aquí precisamente para que
> puedan ser cuestionados.

---

## Estructura del repositorio

```
.
├── data/
│   ├── biocatch_sinthetic_data.csv              ← el dataset canónico (100.000 × 58)
│   ├── generar_dataset_sintetico.py             ← el generador que lo produce
│   ├── diccionario_datos_biocatch_sintetico.md  ← diccionario de datos, columna a columna
│   ├── notas_metodologicas_generacion.md        ← decisiones de diseño y anclaje de supuestos
│   ├── reporte_validacion.md                    ← 7 comprobaciones automáticas, legible
│   └── reporte_validacion.json                  ← lo mismo, para máquina
│
├── src/
│   ├── _build_notebooks.py                      ← genera los nueve notebooks
│   ├── R0_contrato_y_split.ipynb                ← contrato de datos + split canónico
│   ├── R1_eda_train.ipynb                       ← análisis exploratorio (solo train)
│   ├── R2_ctgan_train_only.ipynb                ← aumento con CTGAN (GPU)
│   ├── R3_balanceo_train.ipynb                  ← 24 conjuntos de entrenamiento balanceados
│   ├── R4_entrenamiento_seleccion.ipynb         ← 260 modelos, selección sobre validación
│   ├── R5_evaluacion_final.ipynb                ← única apertura del conjunto de test
│   ├── R6_ablation.ipynb                        ← ablation por familias de variables
│   ├── R7_automl_comparable.ipynb               ← AutoGluon sobre los mismos splits (GPU)
│   └── R8_figuras_e_interpretabilidad.ipynb     ← figuras, importancia por permutación, SHAP
│
├── utils/
│   ├── vishing_common.py                        ← núcleo compartido: rutas, split, contrato, métricas
│   ├── _smoke_local.py                          ← corre el pipeline contra el sistema de archivos local
│   ├── _calibrar.py                             ← banco de calibración rápido del generador
│   └── _oraculo_aumento.py                      ← el experimento del generador oráculo
│
└── requirements.txt
```

Dos convenciones que conviene conocer antes de tocar nada:

- **Los notebooks se generan, no se editan a mano.** `python src/_build_notebooks.py`
  reescribe los nueve. Un cambio hecho directamente sobre un `.ipynb` se pierde en
  la siguiente regeneración.
- **Toda la E/S pasa por `vishing_common`** (`vc.read_csv`, `vc.write_parquet`, …).
  Es la única fuente de verdad para las rutas de S3, el identificador de fila, el
  contrato de features y las aserciones anti-fuga.

---

## El dataset

`data/biocatch_sinthetic_data.csv` — **100.000 sesiones × 58 columnas, semilla 42**.

| Característica | Valor |
|---|---|
| Sesiones | 100.000 |
| Sesiones de vishing | 5.000 (5,00 %) |
| Clientes | 38.000 (media 2,63 sesiones/cliente, mediana 2, p99 10, máx. 22) |
| Clientes víctima | 4.893 (máx. 2 sesiones de vishing por cliente) |
| Columnas | 58 (44 entran al modelo) |
| Canal | 100 % móvil |
| Generador | `data/generar_dataset_sintetico.py` (solo numpy + pandas + scipy) |

El dataset es **un artefacto diseñado, no una muestra de conveniencia**. Sus
variables se derivan de dos fuentes: el catálogo de indicadores que BioCatch
describe públicamente para la detección comportamental de fraude por ingeniería
social (qué indicadores existen y en qué dirección se mueven, no sus parámetros)
y mediciones publicadas del comportamiento humano promedio en dispositivos
móviles. A cada variable se le asigna una marginal legítima y una marginal de
vishing; los supuestos detrás de cada una, y cuán firmemente está anclado cada
uno, están documentados en
[`data/notas_metodologicas_generacion.md`](data/notas_metodologicas_generacion.md).

### Arquitectura del generador

1. **Población de clientes con factores latentes estables** (velocidad, pulso,
   cuidado/pericia), de modo que un mismo cliente es reconocible entre sesiones.
   Correlación intra-cliente ρ = 0,55; alrededor del 55 % de la varianza
   conductual es atribuible al cliente y no a la sesión.
2. **Cópula gaussiana.** Cada variable se genera como *z* ~ N(0,1) con una
   estructura de correlación (factores del cliente + un factor de estado por
   sesión) y luego se mapea a su marginal objetivo mediante la función cuantil.
   Esto preserva exactamente la marginal pedida y a la vez induce correlación
   realista dentro del cliente y dentro de la sesión.
3. **El ataque como interpolación, no como sustitución.** Para una sesión de
   vishing con activación *w* ∈ [0,1]:

   ```
   x = (1 − w) · F_legit⁻¹(u) + w · F_vishing⁻¹(u)      con el MISMO u
   ```

   La víctima conserva su rango relativo —un mecanógrafo rápido sigue siendo
   relativamente rápido bajo coacción— y *w* controla cuánto se manifiesta el
   ataque. *w* se deriva de una variable latente de sofisticación por sesión: el
   **≈15 % de los ataques (`P_ATAQUE_SOFISTICADO = 0.15`) apenas activa los
   indicadores y queda dentro de la nube legítima**, creando un suelo irreducible
   de falsos negativos que un problema de fraude tiene que tener.
4. **Ruido intencional en la clase legítima.** ≈2,5 % de las sesiones legítimas
   recibe la misma maquinaria de superposición con intensidad moderada: un usuario
   estresado, un adulto mayor, alguien realmente hablando por teléfono con un
   familiar.

### La regla rectora: un techo explícito de separabilidad

> **Ninguna feature puede superar un AUC univariado de
> `TECHO_AUC_UNIVARIADO` = 0,85 frente a `is_vishing`.** Cuando esa regla choca
> con las distribuciones del diseño de partida, gana la regla.

Es la decisión de diseño más importante del proyecto. Si una sola variable
explica la etiqueta, el dataset deja de ser evidencia de que el modelo aprende un
patrón multivariado, y todo el experimento se vuelve un ejercicio contable. El
techo convierte la separabilidad inducida por la simulación en una **cantidad
declarada**.

Tres constantes explícitas controlan la separabilidad, y se reportan con cada
corrida:

| Constante | Valor | Papel |
|---|---|---|
| `TECHO_AUC_UNIVARIADO` | 0,85 | El techo mismo |
| `ESCALA_GANANCIA` | 1,24 | Multiplicador global de la tabla de ganancia por variable, recortado a 1,0 |
| `BETA_INTENSIDAD_BURDO` | (6,0, 1,7) | Intensidad del ataque burdo (media ≈ 0,779) |
| `P_ATAQUE_SOFISTICADO` | 0,15 | Proporción de ataques que quedan dentro de la nube legítima |

Resultado medido: **AUC univariado máximo = 0,8380** (`max_hesitation_duration_s`),
**0 features por encima del techo**, 4 por encima de 0,80. El margen es una
propiedad de la regla, no suerte de la semilla: entre las semillas 1, 7, 42 y
2026, el peor caso es 0,8395.

Cabeza del ranking univariado, restringido a las 44 variables de modelado y
medido sobre el dataset completo (la tabla completa, con las columnas excluidas,
está en [`data/reporte_validacion.md`](data/reporte_validacion.md)):

| # | Variable | AUC |
|---|---|---|
| 1 | `max_hesitation_duration_s` | 0,8380 |
| 2 | `data_familiarity_score` | 0,8330 |
| 3 | `segmented_typing_ratio` | 0,8129 |
| 4 | `swipe_directional_variance` | 0,8027 |
| 5 | `hesitation_count` | 0,7973 |
| 6 | `transaction_amount_cop` | 0,7927 |
| 7 | `dead_time_periods` | 0,7899 |
| 8 | `call_overlap_duration_s` | 0,7880 |
| 9 | `total_dead_time_s` | 0,7853 |
| 10 | `session_duration_s` | 0,7840 |

`days_to_claim` queda en AUC 0,8846 y está **deliberadamente por encima del
techo**: es una columna post-hoc, que solo se conoce después de que la víctima
reclamó, y `vishing_common.POSTHOC_COLS` la descarta antes del modelado. El
generador la excluye de la regla de forma explícita.

### Dos notas de diseño que cambian cómo se leen los datos

**`phone_call_active` no significa «hubo llamada».** Significa *se detectó una
llamada en el mismo dispositivo*. Tomar literalmente el 85 % del diseño de
partida hace que esa sola variable alcance AUC 0,90 y vuelve trivial el dataset.
Reinterpretada, su prevalencia observada dentro del vishing es 60,4 % y su AUC
univariado 0,7723 — que es lo que hace que la prueba del proxy de llamada en R6
sea significativa y no circular.

**El centinela `-1` no es un cero ni un faltante.** En
`transaction_amount_cop`, `time_to_transaction_s` y `days_to_claim`, `-1`
significa «no aplica a esta sesión». Toda fila sintética —CTGAN en R2, SMOTE en
R3— debe restaurarlo, o se acaba con montos interpolados en sesiones donde no
hubo intento de pago. `vc.repair_coherence` lo impone y `vc.check_coherence` lo
verifica.

### Validación automática

`data/reporte_validacion.md` se regenera junto con el dataset y reporta siete
comprobaciones; la corrida vigente las **PASA** todas:

| # | Comprobación | Resultado |
|---|---|---|
| 1 | Columnas constantes | 0 |
| 2 | Columnas duplicadas (\|r\| ≥ 0,999) | 0 pares |
| 3 | Balance de clases | 5.000 / 95.000 (5,00 %) |
| 4 | Techo de AUC univariado | máx. 0,8380, techo 0,85, 0 por encima |
| 5 | Sesiones por cliente | media 2,632, máx. 22 |
| 6 | Co-ocurrencia dentro de `is_vishing = 1` | todas positivas, ninguna en 1,0 (0,079–0,175) |
| 7 | Rangos y sanidad básica | 15/15 aserciones verdaderas, sin nulos, `row_id` único |

La comprobación 6 es la que mantiene honesto el problema: los indicadores
co-ocurren, pero ningún patrón de ataque enciende todos los indicadores a la vez.

---

## Variables

De las 58 columnas, **44 entran al modelo**. El resto se excluye por categoría, y
las listas de exclusión viven en `vishing_common.py` para que no se desincronicen
entre notebooks:

| Lista | Columnas | Por qué se excluyen |
|---|---|---|
| `ID_COLS` | `session_id`, `customer_id`, `session_timestamp`, `os_type`, `app_version` | Identificadores y metadatos; no son señal de fraude |
| `LEAKAGE_COLS` | `biocatch_risk_score`, `biocatch_genuine_score`, `biocatch_ato_indicator`, `biocatch_social_eng_indicator`, `biocatch_bot_indicator` | Son salidas del propio motor de riesgo que se quiere replicar |
| `POSTHOC_COLS` | `days_to_claim`, `claim_category` | Solo se conocen después de que el fraude fue reclamado |

### Grupos funcionales (los que usa la ablation)

| Grupo | n | Variables |
|---|---|---|
| `keystroke` | 5 | tiempo de tecla, latencia entre teclas, velocidad, variabilidad, ratio de escritura segmentada |
| `touch` | 5 | presión, tamaño de toque, velocidad de swipe, varianza direccional, velocidad de scroll |
| `motion` | 5 | inclinación media y variabilidad, giroscopio, jerk del acelerómetro, eventos de movimiento |
| `hesitation` | 3 | cantidad, duración media y máxima de las pausas |
| `dead_time` | 3 | períodos, segundos totales, ratio |
| `navigation` | 3 | pantallas únicas, navegación hacia atrás, tiempo de transición |
| `corrections` | 5 | errores de entrada, correcciones, campo de monto, campo de beneficiario, copiar/pegar |
| `derived` | 4 | familiaridad con los datos, doodling, errores por minuto, compuesto de hesitación |
| `context` | 7 | duración, hora, hora atípica, llamada activa, **solapamiento de llamada**, herramienta de acceso remoto, app sospechosa |
| `transaction` | 4 | intento, monto, beneficiario nuevo, tiempo hasta la transacción |

Agregados en tres familias para la ablation:

- **`behavioral`** (33 variables) = keystroke + touch + motion + hesitation +
  dead_time + navigation + corrections + derived
- **`context`** (7 variables)
- **`transaction`** (4 variables)

`call_overlap_duration_s` está en **`context`**, no en `transaction`: describe
cuánto solapó la llamada con la sesión, que es contexto de sesión y no un
atributo del pago.

---

## Protocolo de evaluación

El protocolo es la parte de este proyecto que hace que las cifras merezcan ser
leídas. Cinco propiedades, cada una aseverada en código y no descrita en prosa:

1. **Particionar primero, transformar después.** `R0` produce el split canónico
   antes de que ningún balanceo, aumento o trabajo de features toque los datos.
2. **Agrupado por cliente.** Con 2,63 sesiones por cliente y ~55 % de la varianza
   conductual atribuible al cliente, un split por fila deja sesiones de la misma
   persona a ambos lados de la frontera e infla el desempeño. El split es
   `SPLIT_MODE = "grouped"`, por `customer_id`, estratificado según si el cliente
   tuvo alguna sesión de vishing, en 60/20/20.
3. **Una identidad de fila que sobrevive a todo.** `row_id` es una columna real
   del CSV y viaja por todos los parquet, de modo que `vc.assert_disjoint` puede
   demostrar que ninguna fila de entrenamiento —original, generada por CTGAN o
   remuestreada— aparece en validación o test.
4. **Un contrato de features calculado solo sobre train.**
   `vc.build_feature_contract` audita las columnas por constancia y duplicación
   exacta *dentro de train* y congela la lista. Rankear variables por poder
   discriminante mirando también el test es contaminación, leve pero real.
5. **Una sola apertura del test.** La selección de modelo y la del umbral ocurren
   ambas sobre validación, en `R4`. `R5` mide una vez. Volver atrás y cambiar algo
   después de leer `R5` convierte el test en un segundo conjunto de validación.

El balanceo y el aumento son **ramas del conjunto de entrenamiento**, nunca del
dataset. Cada rama se vuelve a comprobar contra validación y test.

### La métrica de referencia

Con una prevalencia cercana al 5 %, el ROC-AUC es engañosamente optimista porque
penaliza poco los falsos positivos frente a una clase mayoritaria abrumadora. El
**PR-AUC** es la métrica de referencia en todo el proyecto. Junto a él reportamos
dos cosas que un banco realmente tiene que planificar:

- **recall @ Precisión = 0,90** — cuánto fraude se atrapa si se exige que 9 de
  cada 10 alertas sean reales.
- **Alertas por cada 100.000 sesiones** — la cola que un equipo de operaciones
  tiene que dimensionar.

---

## Pipeline

El orden de ejecución es **R0 → R1 → R2 → … → R8, sin excepciones**. R0 crea el
split del que depende todo lo demás, incluido el análisis exploratorio de R1, que
se calcula solo sobre train.

| Notebook | Dónde | Entrada | Salida |
|---|---|---|---|
| `R0_contrato_y_split` | CPU | CSV crudo | auditoría, contrato de features, manifiesto, mapa de split, train/val/test |
| `R1_eda_train` | CPU | train | separabilidad, razones de momios, correlaciones, perfil, PCA, atípicos, figuras |
| `R2_ctgan_train_only` | **GPU** | train | 2 sintetizadores, train aumentado, reporte KS, detectabilidad |
| `R3_balanceo_train` | CPU | train (± aumentado) | 24 conjuntos de entrenamiento balanceados |
| `R4_entrenamiento_seleccion` | GPU/CPU | 26 conjuntos + validación | 260 modelos, leaderboard de validación, ganador |
| `R5_evaluacion_final` | CPU | ganador + test | reporte final de test con IC por bootstrap |
| `R6_ablation` | CPU | train balanceado + validación | 6 configuraciones de ablation con IC |
| `R7_automl_comparable` | **GPU** | los mismos splits | leaderboard de AutoGluon |
| `R8_figuras_e_interpretabilidad` | CPU | ganador + validación | figuras, importancia por ganancia y permutación, SHAP |

### R0 — Contrato de datos y split canónico

Valida el CSV crudo contra el esquema esperado (`DATASET_VERSION = "v3"`, forma
del archivo) antes que nada — es más barato fallar aquí que descubrir en R4 que se
subió el archivo equivocado. Luego audita las columnas candidatas por constancia y
duplicación exacta **sobre train**, construye el contrato de features, crea el
split agrupado por cliente y escribe un manifiesto con las versiones exactas de
los paquetes resueltos.

Se espera que la auditoría **no dé ninguna baja**, y ese resultado vacío es el
entregable: es la comprobación de que el CSV subido es el correcto. Si vuelve a
aparecer una constante o un duplicado exacto, hay que parar aquí.

El CSV crudo se carga con `vc.read_raw()` y no con `vc.read_csv()`:
`claim_category` usa la cadena vacía, que pandas convertiría en silencio en unos
95.000 NaN falsos.

### R1 — Análisis exploratorio sobre train

Descriptivo; no alimenta a ningún notebook posterior, pero produce todas las
figuras y tablas del EDA. Se calcula solo sobre train, de modo que nada del
ranking exploratorio está informado por el conjunto de test.

Lo destacado de la corrida vigente:

- **Comprobación del techo de AUC univariado**: máximo **0,8399** medido sobre
  train (`max_hesitation_duration_s`), 4 variables por encima de 0,80, ninguna por
  encima del techo de 0,85 — coherente con el 0,8380 medido sobre el dataset
  completo. R1 tiene una constante `TECHO_AUC` que debe coincidir con
  `TECHO_AUC_UNIVARIADO` del generador — es el único acoplamiento entre el
  generador y el pipeline.
- **Razones de momios** — con una advertencia explícita:
  `remote_access_tool_detected` tiene OR 37,67 y AUC univariado 0,5521. Una razón
  de momios alta sobre un evento muy raro **no** es poder predictivo.

  | Variable | OR |
  |---|---|
  | `remote_access_tool_detected` | 37,67 |
  | `phone_call_active` | 23,64 |
  | `suspicious_app_detected` | 12,94 |
  | `is_new_beneficiary` | 11,00 |

- **Montos de transacción**: mediana de 1.662.824 COP en vishing frente a 249.229
  COP en sesiones legítimas (6,67×).
- **Enriquecimiento en colas**: `dead_time_periods` enriquece la tasa de vishing
  17,08× en su cola de atípicos.
- **PCA**: los primeros 5 componentes explican el 33,0 % de la varianza — la señal
  está genuinamente distribuida, no concentrada en un par de direcciones.

Todas las figuras se producen **en inglés** y R1 expone celdas configurables para
regenerar las gráficas sin editar código.

### R2 — Aumento con CTGAN, ajustado solo sobre train

Dos modelos `CTGANSynthesizer` condicionales por clase (SDV ≥ 1.5), ajustados
**solo con train**, que generan 44 columnas de comportamiento. Las variables
binarias se declaran `categorical` para que CTGAN use el mecanismo condicional
discreto. Las filas generadas pasan por `vc.repair_coherence` —la misma capa
determinista que usa R3—, que restaura los centinelas `-1` y las coherencias
lógicas que el GAN no aprende.

El dataset aumentado extiende **únicamente train**. Validación y test siguen
siendo 100 % originales y no se tocan.

La rama corre dos pruebas de calidad: Kolmogorov-Smirnov por variable y un
**clasificador original-vs-sintético**. Ver [Resultados](#resultados) — es un
resultado negativo medido, y se reporta como tal.

Requiere GPU (`ml.g4dn.xlarge` o superior). Instala sus propias versiones fijadas
de SDV/CTGAN en la primera celda.

### R3 — Balanceo, solo sobre train

Cuatro técnicas de remuestreo × tres tasas objetivo de vishing (10 %, 20 %, 25 %),
aplicadas **exclusivamente a train**, sobre cada fuente disponible (`original`, y
`augmented` si R2 corrió) — 24 conjuntos balanceados, más las dos fuentes sin
balancear.

| Técnica | Descripción |
|---|---|
| Random oversampling | Duplica filas de la clase minoritaria |
| SMOTE | Interpolación k-NN de la clase minoritaria |
| Borderline SMOTE | SMOTE enfocado en ejemplos de la frontera de decisión |
| SMOTE + undersampling | Reduce la clase mayoritaria ~10 % y aplica SMOTE |

Cada conjunto resultante se somete a la aserción anti-fuga contra validación y
test. Si R2 no se ejecutó, la fuente `augmented` se omite con un aviso en vez de
reventar a mitad del bucle.

### R4 — Entrenamiento y selección sobre validación

**260 modelos** = 2 tipos de datos × 13 conjuntos de entrenamiento (12 balanceados
+ 1 sin balancear) × 10 configuraciones (7 variantes de XGBoost + regresión
logística + random forest + MLP).

| Variante de XGBoost | Configuración |
|---|---|
| `xgb_base` | profundidad 6, lr 0,1, 100 árboles |
| `xgb_deep` | profundidad 9, lr 0,05, 300 árboles, `min_child_weight=3` |
| `xgb_shallow` | profundidad 3, lr 0,1, 500 árboles |
| `xgb_regularized` | profundidad 6, 200 árboles, L1 1,0, L2 5,0, `min_child_weight=10`, γ 0,3 |
| `xgb_balanced` | `scale_pos_weight = n_neg/n_pos`, calculado por dataset en tiempo de ajuste |
| `xgb_conservative` | subsample 0,7, colsample 0,7, γ 0,5 |
| `xgb_slow_learner` | lr 0,01, 500 árboles, subsample 0,8 |

Tanto el umbral de decisión (óptimo de F1 sobre la curva PR) como la elección del
modelo se deciden **sobre validación**. El test no se abre en este notebook.
XGBoost resuelve `cuda`/`cpu` en ejecución mediante `vc.xgb_base_params()`, así
que el mismo notebook corre en una instancia con GPU o sin ella.

### R5 — Evaluación final

Se ejecuta **una sola vez**, al final. El modelo y el umbral ya quedaron fijados;
aquí no se ajusta nada. Reporta PR-AUC, ROC-AUC, recall, precisión y F1 con
**intervalos de confianza al 95 % por bootstrap**, además de recall@P=0,90, la
carga operativa por cada 100.000 sesiones y la brecha validación → test, que es el
indicador honesto de sobreajuste de selección.

### R6 — Estudio de ablation por familias de variables

Seis configuraciones, todas evaluadas **sobre validación** (el test quedó cerrado
en R5), replicando la configuración ganadora de R4:

1. `context_transaction` — solo contexto y transacción (11 variables)
2. `behavioral` — solo biometría comportamental (33 variables)
3. `behavioral_context`
4. `behavioral_transaction`
5. `all` — todas (44 variables)
6. `all_sin_phone_call` — todas menos `phone_call_active` y `call_overlap_duration_s`

La configuración 6 es la **prueba del proxy de llamada**: la respuesta directa a
la objeción de que un detector comportamental es en realidad un detector de
llamadas.

### R7 — AutoML sobre exactamente los mismos splits

AutoGluon `TabularPredictor`, `eval_metric='average_precision'`,
`presets='best_quality'`, `time_limit=5400` s, `use_bag_holdout=True`,
`num_bag_folds=8`, `num_stack_levels=1` — recibiendo **el mismo train, la misma
validación y el mismo test** que el pipeline dirigido, y medido con las mismas
métricas. Instala su propio árbol de dependencias en la primera celda para no
contaminar los demás notebooks.

### R8 — Figuras e interpretabilidad

Produce las figuras que ningún otro notebook genera (distribución de KS, PR-AUC
por algoritmo y técnica, heatmap variante × técnica, importancia de variables) y
añade la capa de interpretabilidad: **ganancia** acumulada de XGBoost,
**importancia por permutación** sobre validación y, opcionalmente, valores
**SHAP**.

La distinción importa para auditoría: la ganancia acumulada no es aditiva y no
sirve para explicar un caso individual. La permutación y SHAP sí.

---

## Resultados

### 1. La detección en sesión funciona, a un costo que se puede planificar

| Métrica | Test |
|---|---|
| PR-AUC | **0,8458** [0,8266, 0,8652] |
| ROC-AUC | 0,9673 |
| Recall / Precisión / F1 | 0,7903 / 0,8025 / 0,7963 |
| Matriz de confusión | TN 19.008 · FP 193 · FN 208 · TP 784 |
| recall @ P = 0,90 | 0,6472 (umbral 0,9734) |
| Por 100k sesiones | 4.838 alertas · 956 falsas · 1.030 perdidos |

El punto de operación que importa es el segundo. Con una precisión del 90 % —nueve
casos reales por cada diez alertas— el detector todavía atrapa el **65 % de los
ataques mientras la sesión está abierta**. Ese es el número con el que hay que
discutir, no el PR-AUC.

**Sin sobreajuste de selección.** La brecha validación → test es **+0,0046**: el
test salió *mejor* que la validación. Elegir entre 260 modelos no compró nada, y
tampoco costó nada.

**El leaderboard es plano.** 18 modelos quedan a menos de un punto del ganador, y
una **regresión logística simple alcanza 0,8350** — a 1,5 puntos del mejor
XGBoost. El resultado es una propiedad del conjunto de variables, no del
algoritmo. Cualquier conclusión que dependiera de la elección concreta del modelo
sería frágil.

### 2. La señal comportamental es complementaria, y no es la llamada

Ablation, PR-AUC sobre validación:

| Configuración | Variables | PR-AUC |
|---|---|---|
| `all` | 44 | 0,8504 |
| `behavioral_context` | 40 | 0,8435 |
| `all_sin_phone_call` | 42 | 0,8326 |
| `behavioral_transaction` | 37 | 0,8140 |
| **`behavioral`** | 33 | **0,8084** [0,7868, 0,8307] |
| **`context_transaction`** | 11 | **0,7834** [0,7622, 0,8054] |

Las variables comportamentales **superan** al bloque de contexto + transacción por
**+0,0250**, con la estimación puntual por encima del límite superior del IC del
rival, y **aportan materialmente por encima de él**: `all` supera a
`context_transaction` por 0,0670. La formulación correcta es complementariedad: la
traza de comportamiento lleva información que el contexto de la sesión y los
atributos del pago no pueden suministrar.

**La prueba del proxy de llamada.** Quitar ambas variables relacionadas con la
llamada cuesta solo **−0,0178**. Y la **importancia por permutación del indicador
binario de llamada es −0,000044**, indistinguible de cero. El modelo no se apoya
en saber que hay una llamada en curso; la señal que hay vive en
`call_overlap_duration_s`, la duración continua del solapamiento, no en la
bandera binaria.

Esto convierte la objeción más obvia a toda la premisa en un riesgo medido y
descartado.

**Qué usa realmente el modelo.** Se usan 43 de 44 variables (`is_atypical_hour`
nunca), pero 12 tienen importancia por permutación ≤ 0,0005: el modelo se sostiene
efectivamente sobre unas 32 variables.

### 3. Aumento generativo: un resultado negativo medido, con un techo medido

CTGAN pasa un criterio de fidelidad marginal y falla el que importa:

| Prueba | Resultado |
|---|---|
| KS medio por variable | 0,0767 (11 de 44 por encima de 0,10) |
| **Detectabilidad original-vs-sintético** | **AUC 0,8738 ± 0,0034** |
| Efecto sobre el PR-AUC | todas las configuraciones de balanceo **pierden 2,1–3,1 puntos** |

Un clasificador separa filas reales de generadas con AUC 0,87. Las marginales
están bien y la estructura conjunta no — que es exactamente el modo de falla que
el KS por variable no puede ver. **El KS no valida un generador tabular.**

Lo instructivo es el hilo causal entre R2 y R8: la variable que CTGAN peor
reproduce es `hour_of_day` (KS 0,2041, la peor de 44), y `hour_of_day` es la
**tercera más determinante** por importancia de permutación (0,0254). La peor
columna del generador es una de las más importantes del modelo.

**El experimento del generador oráculo.** La respuesta natural a «CTGAN perdió» es
«probemos TabDDPM». `utils/_oraculo_aumento.py` demuestra que esa ruta no tiene
recorrido. Muestrea filas nuevas del proceso generador **verdadero** (otras
semillas de `generar_dataset_sintetico.py`), que es un generador perfecto por
construcción: detectabilidad 0,5, marginales y estructura conjunta exactas,
jerarquía cliente-sesión exacta. Si ni siquiera *ese* ayuda, ningún generador
aprendido puede.

| Variante | Δ PR-AUC vs. baseline (0,8460) |
|---|---|
| Réplica del diseño de R2 (4× @ 1,5 %) | +0,0028 |
| **Mismo volumen @ 5 % de prevalencia** | **+0,0120** ← el techo |
| Solo minoría, 2× positivos | +0,0049 |
| Solo minoría, 5× positivos | +0,0083 |
| Mezcla pequeña (0,5× @ 5 %) | −0,0002 |
| **CTGAN (medido)** | **−0,036** |

La brecha entre CTGAN y la perfección es de 4,8 puntos, pero **el premio total en
juego es 1,2**. No hay que cambiar de generador esperando cifras.

Por qué el techo es bajo: no hay escasez de datos. Con ~60.000 filas de
entrenamiento y ~3.000 positivos, la curva de aprendizaje ya está plana. El
aumento paga cuando se tienen cientos de positivos, no miles.

Dos detalles de diseño cuestan más que la elección del generador:

1. Apuntar a una tasa de vishing del 1,5 % **diluye** la prevalencia del 5,0 % al
   1,88 %. Con el mismo generador perfecto, eso tira el **75 % del beneficio
   disponible** (+0,0028 frente a +0,0120).
2. Generar ~492.500 filas **legítimas**. La clase mayoritaria no necesita aumento;
   solo-minoría logra +0,005/+0,008 con el 2 % del cómputo.

Y hay un desajuste estructural: CTGAN es un modelo de filas independientes,
mientras que el proceso generador tiene ρ = 0,55 dentro de cada cliente sobre 2,63
sesiones. Asignar un cliente sintético por fila sintética deja ~89 % del train
aumentado como clientes con una sola sesión, evaluado contra validación y test que
tienen 2,63 sesiones por cliente.

Este es el hallazgo más difícil de obtener en otro lugar: acotar cuánto *podría*
valer el aumento exige conocer el proceso generador verdadero, que es justamente
lo único que una prueba de concepto sintética sí tiene.

### 4. AutoML: se reporta como no concluyente, a propósito

| | Pipeline dirigido | AutoGluon |
|---|---|---|
| PR-AUC en test | 0,8458 | 0,8276 |
| recall @ P = 0,90 | 0,6472 | 0,6442 |
| Precisión | 0,8025 | **0,8283** |

La rama de AutoGluon (`WeightedEnsemble_L3`) queda 0,0182 de PR-AUC por detrás — y
obtiene la **mejor precisión** de los dos sistemas. Lo reportamos como **no
concluyente, no como una victoria**, porque la comparación que corrimos no tuvo
protocolos equiparados. AutoGluon cargó con tres desventajas medidas:

1. Entrenó sobre el conjunto **aumentado** mientras el pipeline dirigido usaba el
   original. La penalización medida del aumento (0,0227) es **mayor que el déficit
   observado** (0,0182) — en igualdad de condiciones, AutoGluon probablemente
   ganaría.
2. Ajustó solo 14 de 216 modelos candidatos dentro de su presupuesto de tiempo.
3. Corrió sin GPU ni PyTorch, así que sus familias neuronales nunca se entrenaron.

Reclamar aquí una victoria para el diseño dirigido atribuiría al enfoque lo que
plausiblemente es un artefacto del montaje. Una reejecución equiparada —mismo
conjunto de entrenamiento original, mismo presupuesto, GPU disponible— es la forma
honesta de resolverlo.

---

## Qué sostiene la evidencia y qué no

**Sostenido.**

- Un ataque de vishing en curso deja en la sesión bancaria una firma comportamental
  lo bastante fuerte como para soportar una decisión de riesgo **mientras la sesión
  sigue abierta**, usando solo señales que la propia aplicación observa.
- Esa firma **no** es una reformulación de la circunstancia contextual y
  transaccional: las variables comportamentales superan al bloque
  contexto+transacción y aportan por encima de él.
- El resultado **no se reduce a saber que hay una llamada en curso**: quitar ambas
  variables de llamada cuesta 0,0178 de PR-AUC, y la bandera binaria de llamada
  tiene importancia por permutación nula.

**Refutado.**

- El aumento generativo condicional por clase **no** ayuda en este régimen — y el
  experimento del oráculo muestra que el techo para *cualquier* generador es
  +0,012 de PR-AUC.

**No concluyente.**

- Si el diseño dirigido supera a la búsqueda automatizada. La comparación no tuvo
  protocolos equiparados; ver arriba.

**Condiciones de frontera que aplican a todo lo anterior.**

- No se usaron sesiones reales de vishing etiquetadas. Cada cifra se obtuvo sobre
  datos que simulamos nosotros mismos, así que las hipótesis se contrastan **dentro
  de un régimen cuyos parámetros declaramos**, no en producción.
- Los valores absolutos de las métricas son internos a ese régimen. Lo que se
  traslada son las comparaciones que mantienen fijo el generador, y los supuestos
  mismos.
- **No se hace ninguna afirmación de que esto esté listo para producción.** Antes
  de que algo de esto toque un sistema real, los supuestos de
  [`data/notas_metodologicas_generacion.md`](data/notas_metodologicas_generacion.md)
  —la §8 lista qué recalibraríamos primero con datos reales— tienen que volver a
  estimarse contra telemetría real.

---

## Cómo ejecutarlo

### 1. Entorno

```bash
pip install -r requirements.txt
```

Los rangos de versiones siguen la máscara de compatibilidad de SageMaker
Distribution 1.x / Python 3.11. Están duplicados en `VERSIONES_ESPERADAS`, dentro
de `vishing_common.py`; si cambias uno, cambia el otro. La celda de arranque de
todos los notebooks llama a `vc.check_versions()` y avisa si el kernel resolvió
algo fuera del rango esperado.

Dos dependencias pesadas **no se instalan aquí a propósito** — cada notebook que
las necesita las instala en su primera celda, para no arrastrar su árbol de
dependencias a los demás:

- `autogluon.tabular[all]>=1.0,<1.2` (R7)
- `shap>=0.44` (R8, opcional)

### 2. Regenerar el dataset (opcional)

```bash
python data/generar_dataset_sintetico.py --seed 42 --salida ./data
```

Requiere únicamente numpy, pandas y scipy — la generación es paramétrica a
propósito, para tener control total del proceso generador. Reescribe
`reporte_validacion.md` y `reporte_validacion.json` junto al CSV.

**Si cambias el techo, cámbialo en dos sitios**: `TECHO_AUC_UNIVARIADO` en el
generador y `TECHO_AUC` en la celda de configuración de R1. Es el único
acoplamiento entre ambos, y un desajuste o lanza una falsa alarma o —peor— calla
una real.

Antes de tocar parámetros del generador, usa el banco de calibración rápido en
lugar de correr todo el pipeline:

```bash
python utils/_calibrar.py data/biocatch_sinthetic_data.csv
```

Mide, en unos 8 segundos, el techo univariado y el PR-AUC multivariado de las
cuatro configuraciones clave de la ablation (`all`, `behavioral`,
`context_transaction`, `all_sin_phone`), replicando el split agrupado de R0.
Comprueba siempre varias semillas: el margen de separabilidad debe ser una
propiedad de la regla, no de la semilla 42.

### 3. Prueba de humo local

Corre el pipeline contra el sistema de archivos local antes de gastar una
instancia con GPU:

```bash
python utils/_smoke_local.py \
    --csv data/biocatch_sinthetic_data.csv \
    --out /tmp/smoke \
    --notebooks R0 R1 R3 R4 R5 R6 R8
```

R2 (CTGAN) y R7 (AutoGluon) quedan fuera: necesitan GPU y paquetes pesados. Como
R2 no corre, R3 solo produce la rama `original` y R4 entrena 130 modelos en vez de
260. En CPU, R4 tarda ~10 minutos. Dependencias del arnés: `nbformat`, `pyarrow`,
`imbalanced-learn`, `boto3`, `xgboost`.

### 4. Corrida completa en SageMaker

Sube el CSV canónico a la única ruta de entrada que espera el pipeline y ejecuta
los notebooks **en orden**. La estructura de S3 está definida una sola vez, en
`vishing_common.Paths`:

```
s3://poc-vishing/v2/
├── 00_raw/            ← el único archivo que subes
├── 01_contract/       data_audit.csv · feature_contract.json · run_manifest.json
├── 02_splits/         split_map.parquet · train/val/test.parquet
├── 03_augmented/      synthesizers/ · train_augmented.parquet · quality/
├── 04_balanced/       {data_type}/{technique}/{ratio}.parquet
├── 05_models/         {data_type}/{variant}/{technique}/{ratio}.pkl
├── 06_results/        leaderboard · ablation · final_test · eda/ · automl/
└── 07_figures/
```

Cambia `BUCKET` y `PREFIX` en `vishing_common.py` para apuntar a otro sitio.

### 5. Dos notas de rutas antes de la primera corrida

La estructura del repositorio se reorganizó para publicarlo y dos referencias del
código todavía apuntan a la disposición anterior. Ninguna rompe nada que no se
arregle en una línea, pero ambas fallan en la primera celda si se pasan por alto:

- **Nombre del dataset.** `vishing_common.RAW_FILENAME` vale
  `"biocatch_sinthetic_data_v3.csv"`, mientras que el archivo que viene en `data/`
  se llama `biocatch_sinthetic_data.csv`. O renombras el archivo al subirlo a
  `00_raw/`, o cambias la constante.
- **Ubicación del módulo.** La celda de arranque de todos los notebooks hace
  `sys.path.insert(0, os.getcwd())` e importa `vishing_common`, es decir, espera
  que el módulo esté **junto a los notebooks**. Aquí los notebooks están en `src/`
  y el módulo en `utils/`. Copia `utils/vishing_common.py` al directorio de trabajo
  junto a los notebooks (que es lo que hace de todas formas una corrida en
  SageMaker), o añade `utils/` al `sys.path`.

---

## Reproducibilidad

- **Semillas.** `vc.set_all_seeds()` corre en cada celda de arranque y cubre
  `random`, numpy y torch, incluido cuDNN determinista. El generador tiene semilla
  42 y es reproducible byte a byte.
- **Versiones.** `R0` escribe las **versiones exactas resueltas** en
  `01_contract/run_manifest.json`. Ese archivo, y no `requirements.txt`, es el
  registro de lo que una corrida usó realmente.
- **Fuente única de verdad para las rutas.** Todas las URI de S3 salen de
  `vishing_common.Paths`. Usar `pd.read_csv` sobre una ruta de S3 rompe la prueba
  de humo local — hay que pasar por `vc.read_csv` / `vc.write_parquet`.
- **Resolución de dispositivo.** Los parámetros de XGBoost vienen de
  `vc.xgb_base_params()`, que resuelve `cuda`/`cpu` en ejecución. Nunca escribas
  `device="cuda"` a mano: XGBoost ≥ 2.0 falla en una instancia sin GPU.
- **Salvedad conocida.** En la corrida de referencia, `R7` y `R8` se ejecutaron en
  un intérprete distinto de `R0`–`R6` (numpy 1.26.4 / scikit-learn 1.4.0 /
  xgboost 3.2.0 frente a 2.0.2 / 1.5.2 / 2.1.4). La afirmación de reproducibilidad
  exacta cubre `R0`–`R6`.

### Pendientes abiertos

- Reejecutar `R8` con la corrección de SHAP ya aplicada en `_build_notebooks.py`
  (XGBoost 3.x trae un `base_score` vectorial que `shap < 0.46` no sabe leer; la
  corrección lo normaliza y añade un artefacto `shap_importance.csv`).
- Reejecución de `R7` con protocolo equiparado: `USAR_AUMENTADO = False`, instancia
  con GPU y un presupuesto que permita a AutoGluon ajustar más de 14 de sus 216
  candidatos.
- Estudio de estabilidad de las métricas reportadas entre semillas.

---

## Índice de documentación

| Documento | Qué cubre |
|---|---|
| [`data/diccionario_datos_biocatch_sintetico.md`](data/diccionario_datos_biocatch_sintetico.md) | Cada columna: tipo, rango, unidades, significado, familia, y las notas de diseño que hay que conocer antes de modelar |
| [`data/notas_metodologicas_generacion.md`](data/notas_metodologicas_generacion.md) | Arquitectura del generador, parámetros efectivos por columna, desviaciones respecto del diseño de partida y por qué, anclaje de supuestos (anclado en literatura / rango de literatura / supuesto propio), y qué recalibraríamos con datos reales |
| [`data/reporte_validacion.md`](data/reporte_validacion.md) | Las siete comprobaciones automáticas, con el ranking completo de AUC univariado |

---

## Una nota sobre el origen de los supuestos

BioCatch publica información general sobre los vectores de comportamiento que usa
para detectar fraude en banca, pero **no divulga las variables exactas ni cómo las
computa**. Lo que su documentación pública aporta aquí es un *catálogo de
indicadores y la dirección en la que cada uno se mueve bajo fraude por ingeniería
social* — no parámetros, distribuciones ni umbrales. Los valores numéricos del
generador vienen de tres niveles claramente separados, marcados como tales en
`notas_metodologicas_generacion.md`:

1. **Anclado en literatura citada** — por ejemplo `typing_speed_cps = 3.2`, de un
   estudio publicado a gran escala sobre velocidad de escritura en móvil.
2. **Rangos orientativos de literatura** — límites plausibles, no cifras citables.
3. **Supuestos propios** — marcados en el código con `SUPUESTO PROPIO`.

Esa separación es deliberada. Es lo que hace la simulación cuestionable: quien no
esté de acuerdo con un supuesto puede encontrarlo, ver a qué nivel pertenece,
cambiarlo y volver a correr todo.
