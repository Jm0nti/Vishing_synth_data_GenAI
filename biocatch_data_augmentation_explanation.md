
# Estrategia de Data Augmentation para Dataset Biométrico de Detección de Vishing

## Contexto del Proyecto

Este proyecto busca escalar un dataset sintético de comportamiento biométrico tipo BioCatch desde aproximadamente 50,000 sesiones hasta 1 millón de observaciones, manteniendo una prevalencia realista de fraude tipo vishing cercana al 1%.

El objetivo principal no es únicamente aumentar el tamaño del dataset, sino generar un entorno más cercano a producción bancaria real para:

- Stress testing de modelos
- Evaluación de robustez
- Simulación de fraude raro
- Evaluación de pipelines ML en escenarios de alto volumen
- Simulación de comportamiento humano bajo ingeniería social

---

# Problema Principal

Los datasets sintéticos simples suelen presentar varios problemas:

- Separabilidad excesiva entre clases
- Distribuciones demasiado limpias
- Correlaciones artificiales
- Poca diversidad intra-clase
- Overfitting de modelos
- Synthetic collapse
- Escasa complejidad temporal

Por esta razón, el pipeline implementado utiliza una estrategia híbrida que combina:

- Simulación estructural
- Modelos generativos
- Inyección de ruido controlado
- Drift temporal
- Casos ambiguos
- Validaciones estadísticas

---

# Arquitectura General del Pipeline

El pipeline fue dividido en varias fases independientes para garantizar:

- Escalabilidad
- Modularidad
- Trazabilidad
- Control estadístico
- Validación continua

La arquitectura general es:

1. Auditoría estructural
2. Remoción de leakage
3. Simulación temporal
4. Expansión de clientes/sesiones
5. Generación sintética híbrida
6. Hard-case generation
7. Constraint engine
8. Drift injection
9. Validación estadística
10. Exportación final

---

# 1. Auditoría Estructural del Dataset

## Objetivo

Antes de realizar augmentation, es fundamental entender:

- Qué tan separable es el problema
- Qué variables dominan el fraude
- Qué correlaciones existen
- Qué tan realista es el dataset base

## Acciones realizadas

Se analizaron:

- Distribuciones
- Correlaciones
- PCA
- Importancia de variables
- Separabilidad del modelo baseline

## Hallazgos

El dataset presentaba:

- Complejidad razonable
- Overlap entre clases
- Señales cognitivas plausibles
- Separabilidad moderada
- AUC baseline realista (~0.72)

Esto permitió confirmar que el objetivo no debía ser “hacer más difícil el problema”, sino aumentar diversidad estadística.

---

# 2. Remoción de Data Leakage

## Problema

Algunas columnas representan outputs simulados de BioCatch:

- biocatch_risk_score
- biocatch_genuine_score
- social engineering indicators
- claim metadata

Estas variables contienen información derivada directamente del fraude.

## Riesgo

Si un modelo generativo aprende estas variables:

- El synthetic data puede incorporar leakage implícito
- El modelo downstream puede sobreajustarse
- El fraude puede volverse artificialmente detectable

## Solución

Estas columnas fueron removidas antes del entrenamiento de los generadores sintéticos.

---

# 3. Simulación Temporal

## Motivación

Los datasets IID no representan producción bancaria real.

En producción existen:

- Picos horarios
- Variación diaria
- Campañas de fraude
- Drift conductual
- Cambios por versión de app

## Implementación

Se simularon timestamps con distribuciones horarias ponderadas.

Ejemplo:

| Hora | Actividad |
|---|---|
| 00-05 | Baja |
| 10-18 | Alta |
| 22-23 | Media/Baja |

Esto permite:

- Modelar tráfico realista
- Introducir drift temporal
- Simular horarios atípicos

---

# 4. Expansión de Clientes y Sesiones

## Motivación

En producción:

- Un cliente tiene múltiples sesiones
- Existen usuarios heavy-use
- Existen usuarios casuales

Un dataset completamente IID sería poco realista.

## Implementación

Se simuló una distribución tipo Zipf/Pareto para generar:

- Clientes con pocas sesiones
- Clientes con muchas sesiones

Esto introduce:

- Persistencia conductual
- Repetición natural
- Distribución realista de actividad

---

# 5. Clustering de Subtipos de Fraude

## Motivación

No todos los fraudes vishing son iguales.

Existen múltiples comportamientos:

- Fraude lento
- Fraude con alta hesitación
- Fraude con copy-paste
- Fraude nocturno
- Fraude con navegación errática

## Implementación

Se aplicó KMeans sobre variables cognitivas:

- hesitation_count
- dead_time_periods
- segmented_typing_ratio
- keystroke_variability
- input_error_count

## Beneficio

Permite:

- Generar diversidad intra-fraude
- Evitar modos únicos de fraude
- Incrementar realismo conductual

---

# 6. Generación Sintética Híbrida

## Motivación

Una única técnica generativa puede introducir sesgos.

Por ejemplo:

| Técnica | Riesgo |
|---|---|
| SMOTE | Interpolación artificial |
| Bootstrap | Duplicación |
| GAN puro | Memorization |
| Gaussian Noise | Distribuciones irreales |

## Estrategia implementada

Se utilizó una combinación de:

- CTGAN
- TVAE
- Bootstrap perturbado

---

# 6.1 CTGAN

## Objetivo

Modelar relaciones no lineales complejas entre variables tabulares mixtas.

## Beneficios

- Manejo de variables categóricas
- Preservación de correlaciones
- Generación de nuevos modos estadísticos

## Uso

Se entrenaron modelos separados para:

- Sesiones legítimas
- Sesiones vishing

Esto evita contaminación entre clases.

---

# 6.2 TVAE

## Objetivo

Introducir suavización distribucional.

## Beneficios

- Reduce synthetic memorization
- Genera manifold más suave
- Introduce variabilidad continua

## Uso

TVAE se utilizó como complemento de CTGAN.

---

# 6.3 Bootstrap Perturbado

## Objetivo

Preservar regiones reales del espacio estadístico.

## Beneficio

Ayuda a:

- Mantener densidades plausibles
- Evitar colapso generativo
- Preservar estabilidad

---

# 7. Hard-Case Generation

## Motivación

Los datasets sintéticos suelen carecer de sesiones ambiguas.

Esto hace que:

- Los modelos sean demasiado optimistas
- El recall sea artificialmente alto
- Existan pocos falsos positivos difíciles

## Solución

Se generaron manualmente:

### Hard Legitimate Sessions

Sesiones legítimas con:

- Llamada activa
- Hesitación alta
- Copy-paste
- Navegación lenta

### Soft Vishing Sessions

Fraudes con:

- Baja hesitación
- Menos errores
- Menor fricción cognitiva

## Resultado

Esto incrementa:

- Robustez
- Generalización
- Realismo operacional

---

# 8. Constraint Engine

## Problema

Los modelos generativos pueden producir combinaciones imposibles.

Ejemplos:

- phone_call_active = 0 y call_overlap_duration > 0
- transaction_attempted = 0 y transaction_amount > 0

## Solución

Se implementó un motor de restricciones post-generación.

## Reglas implementadas

### Llamadas

Si:

```python
phone_call_active == 0
```

Entonces:

```python
call_overlap_duration_s = 0
```

### Transacciones

Si:

```python
transaction_attempted == 0
```

Entonces:

```python
transaction_amount_cop = 0
```

## Beneficio

Evita inconsistencias lógicas y mejora realismo.

---

# 9. Drift Injection

## Motivación

En producción los patrones cambian con el tiempo.

Ejemplos:

- Nuevas versiones de la app
- Cambios en comportamiento táctil
- Evolución del fraude

## Implementación

Se introdujeron variaciones mensuales controladas sobre:

- typing_speed
- comportamiento táctil
- distribuciones temporales

## Beneficio

Permite:

- Evaluar robustez temporal
- Simular concept drift
- Evaluar estabilidad de modelos

---

# 10. Adversarial Validation

## Objetivo

Evaluar si el synthetic data es demasiado detectable.

## Estrategia

Entrenar un clasificador:

```text
original vs synthetic
```

## Interpretación

| AUC | Calidad |
|---|---|
| 0.50 - 0.60 | Excelente |
| 0.60 - 0.70 | Aceptable |
| > 0.80 | Synthetic detectable |

## Objetivo

Lograr que el synthetic data sea difícil de distinguir del original.

---

# 11. Validación Estadística

## Técnicas usadas

### KS-Test

Compara distribuciones univariadas.

### Wasserstein Distance

Mide distancia distribucional.

### Correlaciones

Verifica preservación de relaciones entre variables.

## Objetivo

Garantizar:

- Consistencia estadística
- Ausencia de colapso
- Preservación de estructura

---

# 12. Resultado Final

El pipeline genera:

- ~1,000,000 sesiones
- ~1% fraude
- Mayor diversidad intra-clase
- Drift temporal
- Casos ambiguos
- Distribuciones plausibles
- Correlaciones preservadas

---

# Beneficios del Enfoque

## Frente a oversampling tradicional

Este pipeline:

- NO duplica simplemente observaciones
- NO depende únicamente de SMOTE
- NO genera interpolaciones irreales

## En cambio:

- Expande el espacio estadístico
- Introduce complejidad operacional
- Simula producción real
- Mejora robustez

---

# Limitaciones

Aunque el pipeline es avanzado, siguen existiendo limitaciones:

- El dataset base sigue siendo sintético
- No existen señales biométricas reales
- Los patrones fueron inspirados en supuestos
- No se modela comportamiento secuencial profundo

---

# Trabajo Futuro

Posibles mejoras:

- Modelado secuencial por usuario
- Graph fraud simulation
- Diffusion models tabulares
- Simulación multi-dispositivo
- Behavioral personas
- Session replay generation
- Online augmentation

---

# Conclusión

La estrategia implementada busca balancear:

- Realismo
- Escalabilidad
- Robustez
- Diversidad estadística
- Complejidad operacional

Esto permite construir un entorno sintético mucho más cercano a escenarios reales de fraude conductual tipo vishing dentro de una institución financiera.
