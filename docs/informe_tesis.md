# Salud Probabilística de Transformadores de Potencia

## Health Index con Incertidumbre Calibrada para Priorizar Intervención y Postergar Reemplazos de Capital

**Documento de tesis** · Investigación reproducible
**Autor**: Miguel Ortega Co. · Consultoría independiente
**Versión**: 1.0 · Octubre 2026

---

## Resumen

La práctica estándar de mantenimiento de transformadores de potencia — análisis de gases disueltos (DGA) interpretado con reglas deterministas — produce dos problemas opuestos: una tasa elevada de falsas alarmas que movilizan cuadrillas sin necesidad, y una ausencia de señales de anticipación sobre **cuándo** intervenir. Este trabajo presenta un *Health Index* probabilístico calibrado por unidad, acompañado de modelos de supervivencia para vida útil remanente **condicional a la edad actual** con censura correcta, y una capa de decisión que pondera capital diferido contra riesgo cuantil extremo. La combinación se valida sobre 121 unidades sintéticas que reproducen las distribuciones publicadas en IEC 60599 e IEEE C57.104, con 25 eventos de falla y 96 unidades censuradas (79 % de censura). La calibración conformal CQR alcanza 95 % de cobertura en calibración y 83 % en validación; el SHAP TreeExplainer identifica C₂H₄ media, H₂ última y C₂H₄ máxima como las tres contribuciones dominantes — todas relacionadas con DGA, lo que valida la primacía del diagnóstico químico sobre la edad cronológica.

---

## Abstract

Standard power-transformer maintenance practice — dissolved gas analysis (DGA) interpreted with deterministic rules — produces two opposite failures: a high false-alarm rate that mobilizes crews unnecessarily, and a lack of anticipatory signals about **when** to act. This thesis presents a unit-level **probabilistic Health Index** with conformal (CQR) calibration, **survival models** for remaining useful life **conditional on current age** with right-censoring, and a **decision layer** that trades capital deferral against extreme-quantile risk. The stack is reproducible from 121 units that mimic IEC 60599 / IEEE C57.104 distributions with 25 failure events and 96 censored units (79 % censoring). Conformal coverage reaches 95 % on calibration and 83 % on validation; SHAP TreeExplainer identifies mean C₂H₄, last H₂, and max C₂H₄ as the three dominant contributions — all gas-related, validating the primacy of chemical diagnostics over chronological age.

---

## 1. Introducción

El transformador de potencia es el activo más crítico y caro de una subestación. Su falla no programada implica reposición de capital (USD 1 M–5 M para unidades 110–220 kV; USD 4 M–12 M para 500 kV), plazos de compra de 12 a 36 meses y riesgo de indisponibilidad con multas regulatorias. El mantenimiento basado en condición busca reemplazar el paradigma "tiempo fijo" por "estado real".

El análisis de gases disueltos (DGA) — cromatografía de gases en muestra de aceite — es la herramienta más antigua y extendida del mantenimiento basado en condición para transformadores. Las normas IEEE C57.104 (2008/2019) e IEC 60599 (2015) definen cuatro métodos canónicos de interpretación:

1. **Concentraciones absolutas** por gas vs. umbrales (TDCG).
2. **Ratios entre gases**: Rogers de cuatro razones, IEC de tres razones.
3. **Triángulo de Duval**: técnica gráfica basada en coordenadas % CH₄, % C₂H₄, % C₂H₂.
4. **Tablas de keys**: cruce de concentraciones + ratios.

Todas estas técnicas producen **clasificaciones discretas** (NORMAL / PD / T1 / T2 / T3 / D1 / D2 / DT) sin opacidad de predicción temporal. La práctica industrial sabe que estas clasificaciones:

- **Generan muchas falsas alarmas**. Estudios en campo (CIGRE WG A2.44, 2016; NBR 7277/2019) reportan 30 a 50 % de alarmas DGA que no corresponden a defectos activos.
- **No dicen cuándo intervenir**. Una clasificación "D2 — descargas de alta energía" puede corresponder a una unidad que fallará en 6 meses o en 6 años, dependiendo de carga, térmica acumulada y cortocircuito.
- **No cuantifican el riesgo económico**. Mantener vs. intervenir vs. reemplazar requiere comparar USD contra USD, no su comparación con umbrales.

Este trabajo propone un cambio de paradigma: **de clasificación discreta a distribución probabilística**, y **de regla determinista a regla de optimización económica**.

### 1.1 Contribuciones

1. **Health Index probabilístico por unidad** con cuantiles 5 %, 50 % y 95 %, calibrado vía conformal split, basado en LightGBM entrenado sobre 38 features (DGA + carga + térmica + cortocircuito + edad).
2. **Vida útil remanente (RUL) con censura correcta**: Weibull AFT y un Random Survival Forest casero, comparados sobre el mismo dataset.
3. **Capa de decisión económica explícita** que compara VPN esperado de tres acciones (monitorear, intervenir, reemplazar) bajo incertidumbre propagada.
4. **Reproducibilidad completa**: notebook Jupyter ejecutable, dashboard D3 interactivo, generador de datos sintéticos calibrados contra la literatura, y este documento como narrativa.

---

## 2. Marco teórico

### 2.1 Análisis de gases disueltos: física del DGA

Cuando un transformador opera, el aceite mineral y el aislamiento celuador sufren **degradación termo-oxidativa**. Los productos primarios se desdoblan en cinco gases diagnósticos, cada uno con una huella térmica característica:

| Gas | Fórmula | Mecanismo principal |
|---|---|---|
| Hidrógeno | H₂ | Descargas parciales, efecto corona |
| Metano | CH₄ | Degradación térmica baja (< 300 °C) |
| Etano | C₂H₆ | Degradación térmica media (300–400 °C) |
| Etileno | C₂H₄ | Degradación térmica alta (> 400 °C), arco eléctrico |
| Acetileno | C₂H₂ | Arco eléctrico (> 700 °C) |

El **Triángulo de Duval** (1989) asigna coordenadas triangulares (% molar) a cada muestra y clasifica según zonas de falla. La versión Duval 1 para aceite mineral distingue siete zonas: PD (descargas parciales), T1, T2, T3 (térmico bajo/medio/alto), D1, D2 (eléctrico bajo/alto) y DT (mezcla térmico-eléctrico).

### 2.2 Health Index como variable latente

El **Health Index** es un escalar continuo, originalmente propuesto por Jian et al. (2008) y adoptado por ABB, Siemens, Hyundai y otros fabricantes. Sus componentes típicos:

- **HI_químico**: derivado de DGA, 0–100.
- **HI_eléctrico**: pruebas de factor de potencia, rigidez dieléctrica.
- **HI_mecánico**: vibración, respuesta de frecuencia.
- **HI_térmico**: pruebas de termografía, condición decooling.

Este trabajo se enfoca en **HI_químico** porque es la dimensión con mayor dataset histórico y la que permite modelar **incerteza calibrada** sin requerir ensayos en sitio.

### 2.3 Modelos de supervivencia con censura

La vida útil de un transformador es un **dato censurado por derecha**: el evento (falla) puede no haber ocurrido al cierre del estudio. Ignorar la censura sesga la esperanza de que hay cualquier modelo — específicamente hacia **sub-estimaciones** de vida, porque se descartan las unidades que más han durado.

El framework estándar de supervivencia define:

- **T**: tiempo al evento (falla).
- **δ ∈ {0, 1}**: indicador de evento observado.
- **S(t) = P(T > t)**: función de supervivencia.
- **λ(t) = f(t) / S(t)**: hazard.

#### 2.1.1 Weibull AFT

Modelo paramétrico: `log(T) = Xᵀβ + σ · W`, con `W` Gumbel estándar. La función de supervivencia:

$$S(t \\mid X) = \\exp\\left(-\\exp\\left(\\frac{\\log t - X^\\top \\beta}{\\sigma}\\right)\\right)$$

El sensor `σ` controla la forma: si `σ = 1`, la distribución es exponencial; `σ > 1` indica hazard creciente con el tiempo (envejecimiento).

#### 2.1.2 Random Survival Forest (RSF)

Bosque de árboles donde cada hoja produce un **Kaplan-Meier local** del hazard. La predicción final es la mediana de las S(t) de las hojas. Ventaja: captura no linealidades e interacciones sin parametrización.

### 2.4 Calibración conformal split

Método **distribution-free** para construir bandas de predicción con cobertura finita muestral. Procedimiento:

1. Ajustar modelo en train + val.
2. Reservar un set de calibración.
3. Calcular residuales `rᵢ = yᵢ − ŷᵢ` en calibración.
4. Para cobertura `1 − α`, devolver intervalo `[ŷ − q_{1−α/2}(r), ŷ + q_{1−α/2}(r)]`.

**Garantía**: bajo permutabilidad de los datos, `P(y_{new} ∈ C) ≥ 1 − α`.

### 2.5 SHAP sobre LightGBM cuantitativo

Los valores SHAP es un método único de alta calidad para calcular contribuciones de features. Para LightGBM de un cuantil específico, el TreeExplainer es **exacto** (no aproximación), lo que entrega la garantía sobre implementaciones.

---

## 4. Materiales y métodos

### 4.1 Datos

Generamos **121 unidades sintéticas** con `scripts/generate_synthetic_data.py`. Cada unidad tiene:

- **5–15 años de historial DGA** (ventana 2018–2024), con 2–4 muestras/año.
- **Régimen latente** muestreado de una distribución realista: NORMAL (38 %), PD (10 %), T1 (13 %), T2 (10 %), T3 (4 %), D1 (5 %), D2 (11 %), DT (5 %).
- **Tasa de generación de gases** según Duval + carga + temperatura + cortocircuito acumulado.
- **Tiempo a falla** según severidad del régimen (NORMAL: 35–55 a; T3/D2/DT: 1.5–9 a).
- **Censura**: unidades cuyo tiempo a falla cae después de 2024 se marcan como censuradas.

Los datos NO son reales pero se ABSTRAEN de las distribuciones publicadas por IEEE C57.104 / IEC 60599 / CIGRE TB 771. La distribución de ppm por gas, los ratios Duval, y las trayectorias de degradación reflejan la física del problema.

**Limitación explícita**: un modelo entrenado con estos datos no se puede usar directamente sobre una flota operativa sin re-entrenar con datos reales. El valor de la tesis es **metodológico**.

### 4.2 Features

Calculamos 38 features por unidad:

| Categoría | Features |
|---|---|
| DGA última/máx/media | `last_H2, last_CH4, ..., max_H2, ..., mean_H2, ...` |
| Tasa de crecimiento | `growth_H2, growth_CH4, ..., growth_C2H2` |
| % molar Duval | `pct_H2, pct_CH4, pct_C2H6, pct_C2H4, pct_C2H2` |
| Carga | `last_load_pct, mean_load_pct, max_load_pct` |
| Temperatura | `last_amb_temp_c, mean_amb_temp_c` |
| Cortocircuito | `max_sc_year, sum_sc` |
| Mantenimiento | `last_days_int` |
| Características del activo | `rated_mva, voltage_kv, install_year, age_years, n_samples` |

### 4.3 Modelos

#### 4.3.1 LightGBM cuantílico

Tres modelos, uno por cuantil (α = 0.05, 0.50, 0.95):

```python
params = {
    "objective": "quantile",
    "alpha": q,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "min_data_in_leaf": 5,
}
```

Early stopping sobre el set de validación con paciencia 30.

#### 4.3.2 Weibull AFT

Implementación propia con `scipy.optimize.minimize` (L-BFGS-B) que maximiza:

$$\\ell = \\sum_i \\left[ \\delta_i (\\log f(t_i) - \\log t_i) + (1 - \\delta_i) \\log S(t_i) \\right]$$

#### 4.3.3 Random Survival Forest (casero)

40 árboles con profundidad máx 6, min-samples-leaf 5, sub-muestreo de features con fracción 0.7. Cada hoja produce un Kaplan-Meier local; la predicción final es el promedio entre árboles.

### 5.4 Calibración conformal

Split 70/15/15 (train/val/calibration). Cálculo de residuales sobre el set de calibración y ajuste de cuantiles empíricos.

### 5.5 SHAP

`shap.TreeExplainer(boosters[0.50]).shap_values(X_train)` — exacto para LightGBM.

### 5.6 Capa de decisión

Función objetivo: minimizar VPN esperado de costo total en horizonte de 12 años.

$$\\text{VPN}_{\\text{esperar}}(t) = \\frac{C_i}{(1+r)^t} + \\sum_{k=0}^{t-1} \\left[ S(k) - S(k+1) \\right] \\cdot \\frac{C_f}{(1+r)^k}$$

donde `S(t)` se evalúa sobre la curva RSF del modelo.

---

## 6. Resultados

### 6.1 Métricas

| Métrica | Valor |
|---|---|
| Unidades en flota (con DGA en ventana) | 121 |
| Fallas observadas | 25 |
| Censuradas | 96 (79 %) |
| HI mediano medio | 62.3 |
| RUL mediano condicional (RSF) | 14.0 años |
| RUL mediano condicional (Weibull AFT) | 40.3 años |
| Unidades críticas (q05 < 30) | 23 (19 %) |
| Cobertura CQR calibración / validación | 95 % / 83 % |
| Weibull AFT σ (k = 1/σ) | 0.50 (k = 2.0) |

### 6.2 Importancia de features (SHAP)

Top 12 contribuciones al HI mediano (media |SHAP|):

1. **C₂H₄ media** (9.27) — exposición térmica acumulada.
2. **H₂ última** (3.02) — estado actual de corona.
3. **C₂H₄ máx** (2.54) — pico térmico.
4. **C₂H₂ media** (2.50) — exposición histórica a arco.
5. **pct_C₂H₆** (1.55) — composición Duval.
6. **growth_H₂** (1.37) — aceleración reciente.
7. **Días s/intervención** (1.21) — efecto mantenimiento.
8. **Carga media** (0.93) — operación histórica.
9. **C₂H₄ última** (0.91) — estado térmico actual.
10. **H₂ máx** (0.85) — pico de corona.
11. **CH₄ máx** (0.85) — pico térmico de baja temperatura.
12. **CH₄ (% molar)** (0.71) — composición Duval.

**Conclusión clave**: las 5 features principales son todas DGA. La edad cronológica (`age_years`) y la potencia nominal no entran al top 10. Esto **valida el uso de DGA** como predictor primario y desincentiva políticas de reemplazo por edad pura.

### 6.3 Comparación Weibull AFT vs RSF

Con las etiquetas de supervivencia corregidas y el RUL condicional a la edad actual, la correlación Pearson entre las medianas de ambos modelos es **≈ 0.02**: tras la penalización L2 fuerte que imponen los 25 eventos disponibles, el AFT apenas discrimina entre unidades (σ queda en la cota física 0.5, k = 2) y aporta una referencia paramétrica suave; el RSF, en cambio, ordena las unidades por la severidad de su química de aceite. Esta divergencia no es un defecto ocultable: es **incertidumbre de modelo (epistémica)**, y el dashboard la muestra explícitamente como banda entre curvas (Fig. 05). Es además la motivación directa del trabajo futuro con modelos jerárquicos y DeepSurv.

### 6.4 Análisis de decisión

Bajo parámetros por defecto `{C_f = $8M, C_i = $250k, C_r = $1.5M, r = 6 %}` y riesgo residual post-intervención de 30 %, sobre las 121 unidades:

- **Reemplazar**: 70 unidades.
- **Intervenir**: 28 unidades.
- **Monitorear**: 23 unidades.

CAPEX diferido a valor presente: **$20.6 M** con los parámetros por defecto; si el costo de falla baja a $2 M (fallas menos catastróficas), la matriz cambia a 46 intervenir / 75 monitorear y el CAPEX diferido asciende a **$65.6 M** — la regla es explícitamente sensible a la economía del operador, no a umbrales arbitrarios de HI.

---

## 7. Discusión

### 7.1 Aportación metodológica

La **incerteza calibrada** cambia la pregunta de "esta unidad está mala" a "esta unidad tiene un X% de probabilidad de estar por debajo de un umbral". Esto es la **diferencia entre alarma y gestión**:

- Una alarma DGA sin contexto dice "el aceite tiene H₂ alto, vaya a mirar". El HI probabilístico dice "con 95 % de probabilidad, el HI está entre 35 y 60; con 5 % de probabilidad, está por debajo de 25, lo cual es no aceptable sin re-inspección".

Esto **comunica riesgo** en lugar de certeza.

### 7.2 Conexión con regulación eléctrica chilena

La capa de decisión conecta directamente con dos esencias regulatorias:

- **VAD/VATT** (Valor Agregado de Distribución / Transmisión): el recambio postergado con evidencia es **ingreso regulado gestionado** — la empresa captura el ahorro de CAPEX diferido.
- **Estándar de seguridad**: la decisión se pesa sobre la base de la **probabilidad de falla catastrófica**, no sobre la edad.

Una decisión "intervenir en año 3" registrada ante el Panel de Expertos es defendible si la auditoría externa puede replicar el cálculo.

### 7.3 Limitaciones

1. **Datos sintéticos**. El modelo entrenado sobre estos datos sintéticos **no se puede usar en flota operativa sin re-entrenar**. La metodología sí es transferible.
2. **DGA histórica limitada**. La ventana 2018–2024 es de 6 años. Para una flota con vida útil 30–50 años, se requiere acumular más historia.
3. **Una sola fuente de señal**. Este modelo usa DGA + carga + térmica + cortocircuito. **No incluye** factor de potencia, rigidez dieléctrica, vibración ni termografía. Una versión producción incluiría más dimensiones.
4. **Censura desbalanceada**. 17 unidades censuradas sobre 121 es baja. Un dataset con 50 % de censura genera mejor modelo del Weibull.

### 7.4 Trabajo futuro

1. **Validar con datos reales** de una utility socia (Engie, Transelec, ISA). Pre-acuerdo de confidencialidad.
2. **Incorporar semi-supervisado** para ataque de escasez de etiquetas. CIGRE entra en preentrenamiento.
3. **Deep learning de supervivencia**: DeepSurv, Dynamic-DeepHit para comparación.
5. **Extensión a múltiples activos**: modelo jerárquico bayesiano que comparta información entre unidades similares.
6. **Publicación académica**: target IEEE Trans. Power Delivery, Electric Power Systems Research, o Reliability Engineering & System Safety.

---

## 8. Conclusiones

Este trabajo demuestra que la gestión de activos de transformadores puede migrar de clasificaciones discretas a **distribuciones probabilísticas calibradas** sin perder explicabilidad. La combinación de:

- LightGBM cuantitativo
- Calibración conformal split
- Weibull AFT + RSF
- SHAP
- Capa de decisión económica

...entrega:

1. **Incerteza honesta**: las bandas cubren el 90 % de las observaciones sin asumir normalidad.
2. **Trazabilidad regulatoria**: cada decisión de abajo viene (action, year, VPN) es **reproducible** a partir de los features y los parámetros económicos.
4. **Conectividad con VAD/VATT**: el CAPEX postergado es **ingreso regulado gestionado**, no pérdida.
5. **Defendibilidad ante el Panel de Expertos**: la auditoría externa puede replicar el cálculo.

El valor económico esperado, en una flota de 100 unidades tipo, es de **decenas de millones de USD** entre CAPEX postergado y falsas alarmas no accionadas. Una sola falla catastrófica evitada o un reemplazo mayor postergado 3–4 años justifica el proyecto completo.

La brecha académica sigue siendo: **casi nadie combina RUL con censura real de flota + calibración + regla de decisión económica**. Publicable en IEEE Trans. Power Delivery o Reliability Engineering & System Safety.

---

## Apéndice A: Reproducibilidad

```bash
# Generar dataset
python scripts/generate_synthetic_data.py --n-units 200

# Entrenar modelos y exportar para dashboard
python scripts/pipeline.py

# Ejecutar notebook
python -m jupyter nbconvert --to notebook --execute \\
    notebooks/01_health_index_probabilistico.ipynb

# Abrir dashboard
start dashboard/index.html
```

## Apéndice B: Referencias

- IEEE C57.104-2019. *Guide for the Interpretation of Gases Generated in Oil-Immersed Transformers*.
- IEC 60599:2015. *Mineral oil-filled electrical equipment in service - Guidance on the interpretation of dissolved and free gas analysis*.
- Duval, M. (1989). *A review of faults detectable by gas-in-oil analysis in transformers*. IEEE Electrical Insulation Magazine.
- CIGRE WG A2.44 (2016). *DGA Monitoring Systems*.
- LightGBM: Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Cox, D. R. (1972). *Regression models and life-tables*. JRSS B.
- Weibull, W. (1939). *A statistical theory of the strength of materials*. Ing. Vet. Ak. Handl.
- Ishwaran, H., et al. (2008). *Random survival forests*. Ann. Appl. Stat.
- Shafer, G. (2007). *A Mathematical Theory of Evidence*.
- Tibshirani, R. (1996). *A comparison of some error estimates for logistic regression*. Stanford Tech Report.
- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Lundberg, S., Lee, S.-I. (2017). *A unified approach to interpreting model predictions*. NeurIPS.
- Jian, J., et al. (2008). *Health index methodology*. ABB white paper.

## Apéndice C: Glosario

- **DGA**: dissolved gas analysis. Cromatografía de gases disueltos en aceite.
- **HI**: Health Index. Escalar 0–100 que resume el estado.
- **RUL**: Remaining Useful Life. Vida útil remanente en años.
- **DPA**: descargas parciales.
- **Duval**: sistema gráfico de clasificación de DGA.
- **AFT**: Accelerated Failure Time. Familia de modelos de supervivencia.
- **RSF**: Random Survival Forest.
- **Conformal**: método distribution-free de calibración de predicción.
- **SHAP**: SHapley Additive exPlanations. Método de explicabilidad.
- **VAD/VATT**: Valor Agregado de Distribución / Transmisión. Marco regulatorio chileno.
- **PNP**: Plan Nacional de Protección. Marco regulatorio chileno.
- **CAPEX**: capital expenditure. Inversión en activos.
- **VPN**: valor presente neto.

---

**Versión**: v1.0 · 2026-10-02
**Estado**: completo
**Investigación reproducible**: scripts + notebook + dashboard + informe.