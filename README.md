# Salud Probabilística de Transformadores

> Health Index con incertidumbre calibrada · Vida útil remanente **condicional** con censura correcta · Decisión económica explícita.

Investigación reproducible que migra la gestión de activos de transformadores desde clasificaciones DGA deterministas a **distribuciones probabilísticas calibradas**, con conexión directa al marco VAD/VATT chileno.

---

## ¿Qué problema ataca?

| Problema | Solución propuesta |
|---|---|
| Clasificación DGA produce 30–50 % de falsas alarmas | Health Index cuantílico (q05/q50/q95) con bandas **CQR** (Romano et al. 2019) al 90 % |
| No se sabe cuándo intervenir | Vida útil remanente **condicional** S(a+r)/S(a) con Weibull AFT + Random Survival Forest (log-rank) y censura correcta |
| Reglas deterministas de intervención son arbitrarias | Capa de decisión que compara VPN esperado de monitorear / intervenir / reemplazar, con riesgo pre-acción y hazard residual |
| Caja negra inaceptable para un regulador | SHAP TreeExplainer sobre LightGBM cuantílico |

## Entregables

1. **Dashboard D3 interactivo** (`dashboard/index.html`): flota, HI con bandas de incertidumbre, triángulo de Duval, series temporales DGA, curva de supervivencia, capa de decisión, SHAP.
2. **Notebook Jupyter ejecutable** (`notebooks/01_health_index_probabilistico.ipynb`): análisis completo de inicio a fin con figuras.
3. **Informe tipo tesis** (`docs/informe_tesis.md`): narrativa formal con marco regulatorio, matemática y referencias.
4. **Pipeline reproducible** (`scripts/`): generador de datos + entrenamiento + export.

---

## Stack técnico

- **LightGBM 4.7.0**: cuantílicos α = 0.05, 0.50, 0.95 sobre 38 features.
- **Conformalized Quantile Regression**: un único δ con corrección de muestra finita; monotonía q05 ≤ q50 ≤ q95 reparada.
- **Weibull AFT**: implementación propia con `scipy.optimize`, features estandarizadas, σ acotado al rango físico (k = 1/σ ∈ [0.5, 2]).
- **Random Survival Forest**: casero — bootstrap + splitting por **test log-rank** + Kaplan-Meier por hoja.
- **RUL condicional**: S(a+r)/S(a) con a = edad actual; la edad no entra como feature de supervivencia (evita la fuga time ≡ age_years en censuradas).
- **SHAP TreeExplainer**: explicabilidad exacta sobre LightGBM.
- **D3.js 7**: dashboard interactivo, edición «papel milimetrado».
- **Python 3.14**: núcleo del pipeline (sembrado y reproducible).

> **Nota técnica**: `lifelines` y `scikit-survival` requieren compilar `ecos` (extensión C con MSVC) y no se instalaron en este entorno. La tesis usa implementaciones equivalentes en scipy puro, documentadas en `scripts/pipeline.py`. Para ambientes con `lifelines` instalado, el mismo flujo es directo de portar.

---

## Cómo ejecutar

### 1. Generar datos sintéticos

```bash
python scripts/generate_synthetic_data.py --n-units 200 --study-end-year 2024
```

Genera 200 unidades con 5–15 años de historial DGA, censura y regímenes Duval realistas.

### 2. Entrenar modelos y exportar al dashboard

```bash
python scripts/pipeline.py
```

Produce:
- `models/*.pkl` — modelos serializados (LightGBM × 3, Weibull AFT, RSF, SHAP explainer).
- `dashboard/data/fleet_data.json` — payload para el dashboard.

### 3. Ejecutar el notebook

```bash
python -m jupyter nbconvert --to notebook --execute \
    notebooks/01_health_index_probabilistico.ipynb \
    --output 01_health_index_probabilistico_executed.ipynb
```

Las figuras se guardan en `notebooks/results/`.

### 4. Abrir el dashboard

El dashboard carga `data/fleet_data.json` por `fetch`, que los navegadores bloquean sobre `file://`. Sírvelo por HTTP:

```bash
cd dashboard
python -m http.server 8000
# abre http://localhost:8000
```

Edición «papel milimetrado»: hero con haz de barrido sobre la flota, triángulo de Duval con **coordenadas canónicas IEC 60599**, curvas de supervivencia condicionales con banda de desacuerdo entre modelos, capa de decisión con hazard residual, SHAP, filtros por zona Duval, navegación por teclado (↑/↓, `/`) y tooltips en todas las figuras.

### 5. Desplegar en Vercel

El repositorio ya incluye `vercel.json` (`outputDirectory: dashboard`), D3 v7 self-hosted
(`js/vendor/`), favicon, metadatos OG y payloads redondeados (~740 KB, ~150 KB con brotli).
Dos formas equivalentes:

- **Vercel CLI**: `npm i -g vercel && vercel --prod` desde la raíz del repo.
- **Git integrado**: importa `github.com/mortizcoilla/transformadores` en vercel.com →
  Framework Preset: *Other* → Deploy (sin build command; la configuración ya está en `vercel.json`).

Sin pasos adicionales: es un sitio 100 % estático. Las únicas referencias externas son las
fuentes de Google Fonts (con `display=swap` y fallback de sistema).

---

## Estructura

```
Transformadores/
├── README.md                                 # Este archivo
├── data/
│   └── processed/                            # CSVs y meta.json (generados)
│       ├── dga_readings.csv                  # 2006 lecturas DGA
│       ├── failure_outcomes.csv              # 200 outcomes (104 fallas, 96 censurados)
│       ├── fleet_specs.csv                   # 200 especificaciones de unidades
│       └── meta.json                         # Metadatos del dataset
├── notebooks/
│   ├── 01_health_index_probabilistico.ipynb  # Notebook ejecutable
│   ├── 01_health_index_probabilistico_executed.ipynb  # Notebook ejecutado
│   ├── build_notebook.py                     # Generador del notebook
│   └── results/                              # Figuras PNG (generadas)
├── scripts/
│   ├── generate_synthetic_data.py            # Generador de flota sintética
│   └── pipeline.py                           # Entrenamiento + export
├── dashboard/
│   ├── index.html                            # Layout principal (edición papel milimetrado)
│   ├── css/styles.css                        # Sistema de diseño papel/tinta
│   ├── js/
│   │   ├── duval.js                          # Zonas Duval canónicas + clasificador
│   │   ├── charts.js                         # Visualizaciones D3 (7 figuras)
│   │   ├── decision.js                       # Capa de decisión económica
│   │   └── app.js                            # Orquestador (estado, filtros, teclado)
│   └── data/fleet_data.json                  # Payload para el dashboard
├── docs/
│   └── informe_tesis.md                      # Informe tipo tesis (v1.0)
└── models/                                   # Modelos serializados (generados)
    ├── lgbm_q05.pkl, lgbm_q50.pkl, lgbm_q95.pkl
    ├── weibull_aft.pkl
    ├── rsf.pkl
    └── explainer.pkl
```

---

## Resultados clave (sobre 121 unidades con DGA, ventana 2018–2024)

| Métrica | Valor |
|---|---|
| Unidades | 121 |
| Fallas observadas | 25 |
| Censuradas | 96 (79 %) |
| HI mediano medio | 62.3 |
| RUL mediano condicional (RSF) | 14.0 años |
| RUL mediano condicional (Weibull) | 40.3 años |
| Unidades críticas (q05 < 30) | 23 (19 %) |
| Cobertura CQR calib. / valid. | 95 % / 83 % |
| Weibull AFT σ (k = 1/σ) | 0.50 (k = 2.0) |

**Features top (SHAP)**: C₂H₄ media, H₂ última, C₂H₄ máx, C₂H₂ media, pct_C₂H₆. La edad cronológica no entra al top 10 — esto valida que **la química del aceite supera a la edad como predictor de estado**.

**Divergencia AFT/RSF**: la correlación entre medianas de RUL de ambos modelos es ≈ 0 (el AFT, fuertemente penalizado con 25 eventos, aporta una referencia paramétrica; el RSF discrimina por química). El dashboard muestra esa divergencia como banda de incertidumbre de modelo — honestidad epistémica por diseño.

---

## Conexión regulatoria

La capa de decisión conecta explícitamente con dos mecanismos del marco regulatorio chileno:

- **VAD/VATT** (Valor Agregado de Distribución / Transmisión): el reemplazo postergado con evidencia es **ingreso regulado gestionado** — la empresa captura el ahorro de CAPEX diferido.
- **PNP semestral**: la decisión registrada ante el Panel de Expertos debe ser reproducible — el método entrega esa garantía.

---

## Limitaciones y honestidad

1. **Datos sintéticos**: el modelo NO se puede desplegar directamente sobre una flota operativa. La metodología es transferible; los datos requieren re-entrenamiento con casos reales.
2. **Una sola fuente**: solo DGA + carga + térmica + cortocircuito + mantenimiento. Faltan factor de potencia, rigidez dieléctrica, vibración, termografía.
3. **Pocos eventos (25) y censura dominante (79 %)**: el AFT queda dominado por su forma/prior (σ en la cota física); el RSF es el discriminante operativo. La banda entre modelos comunica esa debilidad en lugar de ocultarla.
4. **Solo aceite**: el modelo asume aceite mineral. No aplica a transformadores secos o de resina.
5. **Supuestos de la capa de decisión**: intervenir reduce el hazard 70 % (no lo elimina), reemplazo lo elimina dentro del horizonte, sin valor residual.

---

## Trabajo futuro

1. Validar con una utility real (Engie, Transelec, ISA) bajo NDA.
2. Incorporar semi-supervisado con datasets CIGRE para ataque de escasez de etiquetas.
3. DeepSurv / Dynamic-DeepHit para comparación con deep learning de supervivencia.
4. Modelo jerárquico bayesiano que comparta información entre unidades similares.

---

## Referencias seleccionadas

- IEEE C57.104-2019 · *Guide for the Interpretation of Gases Generated in Oil-Immersed Transformers*
- IEC 60599:2015 · *DGA interpretation*
- Duval, M. (1989) · *A review of faults detectable by gas-in-oil analysis in transformers*
- CIGRE WG A2.44 (2016) · *DGA Monitoring Systems*
- LightGBM: Ke et al. (2017) · NeurIPS
- Ishwaran, H. et al. (2008) · *Random survival forests* · Ann. Appl. Stat.
- Vovk, Gammerman, Shafer (2005) · *Algorithmic Learning in a Random World*
- Lundberg, Lee (2017) · SHAP · NeurIPS

---

**Versión**: v1.1 · 2026-10-02 · *(v1.1: etiquetas de supervivencia corregidas, RUL condicional, CQR, log-rank real, Duval canónico, decisión con riesgo pre-acción y hazard residual, dashboard «papel milimetrado»)*
**Tipo**: Investigación reproducible + dashboard + tesis
**Próxima versión**: v1.2 con datos reales de utility socia (pendiente NDA).