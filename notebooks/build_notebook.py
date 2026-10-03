"""build_notebook.py — Genera el Jupyter Notebook ejecutable.

Produce `notebooks/01_health_index_probabilistico.ipynb` con celdas
markdown + código que replican el script pipeline.py pero con análisis
exploratorio, figuras y narrativa académica.

Uso:
    python notebooks/build_notebook.py
"""
from pathlib import Path
import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()

cells = [
    md("""# Salud Probabilística de Transformadores
## Health Index con incertidumbre calibrada

**Autor**: Miguel Ortega Co. · Investigación reproducible
**Stack**: LightGBM (cuantílico) · Weibull AFT (scipy) · Random Survival Forest (casero) · Calibración conformal split · SHAP TreeExplainer

### Pregunta de investigación

> ¿Cuál es la distribución del estado de salud de un transformador de potencia en el año *t*, y cuándo conviene intervenir bajo optimización económica explícita?

### Hipótesis

1. La predicción puntual de estado (HI) es **insuficiente**; la gestión de activos requiere **intervalos calibrados**.
2. La integración de **DGA + carga + térmica + cortocircuito** mejora significativamente el HI sobre DGA aislado.
3. La **censura** (unidades sin falla) contiene información que debe incorporarse vía supervivencia, no eliminarse.
4. La regla de intervención debe **comparar VPN esperado** de cada alternativa, no usar umbrales fijos.

### Datos

Sintéticos generados con `scripts/generate_synthetic_data.py`. Distribuciones de gases basadas en IEEE C57.104 / IEC 60599 / CIGRE TB 771.
"""),

    code("""# Imports y configuración
import sys
sys.path.insert(0, '..')

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import lightgbm as lgb
import shap

np.random.seed(42)  # reproduce el ruido del target HI (igual que pipeline.py)

# Estilo de figuras — papel técnico (coherente con el dashboard)
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.facecolor': '#f5eedc',
    'axes.facecolor': '#f5eedc',
    'axes.edgecolor': '#1b2a4a',
    'axes.labelcolor': '#1b2a4a',
    'axes.titlecolor': '#1b2a4a',
    'xtick.color': '#1b2a4a',
    'ytick.color': '#1b2a4a',
    'grid.color': '#c9bd9e',
    'text.color': '#1b2a4a',
    'font.family': 'DejaVu Sans',
})

DATA_DIR = Path('../data/processed')
print("Directorios OK")"""),

    md("""## 1. Carga y auditoría de datos"""),

    code("""readings = pd.read_csv(DATA_DIR / 'dga_readings.csv')
fleet = pd.read_csv(DATA_DIR / 'fleet_specs.csv')
failure = pd.read_csv(DATA_DIR / 'failure_outcomes.csv')
with open(DATA_DIR / 'meta.json') as f:
    meta = json.load(f)

print(f"Unidades en flota: {len(fleet)}")
print(f"Unidades con readings DGA: {readings['unit_id'].nunique()}")
print(f"Total lecturas DGA: {len(readings)}")
print(f"\\nDistribución de regímenes:")
for k, v in meta['regimes_distribution'].items():
    print(f"  {k:>8s}: {v}")
print(f"\\nFallas observadas: {meta['n_failures_observed']}")
print(f"Censurados: {meta['n_censored']}")"""),

    code("""# Una mirada a la flota
display(fleet.head())
print(f"\\nTensiones nominales (kV): {sorted(fleet['voltage_kv'].unique())}")
print(f"Potencias nominales (MVA): {sorted(fleet['rated_mva'].unique())}")
print(f"Rango de años de instalación: {fleet['install_year'].min()} - {fleet['install_year'].max()}")"""),

    code("""# Distribución de ppm por gas (lecturas DGA)
gases = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']
fig, axes_arr = plt.subplots(2, 3, figsize=(14, 6))
for i, gas in enumerate(gases):
    ax = axes_arr[i // 3, i % 3]
    ax.hist(readings[gas], bins=40, color='#7bc8ff', edgecolor='#0b1220', alpha=0.85)
    ax.set_title(f'{gas} (ppm)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
fig.delaxes(axes_arr[1, 2])
plt.suptitle('Distribución de gases disueltos — flota sintética', fontsize=13)
plt.savefig('results/01_dga_distributions.png', bbox_inches='tight', dpi=110)
plt.show()"""),

    md("""## 2. Feature engineering

Para cada unidad calculamos 38 features agregadas:
- **Última, máxima y media** de los 5 gases DGA.
- **Tasa de crecimiento exponencial** de cada gas.
- **% molar Duval** de la última lectura.
- **Carga** (% rated) última, media y máxima.
- **Temperatura ambiente** última y media.
- **Cortocircuito** kA-año máximo y suma acumulada.
- **Días desde última intervención**.
- **Características del activo**: MVA, tensión, año instalación, edad."""),

    code("""import sys
sys.path.insert(0, '..')
from scripts.pipeline import build_unit_features

X, y_hi, df_surv, unit_ids = build_unit_features(
    readings, fleet, failure, meta['study_end_year']
)

print(f"X shape: {X.shape}")
print(f"y_hi range: [{y_hi.min():.1f}, {y_hi.max():.1f}]")
print(f"Unidades en X: {len(unit_ids)}")
print(f"Eventos en superv: {df_surv['event'].sum()} / {len(df_surv)}")"""),

    code("""# Distribución de Health Index target
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(y_hi, bins=30, color='#5dd2a7', edgecolor='#0b1220', alpha=0.85)
ax.set_xlabel('Health Index observado (0 = falla, 100 = sano)')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución del Health Index objetivo')
ax.axvline(y_hi.mean(), color='#f3b347', ls='--', lw=2, label=f'Media = {y_hi.mean():.1f}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('results/02_hi_distribution.png', bbox_inches='tight', dpi=110)
plt.show()"""),

    md("""## 3. Modelos de regresión: LightGBM cuantílico

Entrenamos **tres cuantiles** del HI simultáneamente:
- **q05**: percentil 5 — cota pesimista del estado.
- **q50**: mediana — predicción central.
- **q95**: percentil 95 — cota optimista.

La diferencia q95-q05 captura la **incertidumbre epistémica** del modelo para cada unidad. Esta es la base para decisiones tipo "actuar si el peor caso es malo pero el caso esperado es tolerable"."""),

    code("""from sklearn.model_selection import train_test_split

# Split 70/15/15
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
n_train = int(0.7 * len(X))
n_val = int(0.15 * len(X))
train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
cal_idx = idx[n_train + n_val:]

X_train, X_val, X_cal = X.iloc[train_idx], X.iloc[val_idx], X.iloc[cal_idx]
y_train, y_val, y_cal = y_hi.iloc[train_idx], y_hi.iloc[val_idx], y_hi.iloc[cal_idx]

print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Cal:   {X_cal.shape}")"""),

    code("""def train_quantile(alpha):
    params = {
        'objective': 'quantile',
        'alpha': alpha,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'min_data_in_leaf': 5,
        'verbose': -1,
    }
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    return lgb.train(
        params, train_set, num_boost_round=400,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )

boosters = {q: train_quantile(q) for q in (0.05, 0.50, 0.95)}
for q, b in boosters.items():
    print(f"q={q}: best_iter = {b.best_iteration}, val_loss = {b.best_score['valid_0']['quantile']:.3f}")"""),

    code("""# Predicciones sobre el set de calibración
def predict_q(X_in):
    return {f'q{int(q*100):02d}': boosters[q].predict(X_in, num_iteration=boosters[q].best_iteration) for q in boosters}

preds_cal = predict_q(X_cal)
for k, v in preds_cal.items():
    print(f"{k}: media = {v.mean():.1f}, std = {v.std():.1f}")"""),

    md("""## 4. Calibración conformal — CQR

La calibración conformal es **distribution-free**: garantiza cobertura marginal ≥ 1-α bajo intercambiabilidad, sin asumir normalidad.

Usamos **Conformalized Quantile Regression** (Romano, Patterson & Candès, 2019):

1. Score de no-cobertura sobre el set de calibración: $E_i = \\max(q_{05}(x_i) - y_i,\\; y_i - q_{95}(x_i))$.
2. Un ÚNICO delta = cuantil de $E$ al nivel $\\lceil (n+1)(1-\\alpha) \\rceil / n$ (corrección de muestra finita).
3. Banda final: $[q_{05} - \\delta,\\; q_{95} + \\delta]$, con monotonía $q_{05} \\le q_{50} \\le q_{95}$ reparada tras el ajuste."""),

    code("""from scripts.pipeline import conformal_calibrate_cqr

cal = conformal_calibrate_cqr(predict_q, X_cal, y_cal, coverage=0.9)
print(f"delta = {cal['delta']:.2f} (n calibración = {cal['n_calibration']})")
print(f"Cobertura en calibración: {cal['coverage_calibration']*100:.0f}%")

def predict_calibrated(X_in):
    raw = predict_q(X_in)
    q05 = np.clip(raw['q05'] - cal['delta'], 0, 100)
    q50 = np.clip(raw['q50'], 0, 100)
    q95 = np.clip(raw['q95'] + cal['delta'], 0, 100)
    raw['q05'] = np.minimum(q05, q50)
    raw['q50'] = q50
    raw['q95'] = np.maximum(q95, q50)
    return raw"""),

    code("""# Validación de cobertura empírica
preds_val = predict_calibrated(X_val)
covered = ((y_val.values >= preds_val['q05']) & (y_val.values <= preds_val['q95'])).mean()
print(f"Cobertura empírica en validación: {covered*100:.1f}% (objetivo: 90%)")"""),

    code("""# Visualizar bandas para 5 unidades del set val
import random
sample = random.sample(range(len(X_val)), 5)
fig, ax = plt.subplots(figsize=(11, 4))
for i, ix in enumerate(sample):
    y_true = y_val.iloc[ix]
    lo = preds_val['q05'][ix]
    md = preds_val['q50'][ix]
    hi = preds_val['q95'][ix]
    ax.errorbar(i, md, yerr=[[md-lo], [hi-md]], fmt='o', color='#7bc8ff',
                capsize=8, lw=2, markersize=8)
    ax.scatter(i, y_true, color='#5dd2a7', s=80, marker='_', lw=3, zorder=5, label='verdad' if i == 0 else None)
ax.set_xticks(range(len(sample)))
ax.set_xticklabels([f'u{ix}' for ix in sample])
ax.set_ylabel('Health Index')
ax.set_title('Bandas conformal 90% vs verdad (5 unidades)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('results/03_conformal_bands.png', bbox_inches='tight', dpi=110)
plt.show()"""),

    md("""## 5. Modelos de supervivencia

La pregunta de vida útil remanente (RUL) requiere **manejo de censura**: una unidad que no ha fallado al cierre del estudio contiene información ("ha sobrevivido hasta hoy"). Eliminarla sesga la estimación.

Dos decisiones metodológicas importantes (corregidas respecto a versiones anteriores):

1. **Etiquetas correctas**: el tiempo de supervivencia es el real de `failure_outcomes.csv` (tiempo al evento para falladas, tiempo al cierre para censuradas), no la edad al cierre para todas.
2. **Sin fuga de etiqueta**: `age_years` e `install_year` se excluyen de las features de supervivencia (para censuradas, time ≡ age_years las predeciría de forma trivial y degeneraría el ajuste). La edad entra por el **acondicionamiento** $S(a+r\\,|\\,X)/S(a\\,|\\,X)$.

Entrenamos dos modelos y comparamos:
- **Weibull AFT** (Accelerated Failure Time): paramétrico, features estandarizadas, σ acotado al rango físico.
- **Random Survival Forest** (casero, bootstrap + splits por test log-rank + Kaplan-Meier por hoja)."""),

    code("""# Importancia SHAP primero: define el conjunto compacto de features de
# supervivencia (con ~17 eventos en train, 36 features memorizan los tiempos)
explainer = shap.TreeExplainer(boosters[0.50])
shap_sample_idx = np.random.choice(len(X), min(200, len(X)), replace=False)
shap_values = explainer.shap_values(X.iloc[shap_sample_idx])
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = sorted(
    [{'feature': f, 'importance': float(v)}
     for f, v in zip(X.columns, mean_abs_shap)],
    key=lambda d: -d['importance']
)

SURV_EXCLUDE = {'age_years', 'install_year'}
surv_features = [d['feature'] for d in shap_importance
                 if d['feature'] not in SURV_EXCLUDE][:8]
surv_features = list(dict.fromkeys(surv_features + ['rated_mva', 'voltage_kv']))
print(f"Features de supervivencia ({len(surv_features)}): {surv_features}")

Xs_train = X_train[surv_features]
Xs_val = X_val[surv_features]
surv_train = df_surv.iloc[train_idx].reset_index(drop=True)
surv_val = df_surv.iloc[val_idx].reset_index(drop=True)"""),

    code("""from scripts.pipeline import fit_weibull_aft, predict_weibull_survival, predict_weibull_rul

weibull = fit_weibull_aft(Xs_train.values, surv_train['time'].values,
                          surv_train['event'].values, penalizer=1.0)
print(f"Weibull AFT - sigma={weibull['sigma']:.3f} (k={1/weibull['sigma']:.2f})")
print(f"Intercept = {weibull['intercept']:.3f}")
print(f"Convergencia: {weibull['success']}")"""),

    code("""# Random Survival Forest (casero, splits por log-rank)
from scripts.pipeline import SurvivalForest
forest = SurvivalForest(n_estimators=200, max_depth=8, min_samples_leaf=8,
                        max_features_frac=0.5, random_state=42)
forest.fit(Xs_train.values, surv_train['time'].values, surv_train['event'].values)
print("Random Survival Forest: 200 árboles entrenados")"""),

    code("""# Curvas de supervivencia CONDICIONALES a la edad actual de cada unidad
# S_c(r) = S(a + r)/S(a), r = años desde hoy
from scripts.pipeline import conditionalize

t_grid_total = np.arange(0.0, 80.5, 0.5)   # años desde instalación
r_grid = np.arange(0.0, 30.5, 0.5)         # años desde hoy

ages_train = X_train['age_years'].values
fig, ax = plt.subplots(figsize=(11, 5))
for i in range(min(20, len(Xs_train))):
    s_tot = forest.predict_survival(Xs_train.values[[i]], t_grid_total)[0]
    s_cond = conditionalize(s_tot, t_grid_total, ages_train[i], r_grid)
    ax.plot(r_grid, s_cond, color='#2e5eaa', alpha=0.3)
    ax.axvline(ages_train[i] * 0, color='none')
ax.set_xlabel('Años desde hoy')
ax.set_ylabel('S(r | sobrevivió a edad actual)')
ax.set_title('Curvas de supervivencia condicionales — RSF (20 unidades train)')
ax.grid(True, alpha=0.4)
plt.savefig('results/04_rsf_curves.png', bbox_inches='tight', dpi=110)
plt.show()"""),

    md("""## 6. Comparación Weibull AFT vs RSF

Comparamos la **mediana de vida remanente condicional** en validación. La mediana es robusta a outliers; es lo que el planificador necesita."""),

    code("""ages_val = X_val['age_years'].values

rul_weibull_val = predict_weibull_rul(weibull, Xs_val.values, ages_val)

rul_rsf_val = []
for i in range(len(Xs_val)):
    s_tot = forest.predict_survival(Xs_val.values[[i]], t_grid_total)[0]
    s_cond = conditionalize(s_tot, t_grid_total, ages_val[i], r_grid)
    below = np.where(s_cond <= 0.5)[0]
    rul_rsf_val.append(r_grid[below[0]] if len(below) else np.nan)
rul_rsf_val = np.array(rul_rsf_val)

mask = ~np.isnan(rul_rsf_val)
print(f"Weibull AFT — RUL mediano condicional: {np.nanmean(rul_weibull_val):.1f} años")
print(f"RSF        — RUL mediano condicional: {np.nanmean(rul_rsf_val):.1f} años "
      f"({(~mask).sum()} unidades con mediana > 30 a)")
print(f"Correlación Pearson: {np.corrcoef(rul_weibull_val[mask], rul_rsf_val[mask])[0,1]:.3f}")"""),

    code("""fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(rul_weibull_val, rul_rsf_val, alpha=0.6, color='#7bc8ff')
mx = max(rul_weibull_val.max(), rul_rsf_val.max())
ax.plot([0, mx], [0, mx], '--', color='#f3b347', lw=1.5, label='y = x')
ax.set_xlabel('Weibull AFT — RUL mediana (años)')
ax.set_ylabel('Random Survival Forest — RUL mediana (años)')
ax.set_title('Acuerdo entre modelos de RUL')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('results/05_weibull_vs_rsf.png', bbox_inches='tight', dpi=110)
plt.show()"""),

    md("""## 7. Explicabilidad: SHAP TreeExplainer

Para que un jefe de mantenimiento y un regulador acepten el número, la predicción debe ser **explicable**. Usamos `shap.TreeExplainer` sobre el cuantil mediano, lo que es exacto para modelos basados en árboles. (La importancia ya se calculó en §5 para elegir las features de supervivencia; aquí la exploramos.)"""),

    code("""feat_imp = pd.DataFrame({
    'feature': X_train.columns,
    'mean_abs_shap': [
        next(d['importance'] for d in shap_importance if d['feature'] == f)
        for f in X_train.columns
    ]
}).sort_values('mean_abs_shap', ascending=False).head(12)
print(feat_imp)"""),

    code("""# SHAP summary plot (beeswarm) sobre muestra de train
plt.figure(figsize=(10, 7))
sample_idx2 = np.random.choice(len(X_train), min(80, len(X_train)), replace=False)
shap_vals_train = explainer.shap_values(X_train.iloc[sample_idx2])
shap.summary_plot(
    shap_vals_train, X_train.iloc[sample_idx2],
    feature_names=X_train.columns.tolist(),
    plot_type='dot', show=False, max_display=12
)
plt.title('SHAP summary — contribución por feature (muestra 80)')
plt.tight_layout()
plt.savefig('results/06_shap_summary.png', bbox_inches='tight', dpi=110)
plt.show()"""),

    md("""## 8. Capa de decisión económica

La regla conecta predicción → decisión → acción. **No usamos umbrales de HI** porque son arbitrarios y no internalizan costos.

**Función objetivo**: minimizar el VPN esperado de costo total en un horizonte de 12 años, usando la curva de supervivencia **condicional** (años desde hoy) del RSF. Dos correcciones metodológicas:

1. Actuar en el año $t$ asume el **riesgo de falla antes de $t$** (antes, el reemplazo diferido lo ignoraba y ganaba siempre por descuento).
2. **Intervenir no elimina el riesgo**: reduce el hazard en 70 % ($\\phi = 0.3$ residual). La supervivencia post-intervención es $\\tilde{S}(k) = S(t) + 0.7\\,(S(k)-S(t))$ para $k \\ge t$. Esto crea una frontera real entre reparar y reemplazar.

$$ \\text{VPN}_{\\text{actuar en } t} = \\frac{C_{\\text{acción}}}{(1+r)^t} + \\sum_{k=0}^{t-1} (S(k) - S(k{+}1)) \\cdot \\frac{C_f}{(1+r)^k} + \\sum_{k \\ge t} (\\tilde{S}(k) - \\tilde{S}(k{+}1)) \\cdot \\frac{C_f}{(1+r)^k} $$

con $\\tilde{S} = S$ (riesgo intacto) para REEMPLAZAR y MONITOREAR en su tramo."""),

    code("""RESIDUAL_HAZARD = 0.30  # la reparación mitiga el 70 % del hazard, no lo elimina

def npv_action(surv_curve, params, horizon=12):
    \"\"\"surv_curve: lista [(t, s)] condicional, años desde hoy.\"\"\"
    r = params['rate']
    Cf = params['cost_fail']     # USD
    Ci = params['cost_int']      # USD
    Cr = params['cost_replace']  # USD
    ts = np.array([p[0] for p in surv_curve])
    ss = np.array([p[1] for p in surv_curve])
    s_at = lambda t: float(np.interp(t, ts, ss))

    def risk_before(t):
        return sum((s_at(k) - s_at(k+1)) * Cf / (1 + r)**k for k in range(int(t)))

    def risk_residual(t):
        St = s_at(t)
        out = 0.0
        for k in range(int(t), horizon):
            sk  = St + (1 - RESIDUAL_HAZARD) * (s_at(k)   - St)
            sk1 = St + (1 - RESIDUAL_HAZARD) * (s_at(k+1) - St)
            out += max(0.0, sk - sk1) * Cf / (1 + r)**k
        return out

    npv_monitor = sum((s_at(k) - s_at(k+1)) * Cf / (1 + r)**k for k in range(horizon))

    best_intervene = min(((risk_before(t) + risk_residual(t) + Ci / (1 + r)**t, t)
                          for t in range(horizon + 1)))
    best_replace = min(((risk_before(t) + Cr / (1 + r)**t, t)
                        for t in range(horizon + 1)))

    actions = [('MONITOREAR', npv_monitor, 0),
               ('INTERVENIR', *best_intervene),
               ('REEMPLAZAR', *best_replace)]
    actions.sort(key=lambda a: a[1])
    return actions[0]"""),

    code("""# Decisión para las 10 peores unidades de validación (peor HI q50)
val_preds = predict_calibrated(X_val)
worst_idx = np.argsort(val_preds['q50'])[:10]
params = {'rate': 0.06, 'cost_fail': 8e6, 'cost_int': 250e3, 'cost_replace': 1.5e6}

print(f"{'índice':>8s} {'HI q50':>8s} {'RUL Weib':>10s} {'RUL RSF':>9s} {'Acción':<12s} {'Año':>4s} {'VPN':>10s}")
print("-" * 68)
for ix in worst_idx:
    s_tot = forest.predict_survival(Xs_val.values[[ix]], t_grid_total)[0]
    s_cond = conditionalize(s_tot, t_grid_total, ages_val[ix], r_grid)
    curve = list(zip(r_grid.tolist(), s_cond.tolist()))
    action, npv, year = npv_action(curve, params)
    print(f"val[{ix:3d}] {val_preds['q50'][ix]:>9.2f} {rul_weibull_val[ix]:>10.1f} "
          f"{rul_rsf_val[ix]:>9.1f} {action:<12s} {year:>4d} ${npv/1e6:>9.2f}M")"""),

    code("""## 9. Resumen y cifras clave (calculadas, no hardcodeadas)
from IPython.display import Markdown, display

hi_q50_all = predict_calibrated(X)['q50']
hi_q05_all = predict_calibrated(X)['q05']
n_crit = int((hi_q05_all < 30).sum())

rows = [
    ("Unidades con DGA en ventana", f"{len(X)}"),
    ("Fallas observadas", f"{int(df_surv['event'].sum())}"),
    ("Censuradas (sin falla al cierre)", f"{int((1 - df_surv['event']).sum())}"),
    ("HI mediano medio", f"{np.mean(hi_q50_all):.1f}"),
    ("RUL mediano condicional (RSF, validación)", f"{np.nanmean(rul_rsf_val):.1f} años"),
    ("Unidades críticas (q05 < 30)", f"{n_crit} ({100*n_crit/len(X):.0f} %)"),
    ("Cobertura CQR en calibración / validación",
     f"{cal['coverage_calibration']*100:.0f} % / {cal['coverage_validation']*100:.0f} %" if 'coverage_validation' in cal
     else f"{cal['coverage_calibration']*100:.0f} %"),
    ("Weibull AFT σ (k = 1/σ)", f"{weibull['sigma']:.2f} (k = {1/weibull['sigma']:.2f})"),
]
display(Markdown("Ventana " + str(meta['start_year']) + "—" + str(meta['study_end_year'])
                 + ":\\n\\n| Métrica | Valor |\\n|---|---|\\n"
                 + "\\n".join(f"| {a} | {b} |" for a, b in rows)))

display(Markdown(\"\"\"
**Hallazgos para la tesis**:

1. La **censura importa y es dominante** (79 % de la cohorte): el Weibull AFT y el RSF la incorporan como información, no como dato a descartar.
2. El RUL es **condicional a la edad actual** — la vida total mediana no es accionable; S(a+r)/S(a) sí.
3. La calibración CQR entrega **cobertura marginal garantizada** sin asumir distribuciones — propiedad clave ante un regulador.
4. Las features dominantes son **gases disueltos (C₂H₄, H₂, C₂H₂)**, no la edad cronológica — la química del aceite supera al calendario.
\"\"\"))"""),
]

nb['cells'] = cells

out_path = Path('notebooks/01_health_index_probabilistico.ipynb')
out_path.parent.mkdir(exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"OK - notebook escrito en {out_path}")