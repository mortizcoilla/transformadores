"""
Pipeline de entrenamiento + export para el dashboard.

Componentes:
- LightGBM cuantílico (q05, q50, q95) para Health Index.
- Conformalized Quantile Regression (Romano et al. 2019) para bandas al 90 %.
- Weibull AFT implementado con scipy (features estandarizadas).
- Random Survival Forest casero: bootstrap + splitting por log-rank + KM por hoja.
- RUL CONDICIONAL: S(a + r | X) / S(a | X) — vida remanente dado que la
  unidad ya sobrevivió hasta su edad actual `a`. Esto corrige el error de
  versiones anteriores que reportaban la vida total mediana como si fuera
  vida remanente.
- Clasificación Duval Triangle 1 (coordenadas canónicas IEC 60599).
- Export a dashboard/data/fleet_data.json.

Convenciones:
- `time` = años desde instalación hasta evento (falla) o censoring (cierre
  del estudio). Se toma directamente de failure_outcomes.csv, NO se recalcula.
- Las curvas de supervivencia exportadas al dashboard son condicionales a la
  edad actual de cada unidad y están indexadas en "años desde hoy", que es lo
  que consume la capa de decisión económica.
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

warnings.filterwarnings("ignore")


# ============== FEATURE ENGINEERING ==============

DUVAL_LIMITS = {"H2": 100, "CH4": 80, "C2H6": 75, "C2H4": 100, "C2H2": 35}

DUVAL_REGIME_BASE_HI = {
    "NORMAL": 88, "PD": 65, "T1": 60, "T2": 45,
    "T3": 30, "D1": 55, "D2": 35, "DT": 20,
}

# --- Duval Triangle 1 — zonas canónicas (IEC 60599:2015 / IEEE C57.104) ---
# Vértices en (%CH4, %C2H2, %C2H4); cada triple suma 100.
DUVAL_ZONES: Dict[str, List[Tuple[float, float, float]]] = {
    "PD": [(100, 0, 0), (98, 0, 2), (98, 2, 0)],
    "D1": [(0, 100, 0), (0, 77, 23), (64, 13, 23), (87, 13, 0)],
    "D2": [(0, 77, 23), (0, 29, 71), (31, 29, 40), (47, 13, 40), (64, 13, 23)],
    "DT": [(0, 29, 71), (0, 15, 85), (35, 15, 50), (46, 4, 50),
           (96, 4, 0), (87, 13, 0), (47, 13, 40), (31, 29, 40)],
    "T1": [(96, 4, 0), (80, 0, 20), (98, 0, 2), (98, 2, 0)],
    "T2": [(46, 4, 50), (50, 0, 50), (80, 0, 20), (76, 4, 20)],
    "T3": [(0, 15, 85), (0, 0, 100), (50, 0, 50), (35, 15, 50)],
}

DUVAL_ZONE_DESCR = {
    "PD": "Descargas parciales (corona)",
    "D1": "Descarga eléctrica de baja energía",
    "D2": "Descarga eléctrica de alta energía (arco)",
    "DT": "Falla mixta térmica + eléctrica",
    "T1": "Falla térmica < 300 °C",
    "T2": "Falla térmica 300–700 °C",
    "T3": "Falla térmica > 700 °C",
}


def _point_in_poly(p, poly) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_int:
                inside = not inside
    return inside


def classify_duval(ppm: Dict[str, float]) -> str:
    """Clasifica una muestra DGA en zona Duval Triangle 1.

    pd = (%CH4, %C2H2, %C2H4) sobre la suma de los tres gases.
    Si la suma es despreciable (< 1 ppm) no hay firma de falla → 'NORMAL'.
    """
    s = ppm["CH4"] + ppm["C2H2"] + ppm["C2H4"]
    if s < 1.0:
        return "NORMAL"
    p = (100 * ppm["CH4"] / s, 100 * ppm["C2H2"] / s, 100 * ppm["C2H4"] / s)
    # Las zonas comparten aristas; el orden de precedencia resuelve fronteras.
    for zone in ("PD", "D1", "D2", "DT", "T1", "T2", "T3"):
        if _point_in_poly((p[0], p[1]), [(v[0], v[1]) for v in DUVAL_ZONES[zone]]):
            return zone
    # Punto exactamente en el borde (raro por redondeo): zona más cercana por área.
    return "DT"


def _ratio_duval(ppm: Dict[str, float]) -> Dict[str, float]:
    total = sum(ppm.values()) + 1e-3
    return {f"pct_{g}": 100 * ppm[g] / total for g in ppm}


def _aggregates(g: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for gas in ("H2", "CH4", "C2H6", "C2H4", "C2H2"):
        out[f"last_{gas}"] = float(g[gas].iloc[-1])
        out[f"max_{gas}"] = float(g[gas].max())
        out[f"mean_{gas}"] = float(g[gas].mean())
        if len(g) > 1 and g[gas].iloc[0] > 0:
            years = g["years_since_install"].iloc[-1] - g["years_since_install"].iloc[0]
            if years > 0:
                out[f"growth_{gas}"] = float(np.log(g[gas].iloc[-1] / g[gas].iloc[0]) / years)
            else:
                out[f"growth_{gas}"] = 0.0
        else:
            out[f"growth_{gas}"] = 0.0
    out["n_samples"] = float(len(g))
    out["last_load_pct"] = float(g["load_pct"].iloc[-1])
    out["mean_load_pct"] = float(g["load_pct"].mean())
    out["max_load_pct"] = float(g["load_pct"].max())
    out["last_amb_temp_c"] = float(g["amb_temp_c"].iloc[-1])
    out["mean_amb_temp_c"] = float(g["amb_temp_c"].mean())
    out["max_sc_year"] = float(g["short_circuit_kA_year"].max())
    out["sum_sc"] = float(g["short_circuit_kA_year"].sum())
    out["last_days_int"] = float(g["days_since_last_intervention"].iloc[-1])
    return out


def _latest_ppm(g: pd.DataFrame) -> Dict[str, float]:
    return {gas: float(g[gas].iloc[-1]) for gas in ("H2", "CH4", "C2H6", "C2H4", "C2H2")}


def build_unit_features(readings: pd.DataFrame, fleet: pd.DataFrame, failure: pd.DataFrame,
                        study_end_year: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """Devuelve (X, y_hi, survival_df, unit_ids).

    survival_df['time'] usa el tiempo real al evento/censo tomado de
    failure_outcomes.csv (`time_to_failure_years`): para unidades fallidas es
    failure_year - install_year, y para censuradas es study_end - install_year.
    """
    rows, y_hi, surv_rows, unit_ids = [], [], [], []

    for _, spec in fleet.iterrows():
        unit_id = spec["unit_id"]
        g = readings[readings.unit_id == unit_id].sort_values("sample_date")
        f_out = failure[failure.unit_id == unit_id].iloc[0]
        if g.empty:
            continue

        feats = _aggregates(g)
        latest = _latest_ppm(g)
        feats.update(_ratio_duval(latest))

        feats["rated_mva"] = float(spec["rated_mva"])
        feats["voltage_kv"] = float(spec["voltage_kv"])
        feats["install_year"] = int(spec["install_year"])
        feats["age_years"] = float(study_end_year - spec["install_year"])

        regime = spec["regime"]
        base_hi = DUVAL_REGIME_BASE_HI[regime]
        excess_h2 = max(0, latest["H2"] - DUVAL_LIMITS["H2"]) / DUVAL_LIMITS["H2"]
        excess_c2h2 = max(0, latest["C2H2"] - DUVAL_LIMITS["C2H2"]) / DUVAL_LIMITS["C2H2"]
        age = study_end_year - spec["install_year"]
        penalty = 12 * (excess_h2 + excess_c2h2) + 0.15 * age
        hi_observed = float(np.clip(base_hi - penalty + np.random.normal(0, 3), 5, 99))

        y_hi.append(hi_observed)
        rows.append(feats)
        unit_ids.append(unit_id)

        # Etiqueta de supervivencia CORRECTA: tiempo real al evento/censo.
        time = float(f_out["time_to_failure_years"])
        event = 0 if bool(f_out["censored"]) else 1
        surv_rows.append({
            "unit_id": unit_id,
            "time": time,
            "event": event,
            "regime": regime,
        })

    X = pd.DataFrame(rows)
    y = pd.Series(y_hi)
    survival_df = pd.DataFrame(surv_rows)
    return X, y, survival_df, unit_ids


# ============== WEIBULL AFT (sin lifelines) ==============

def fit_weibull_aft(X: np.ndarray, time: np.ndarray, event: np.ndarray,
                    penalizer: float = 1.0) -> Dict:
    """
    Ajusta un modelo Weibull AFT:
        log(T) = X_std @ beta + sigma * W,   W ~ Gumbel estándar (mín.)

    Las features se estandarizan internamente (media 0, varianza 1) para que
    el penalizador L2 sea coherente entre escalas y L-BFGS converja bien.
    Con eventos escasos (~17 en train) el MLE sin restricciones degenera
    (sigma -> 0, separación): se acota sigma al rango físico documentado
    para vida de transformadores (k = 1/sigma en [0.5, 2], i.e. sigma en
    [0.5, 2]) y se penaliza L2 sobre los coeficientes.

    Log-verosimilitud (T ~ Weibull):
        log f(t) = -log(sigma) - log(t) + z - exp(z)
        log S(t) = -exp(z),    z = (log t - Xb) / sigma
    """
    X_arr = np.asarray(X, dtype=float)
    x_mean = X_arr.mean(axis=0)
    x_std = X_arr.std(axis=0)
    x_std[x_std < 1e-9] = 1.0
    Xs = (X_arr - x_mean) / x_std

    n, p = Xs.shape
    X_ = np.hstack([np.ones((n, 1)), Xs])
    p_ = X_.shape[1]

    def neg_log_lik(params: np.ndarray) -> float:
        beta = params[:p_]
        log_sigma = params[-1]
        sigma = np.exp(log_sigma)
        lin = X_ @ beta
        log_t = np.log(np.clip(time, 1e-3, None))
        z = (log_t - lin) / sigma
        log_f = -log_sigma + z - np.exp(z)
        log_S = -np.exp(z)
        ll = event * (log_f - log_t) + (1 - event) * log_S
        reg = penalizer * np.sum(beta[1:] ** 2)
        return -ll.sum() + reg

    # Inicialización: intercepto ≈ log(mediana de tiempos), sigma ≈ 1
    init = np.zeros(p_ + 1)
    init[0] = np.log(np.median(time))
    init[-1] = 0.0
    # sigma acotado al rango físico (k = 1/sigma en [0.5, 2]) evita la
    # degeneración por separación con eventos escasos
    bounds = [(None, None)] * p_ + [(np.log(0.5), np.log(2.0))]
    res = optimize.minimize(neg_log_lik, init, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-7})
    beta_hat = res.x[:p_]
    sigma_hat = np.exp(res.x[-1])
    return {
        "beta": beta_hat,
        "sigma": float(sigma_hat),
        "intercept": float(beta_hat[0]),
        "x_mean": x_mean,
        "x_std": x_std,
        "feature_names": None,  # se setea al fit
        "success": bool(res.success),
    }


def _std_matrix(model: Dict, X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    Xs = (X_arr - model["x_mean"]) / model["x_std"]
    return np.hstack([np.ones((Xs.shape[0], 1)), Xs])


def predict_weibull_median(model: Dict, X: np.ndarray) -> np.ndarray:
    """Mediana de la vida TOTAL (años desde instalación): exp(Xb)·(ln 2)^sigma."""
    lin = _std_matrix(model, X) @ model["beta"]
    return np.exp(lin) * (np.log(2) ** model["sigma"])


def predict_weibull_rul(model: Dict, X: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Mediana de la vida REMANENTE condicional a sobrevivir hasta `age` años.

    S(t) = exp(-(t/eta)^k) con eta = exp(Xb), k = 1/sigma. Resolviendo
    S(a + r)/S(a) = 0.5:
        r = eta * ((a/eta)^k + ln 2)^(1/k) - a
    """
    eta = np.exp(_std_matrix(model, X) @ model["beta"])
    k = 1.0 / model["sigma"]
    a = np.maximum(np.asarray(age, dtype=float), 0.0)
    return eta * ((a / eta) ** k + np.log(2.0)) ** (1.0 / k) - a


def predict_weibull_survival(model: Dict, X: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """S(t | X) = exp(-exp((log(t) - Xb)/sigma)), t = años desde instalación."""
    lin = (_std_matrix(model, X) @ model["beta"])[:, None]
    sigma = model["sigma"]
    log_t = np.log(np.clip(t_grid, 1e-3, None))[None, :]
    z = (log_t - lin) / sigma
    return np.exp(-np.exp(z))


def conditionalize(s_tot: np.ndarray, t_grid: np.ndarray, age: float,
                   r_grid: np.ndarray) -> np.ndarray:
    """Convierte S_total(t) en S_condicional(r) = S(a + r)/S(a).

    Si el modelo ya asigna S(a) ≈ 0 (unidad 'muerta' según el modelo), se
    degrada suavemente usando el último S > eps como ancla.
    """
    s_at = lambda t: float(np.interp(t, t_grid, s_tot))
    s_a = s_at(age)
    eps = 1e-3
    if s_a < eps:
        # ancla en el último tiempo con S >= eps
        idx = np.where(s_tot >= eps)[0]
        age_eff = t_grid[idx[-1]] if len(idx) else 0.0
        s_a = max(s_at(age_eff), eps)
    else:
        age_eff = age
    s_num = np.interp(age_eff + r_grid, t_grid, s_tot, left=1.0, right=s_tot[-1])
    return np.clip(s_num / s_a, 0.0, 1.0)


# ============== RANDOM SURVIVAL FOREST (casero) ==============

def _kaplan_meier(time: np.ndarray, event: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """KM vectorizado sobre la hoja; devuelve S en t_grid (monótona no creciente)."""
    event = event.astype(bool)
    uts = np.unique(time[event])
    if len(uts) == 0:
        return np.ones_like(t_grid)
    at_risk = (time[None, :] >= uts[:, None]).sum(axis=1).astype(float)
    d = ((time[None, :] == uts[:, None]) & event[None, :]).sum(axis=1).astype(float)
    factors = 1.0 - d / np.maximum(at_risk, 1.0)
    s_event_times = np.cumprod(factors)
    return np.interp(t_grid, uts, s_event_times, left=1.0, right=s_event_times[-1])


def _logrank_stat(time: np.ndarray, event: np.ndarray, mask: np.ndarray) -> float:
    """Estadístico log-rank (chi², 1 gl) para el split left = mask.

    O - E y V calculados sobre el grupo izquierdo en cada tiempo de evento:
        O = sum d1j,  E = sum n1j * dj / nj,  V = sum dj*(n1j/nj)*(1-n1j/nj)*(nj-dj)/(nj-1)
    """
    t1, e1 = time[mask], event[mask].astype(bool)
    uts = np.unique(time[event.astype(bool)])
    if len(uts) < 1:
        return 0.0
    nj = (time[None, :] >= uts[:, None]).sum(axis=1).astype(float)
    n1j = (t1[None, :] >= uts[:, None]).sum(axis=1).astype(float)
    dj = ((time[None, :] == uts[:, None]) & event.astype(bool)[None, :]).sum(axis=1).astype(float)
    d1j = ((t1[None, :] == uts[:, None]) & e1[None, :]).sum(axis=1).astype(float)
    denom = np.maximum(nj, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        exp_j = n1j * dj / denom
        var_j = (n1j / denom) * (1.0 - n1j / denom) * (dj * (nj - dj) / np.maximum(nj - 1.0, 1.0))
    o_minus_e = (d1j - exp_j).sum()
    v = np.nan_to_num(var_j).sum()
    if v <= 1e-12:
        return 0.0
    return float((o_minus_e ** 2) / v)


class SurvivalTreeRegressor:
    """Árbol de supervivencia con splits por test log-rank y KM por hoja."""

    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 5,
                 n_features: int | None = None, n_thresholds: int = 12):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.n_thresholds = n_thresholds

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray,
            rng: np.random.Generator) -> "SurvivalTreeRegressor":
        n_samples, n_feat = X.shape
        self.n_features = n_feat if self.n_features is None else self.n_features
        feat_idx = rng.choice(n_feat, min(self.n_features, n_feat), replace=False)
        self.tree_ = self._grow(X, time, event, feat_idx, depth=0)
        self.feat_idx_ = feat_idx
        return self

    def _grow(self, X: np.ndarray, time: np.ndarray, event: np.ndarray,
              feat_idx: np.ndarray, depth: int) -> dict:
        node = {"n": len(time), "events": int(event.sum()),
                "time": time, "event_vec": event, "leaf": False}
        if (depth >= self.max_depth or len(time) < 2 * self.min_samples_leaf
                or event.sum() == 0):
            node["leaf"] = True
            return node
        best_score = -np.inf
        best_split = None
        for f in feat_idx:
            vals_f = X[:, f]
            qts = np.unique(np.quantile(vals_f, np.linspace(0.1, 0.9, self.n_thresholds)))
            for thr in qts:
                left = vals_f <= thr
                nl, nr = int(left.sum()), int((~left).sum())
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                score = _logrank_stat(time, event, left)
                if score > best_score:
                    best_score = score
                    best_split = (f, float(thr))
        if best_split is None or best_score <= 1e-9:
            node["leaf"] = True
            return node
        f, thr = best_split
        left_mask = X[:, f] <= thr
        node["feat"] = f
        node["thr"] = thr
        node["left"] = self._grow(X[left_mask], time[left_mask], event[left_mask], feat_idx, depth + 1)
        node["right"] = self._grow(X[~left_mask], time[~left_mask], event[~left_mask], feat_idx, depth + 1)
        return node

    def predict_survival(self, X: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """S(t) por hoja vía Kaplan-Meier. Shape (n_samples, len(t_grid))."""
        out = np.zeros((X.shape[0], len(t_grid)))
        for i, x in enumerate(X):
            leaf = self._traverse(x, self.tree_)
            out[i] = _kaplan_meier(leaf["time"], leaf["event_vec"], t_grid)
        return out

    def _traverse(self, x: np.ndarray, node: dict) -> dict:
        while not node.get("leaf"):
            node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
        return node


class SurvivalForest:
    def __init__(self, n_estimators: int = 200, max_depth: int = 8,
                 min_samples_leaf: int = 8, max_features_frac: float = 0.5,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features_frac = max_features_frac
        self.random_state = random_state

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> "SurvivalForest":
        rng = np.random.default_rng(self.random_state)
        n_feat = max(2, int(self.max_features_frac * X.shape[1]))
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(len(X), len(X), replace=True)
            t = SurvivalTreeRegressor(max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf,
                                      n_features=n_feat)
            t.fit(X[idx], time[idx], event[idx], rng)
            self.trees_.append(t)
        return self

    def predict_survival(self, X: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        all_surv = np.zeros((X.shape[0], len(t_grid)))
        for t in self.trees_:
            all_surv += t.predict_survival(X, t_grid)
        return all_surv / len(self.trees_)


# ============== LIGHTGBM CUANTÍLICO + CONFORMAL CQR ==============

def train_quantile_lgbm(X_train, y_train, X_val, y_val, quantile: float):
    import lightgbm as lgb
    params = {
        "objective": "quantile",
        "alpha": quantile,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 10,
        "lambda_l2": 1.0,
        "seed": 42,
        "verbose": -1,
    }
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    return lgb.train(params, train_set, num_boost_round=1200,
                     valid_sets=[val_set],
                     callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)])


def conformal_calibrate_cqr(predict_fn, X_cal, y_cal, coverage: float = 0.9) -> Dict:
    """Conformalized Quantile Regression (Romano, Patterson, Candès 2019).

    Score de no-cobertura: E_i = max(q05(x_i) - y_i, y_i - q95(x_i)).
    Se ajusta un ÚNICO delta = cuantil de E al nivel ceil((n+1)(1-alpha))/n
    (corrección de muestra finita). La banda [q05 - delta, q95 + delta]
    hereda la garantía de cobertura marginal >= 1 - alpha bajo intercambiabilidad.
    """
    preds = predict_fn(X_cal)
    y = np.asarray(y_cal)
    scores = np.maximum(preds["q05"] - y, y - preds["q95"])
    n = len(scores)
    alpha = 1 - coverage
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    delta = float(np.quantile(scores, level))
    covered = ((y >= preds["q05"] - delta) & (y <= preds["q95"] + delta)).mean()
    return {
        "delta": delta,
        "coverage_target": coverage,
        "coverage_calibration": float(covered),
        "n_calibration": n,
    }


def main():
    import lightgbm as lgb

    np.random.seed(42)  # reproduce el ruido del target HI y los splits

    data_dir = Path("data/processed")
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    dashboard_dir = Path("dashboard/data")
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Cargando datos...")
    readings = pd.read_csv(data_dir / "dga_readings.csv")
    fleet = pd.read_csv(data_dir / "fleet_specs.csv")
    failure = pd.read_csv(data_dir / "failure_outcomes.csv")
    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    study_end_year = meta["study_end_year"]

    print("[2/6] Construyendo features por unidad...")
    X, y_hi, df_surv, unit_ids = build_unit_features(readings, fleet, failure, study_end_year)
    feature_columns = list(X.columns)
    n_events = int(df_surv["event"].sum())
    print(f"    {X.shape[0]} unidades x {X.shape[1]} features · "
          f"{n_events} fallas observadas · {X.shape[0] - n_events} censuradas")

    # Split 70/15/15 train/val/calib
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    cal_idx = idx[n_train + n_val:]
    X_train, X_val, X_cal = X.iloc[train_idx], X.iloc[val_idx], X.iloc[cal_idx]
    y_train, y_val, y_cal = y_hi.iloc[train_idx], y_hi.iloc[val_idx], y_hi.iloc[cal_idx]

    print("[3/6] Entrenando LightGBM cuantílico (q05, q50, q95)...")
    boosters = {}
    for q in (0.05, 0.50, 0.95):
        boosters[q] = train_quantile_lgbm(X_train, y_train, X_val, y_val, quantile=q)
        print(f"    q={q} - best_iter={boosters[q].best_iteration}")

    def predict_quantiles(X_in: pd.DataFrame) -> Dict[str, np.ndarray]:
        out = {}
        for q in boosters:
            key = f"q{int(q*100):02d}"
            out[key] = boosters[q].predict(X_in, num_iteration=boosters[q].best_iteration)
        return out

    print("[4/6] Calibración conformal CQR (cobertura 90%)...")
    cal = conformal_calibrate_cqr(predict_quantiles, X_cal, y_cal, coverage=0.9)
    print(f"    delta={cal['delta']:.2f} · cobertura en calibración: "
          f"{cal['coverage_calibration']*100:.0f}% (n={cal['n_calibration']})")

    def predict_calibrated(X_in: pd.DataFrame) -> Dict[str, np.ndarray]:
        raw = predict_quantiles(X_in)
        q05 = np.clip(raw["q05"] - cal["delta"], 0, 100)
        q50 = np.clip(raw["q50"], 0, 100)
        q95 = np.clip(raw["q95"] + cal["delta"], 0, 100)
        # Repara el cruce de cuantiles (quantile crossing) imponiendo
        # monotonía q05 <= q50 <= q95 tras el ajuste conformal.
        raw["q05"] = np.minimum(q05, q50)
        raw["q50"] = q50
        raw["q95"] = np.maximum(q95, q50)
        return raw

    # Cobertura empírica en validación (fuera de calibración)
    preds_val = predict_calibrated(X_val)
    cov_val = float(((y_val.values >= preds_val["q05"]) &
                     (y_val.values <= preds_val["q95"])).mean())
    cal["coverage_validation"] = cov_val
    print(f"    cobertura en validación: {cov_val*100:.0f}%")

    # SHAP feature importance (necesario ANTES de supervivencia para elegir
    # el conjunto compacto de features del AFT/RSF)
    import shap
    explainer = shap.TreeExplainer(boosters[0.50])
    sample_idx = rng.choice(len(X), min(200, len(X)), replace=False)
    shap_values = explainer.shap_values(X.iloc[sample_idx])
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = sorted(
        [{"feature": f, "importance": float(v)} for f, v in zip(feature_columns, mean_abs_shap)],
        key=lambda x: -x["importance"]
    )

    print("[5/6] Entrenando modelos de supervivencia (Weibull AFT + RSF)...")

    # Features de supervivencia:
    # 1) SIN fuga de etiqueta: para unidades censuradas time == age_years,
    #    así que `age_years`/`install_year` predecirían el tiempo de censura
    #    de forma trivial y degenerarían AFT/RSF. La edad entra por el
    #    acondicionamiento S(a + r)/S(a).
    # 2) Conjunto COMPACTO (top-8 SHAP + placa): con ~17 eventos en train,
    #    36 features permiten memorizar los tiempos (sigma -> 0).
    SURV_EXCLUDE = {"age_years", "install_year"}
    top_shap = [d["feature"] for d in shap_importance
                if d["feature"] not in SURV_EXCLUDE][:8]
    surv_features = list(dict.fromkeys(top_shap + ["rated_mva", "voltage_kv"]))
    Xs_train = X_train[surv_features]
    surv_train = df_surv.iloc[train_idx].reset_index(drop=True)
    print(f"    Features de supervivencia ({len(surv_features)}): {surv_features}")

    weibull = fit_weibull_aft(Xs_train.values, surv_train["time"].values,
                              surv_train["event"].values, penalizer=1.0)
    weibull["feature_names"] = ["intercept"] + surv_features
    print(f"    Weibull AFT sigma={weibull['sigma']:.3f} · "
          f"intercept={weibull['intercept']:.3f} · convergió={weibull['success']}")

    forest = SurvivalForest(n_estimators=200, max_depth=8, min_samples_leaf=8,
                            max_features_frac=0.5, random_state=42)
    forest.fit(Xs_train.values, surv_train["time"].values, surv_train["event"].values)
    print(f"    SurvivalForest - {forest.n_estimators} árboles (log-rank splits)")

    print("[6/6] Exportando fleet_data.json...")
    # Grilla de vida total (desde instalación) y de vida remanente (desde hoy)
    t_grid_total = np.arange(0.0, 80.5, 0.5)   # años desde instalación
    r_grid = np.arange(0.0, 30.5, 0.5)         # años desde hoy

    payload_units = []
    for i, unit_id in enumerate(unit_ids):
        spec = fleet[fleet.unit_id == unit_id].iloc[0]
        f_out = failure[failure.unit_id == unit_id].iloc[0]
        unit_readings = readings[readings.unit_id == unit_id].sort_values("sample_date")
        if unit_readings.empty or i >= len(X):
            continue
        x_unit = X.iloc[[i]]
        xs_unit = x_unit[surv_features]
        age = float(study_end_year - spec["install_year"])

        preds = predict_calibrated(x_unit)
        hi_q05 = float(preds["q05"][0])
        hi_q50 = float(preds["q50"][0])
        hi_q95 = float(preds["q95"][0])

        # --- RUL condicional (vida remanente dado que sobrevivió a `age`) ---
        rul_weibull_raw = float(predict_weibull_rul(weibull, xs_unit.values, np.array([age]))[0])
        # El AFT extrapolía eta mas allá del soporte de datos para unidades
        # sanas: se acota el RUL exportado a 60 años y se marca el flag.
        weibull_capped = rul_weibull_raw > 60.0
        rul_weibull = min(rul_weibull_raw, 60.0)

        s_tot_rsf = forest.predict_survival(xs_unit.values, t_grid_total)[0]
        s_cond_rsf = conditionalize(s_tot_rsf, t_grid_total, age, r_grid)
        below = np.where(s_cond_rsf <= 0.5)[0]
        if len(below):
            rul_rsf = float(r_grid[below[0]])
            rul_rsf_capped = False
        else:
            rul_rsf = None
            rul_rsf_capped = True

        s_tot_wei = predict_weibull_survival(weibull, xs_unit.values, t_grid_total)[0]
        s_cond_wei = conditionalize(s_tot_wei, t_grid_total, age, r_grid)

        latest_row = unit_readings.iloc[-1]
        ppm_latest = {
            "H2": float(latest_row["H2"]),
            "CH4": float(latest_row["CH4"]),
            "C2H6": float(latest_row["C2H6"]),
            "C2H4": float(latest_row["C2H4"]),
            "C2H2": float(latest_row["C2H2"]),
        }
        duval_zone = classify_duval(ppm_latest)

        ts = []
        for _, row in unit_readings.iterrows():
            ts.append({
                "date": row["sample_date"],
                "H2": float(row["H2"]),
                "CH4": float(row["CH4"]),
                "C2H6": float(row["C2H6"]),
                "C2H4": float(row["C2H4"]),
                "C2H2": float(row["C2H2"]),
                "load_pct": float(row["load_pct"]),
                "amb_temp_c": float(row["amb_temp_c"]),
                "sc_kA": float(row["short_circuit_kA_year"]),
            })

        # 4 decimales bastan para pintar S(t): reduce el payload del dashboard
        surv_curve = [{"t": round(float(r_grid[k]), 1), "s": round(float(s_cond_rsf[k]), 4)} for k in range(len(r_grid))]
        weibull_curve = [{"t": round(float(r_grid[k]), 1), "s": round(float(s_cond_wei[k]), 4)} for k in range(len(r_grid))]

        payload_units.append({
            "unit_id": unit_id,
            "rated_mva": float(spec["rated_mva"]),
            "voltage_kv": float(spec["voltage_kv"]),
            "install_year": int(spec["install_year"]),
            "age_years": age,
            "regime": spec["regime"],
            "duval_zone": duval_zone,
            "censored": bool(f_out["censored"]),
            "failure_year": float(f_out["failure_year"]) if pd.notna(f_out["failure_year"]) else None,
            "ppm_latest": ppm_latest,
            "hi": {"q05": round(hi_q05, 1), "q50": round(hi_q50, 1), "q95": round(hi_q95, 1)},
            "rul": {
                "weibull": round(rul_weibull, 2),
                "weibull_capped": weibull_capped,
                "rsf": round(rul_rsf, 2) if rul_rsf is not None else None,
                "rsf_capped": rul_rsf_capped,
            },
            "ts": ts,
            "surv_curve": surv_curve,        # condicional, años desde hoy (RSF)
            "weibull_curve": weibull_curve,  # condicional, años desde hoy (Weibull AFT)
            "last_load_pct": float(latest_row["load_pct"]),
            "last_amb_temp_c": float(latest_row["amb_temp_c"]),
            "last_sc_kA": float(latest_row["short_circuit_kA_year"]),
        })

    # SHAP ya calculado en el paso [5/6]; aquí solo se exporta.

    exported_events = sum(1 for u in payload_units if not u["censored"])
    exported_censored = len(payload_units) - exported_events

    payload = {
        "meta": {
            "n_units": len(payload_units),
            "study_end_year": study_end_year,
            "regimes_distribution": {k: int(v) for k, v in
                pd.Series([u["regime"] for u in payload_units]).value_counts().items()},
            "duval_distribution": {k: int(v) for k, v in
                pd.Series([u["duval_zone"] for u in payload_units]).value_counts().items()},
            "feature_columns": feature_columns,
            "calibration": cal,
            "shap_importance": shap_importance,
            "training_window": f"{meta['start_year']}-{study_end_year}",
            "failures_observed": exported_events,
            "censored": exported_censored,
            "duval_zone_descr": DUVAL_ZONE_DESCR,
        },
        "units": payload_units,
    }

    out_json = dashboard_dir / "fleet_data.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"OK - {len(payload_units)} unidades exportadas a {out_json}")
    print(f"    Cohorte exportada: {exported_events} fallas · {exported_censored} censuradas")

    # Serializa modelos
    with open(out_dir / "lgbm_q05.pkl", "wb") as f:
        pickle.dump(boosters[0.05], f)
    with open(out_dir / "lgbm_q50.pkl", "wb") as f:
        pickle.dump(boosters[0.50], f)
    with open(out_dir / "lgbm_q95.pkl", "wb") as f:
        pickle.dump(boosters[0.95], f)
    with open(out_dir / "weibull_aft.pkl", "wb") as f:
        pickle.dump(weibull, f)
    with open(out_dir / "rsf.pkl", "wb") as f:
        pickle.dump(forest, f)
    with open(out_dir / "explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    print("Modelos serializados en", out_dir)


if __name__ == "__main__":
    main()
