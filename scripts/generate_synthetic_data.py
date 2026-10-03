"""
Generador de datos sintéticos realistas de una flota de transformadores.

Modela:
- 200 unidades con 5-15 años de historial.
- DGA (H2, CH4, C2H6, C2H4, C2H2) muestreado 2-4 veces/año.
- Covariables operacionales: carga (% rated), temperatura ambiente (°C),
  eventos de cortocircuito (kA acumulados), edad del aislamiento, días desde
  última intervención.
- Ocho regímenes latentes (NORMAL + 7 tipos de falla Duval: PD, T1-T3,
  D1-D2, DT) con firma gaseosa coherente con IEEE C57.104 / IEC 60599.
- Censura derecha: unidades sin falla al cierre del estudio.
- Nota: las unidades cuya falla latente ocurre ANTES del inicio de la
  ventana de observación no generan lecturas (quedan fuera de la cohorte
  con DGA; el pipeline las excluye y reporta la cohorte efectiva).

Los datos se ABSTRAEN de distribuciones publicadas (IEEE C57.104, IEC
60599, CIGRE TB 771). NO son reales pero conservan la física del problema.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# Estados Duval canónicos (porcentajes típicos sobre suma DGA)
DUVAL_REGIMES = {
    "NORMAL": {"H2": 0.45, "CH4": 0.30, "C2H6": 0.15, "C2H4": 0.09, "C2H2": 0.01},
    "PD":     {"H2": 0.62, "CH4": 0.20, "C2H6": 0.10, "C2H4": 0.07, "C2H2": 0.01},
    "T1":     {"H2": 0.15, "CH4": 0.50, "C2H6": 0.20, "C2H4": 0.14, "C2H2": 0.01},
    "T2":     {"H2": 0.10, "CH4": 0.40, "C2H6": 0.15, "C2H4": 0.34, "C2H2": 0.01},
    "T3":     {"H2": 0.08, "CH4": 0.30, "C2H6": 0.10, "C2H4": 0.51, "C2H2": 0.01},
    "D1":     {"H2": 0.50, "CH4": 0.20, "C2H6": 0.05, "C2H4": 0.20, "C2H2": 0.05},
    "D2":     {"H2": 0.42, "CH4": 0.18, "C2H6": 0.04, "C2H4": 0.28, "C2H2": 0.08},
    "DT":     {"H2": 0.40, "CH4": 0.22, "C2H6": 0.06, "C2H4": 0.20, "C2H2": 0.12},
}

# Tasas base (ppm/año) en condición NORMAL
BASE_RATES_PPM_YR = {"H2": 12, "CH4": 8, "C2H6": 5, "C2H4": 6, "C2H2": 0.5}


@dataclass
class UnitSpec:
    unit_id: str
    rated_mva: float
    voltage_kv: float
    install_year: int
    regime: str        # régimen latente (NORMAL, T1, D2, etc.)
    base_failure_year: float  # año calendario en que ocurriría la falla sin intervención
    study_end_year: int


@dataclass
class DGAReading:
    unit_id: str
    sample_date: str  # ISO
    years_since_install: float
    H2: float
    CH4: float
    C2H6: float
    C2H4: float
    C2H2: float
    load_pct: float
    amb_temp_c: float
    short_circuit_kA_year: float  # kA acumulado en el año previo
    days_since_last_intervention: int


def _seed_all(seed: int) -> None:
    np.random.seed(seed)


def _assign_regime() -> str:
    """Distribución aproximada de una población con mix realista.

    NORMAL 39%, PD 10%, T1 13%, T2 10%, T3 6%, D1 8%, D2 8%, DT 6%.
    """
    p = [0.39, 0.10, 0.13, 0.10, 0.06, 0.08, 0.08, 0.06]
    keys = ["NORMAL", "PD", "T1", "T2", "T3", "D1", "D2", "DT"]
    return str(np.random.choice(keys, p=p))


def _make_fleet(n_units: int, study_end_year: int) -> List[UnitSpec]:
    units: List[UnitSpec] = []
    for i in range(n_units):
        rated = float(np.random.choice([50, 75, 100, 150, 200, 250, 400]))
        voltage = float(np.random.choice([110, 154, 220, 500]))
        # Edad al inicio del estudio: 1-30 años
        age_start = int(np.random.randint(1, 31))
        install_year = study_end_year - age_start - int(np.random.randint(5, 11))
        regime = _assign_regime()

        # Tiempo hasta falla (años calendario desde install_year):
        # - NORMAL: vida larga (30-50 años)
        # - Defecto severo (DT, T3, D2): 1-8 años
        if regime == "NORMAL":
            ttf = float(np.random.uniform(35, 55))
        elif regime in ("PD", "T1"):
            ttf = float(np.random.uniform(15, 30))
        elif regime in ("T2", "D1"):
            ttf = float(np.random.uniform(7, 18))
        else:  # T3, D2, DT
            ttf = float(np.random.uniform(1.5, 9))
        base_failure_year = install_year + ttf

        units.append(UnitSpec(
            unit_id=f"TR-{i+4:03d}",
            rated_mva=rated,
            voltage_kv=voltage,
            install_year=install_year,
            regime=regime,
            base_failure_year=base_failure_year,
            study_end_year=study_end_year,
        ))
    return units


def _duval_sample(regime: str, years_since_install: float,
                  load_pct: float, amb_temp_c: float) -> dict:
    """Devuelve ppm de los 5 gases respetando Duval y la termodinámica."""
    shares = DUVAL_REGIMES[regime]
    # tasa total TGA crece con carga y temperatura
    base_total = 30 + 4 * years_since_install + 0.15 * load_pct * years_since_install \
          + 0.5 * max(0.0, amb_temp_c - 20)
    if regime != "NORMAL":
        base_total *= 3.5  # activos en falla aceleran acumulación
    base_total = max(base_total, 25)
    total_dga = base_total * float(np.random.lognormal(0.0, 0.25))
    # Composición con Dirichlet perturbada
    alpha = np.array([shares[g] for g in ("H2", "CH4", "C2H6", "C2H4", "C2H2")]) * 25
    comp = np.random.dirichlet(alpha + 1e-3)
    ppm = {g: float(total_dga * comp[i]) for i, g in enumerate(("H2", "CH4", "C2H6", "C2H4", "C2H2"))}
    # ruido de medición ±5%
    for g in ppm:
        ppm[g] *= float(np.random.uniform(0.92, 1.08))
        ppm[g] = max(ppm[g], 0.05)
    return ppm


def _load_profile(years_since_install: float, regime: str) -> float:
    base = 55 + 8 * np.sin(years_since_install * 0.7) + np.random.normal(0, 4)
    if regime != "NORMAL":
        base += 6  # unidades con defecto se cargan más (perfil histórico)
    return float(np.clip(base, 25, 95))


def _amb_temp(year_calendar: float) -> float:
    # Chile central: media anual ~17°C, amplitud estacional ~10°C
    return float(17 + 10 * np.sin((year_calendar - 0.85) * 2 * np.pi) + np.random.normal(0, 1.5))


def _short_circuit(regime: str, year_calendar: float) -> float:
    # Eventos/año en kA acumulados. Más eventos en unidades con defecto.
    base = float(np.random.exponential(2.0))
    if regime in ("D2", "DT"):
        base *= 2.5
    elif regime in ("D1", "T3"):
        base *= 1.6
    return float(np.clip(base, 0.0, 25.0))


def _generate_readings_for_unit(spec: UnitSpec,
                                start_year: int,
                                end_year: int) -> Tuple[List[DGAReading], bool]:
    """Genera lecturas DGA para una unidad entre start_year y end_year.

    Devuelve (lecturas, ocurrió_falla_dentro_ventana).
    """
    readings: List[DGAReading] = []
    failed = False
    # Frecuencia de muestreo 2-4/año
    samples_per_year = int(np.random.choice([2, 3, 4]))
    last_intervention_year = spec.install_year + np.random.uniform(0, 3)

    for y in np.arange(start_year, end_year + 1e-6, 1.0 / samples_per_year):
        if y >= spec.base_failure_year:
            failed = True
            break
        years_since = y - spec.install_year
        load = _load_profile(years_since, spec.regime)
        amb = _amb_temp(y)
        sc = _short_circuit(spec.regime, y)
        ppm = _duval_sample(spec.regime, years_since, load, amb)

        # días desde última intervención
        days_int = int((y - last_intervention_year) * 365 + np.random.randint(-30, 30))
        days_int = max(days_int, 30)

        readings.append(DGAReading(
            unit_id=spec.unit_id,
            sample_date=f"{int(y):04d}-{int(((y - int(y)) * 12) + 1):02d}-15",
            years_since_install=float(years_since),
            H2=round(ppm["H2"], 2),
            CH4=round(ppm["CH4"], 2),
            C2H6=round(ppm["C2H6"], 2),
            C2H4=round(ppm["C2H4"], 2),
            C2H2=round(ppm["C2H2"], 2),
            load_pct=round(load, 1),
            amb_temp_c=round(amb, 1),
            short_circuit_kA_year=round(sc, 2),
            days_since_last_intervention=days_int,
        ))
    return readings, failed


def generate(study_end_year: int = 2024, n_units: int = 200, seed: int = 42) -> dict:
    _seed_all(seed)
    fleet_specs = _make_fleet(n_units, study_end_year)
    all_readings: List[DGAReading] = []
    failure_records = []
    start_year = study_end_year - 6  # ventana histórica de 6 años

    for spec in fleet_specs:
        readings, failed = _generate_readings_for_unit(spec, start_year, study_end_year)
        all_readings.extend(readings)
        if failed:
            # registra el evento de falla como outcome para análisis de RUL
            failure_records.append({
                "unit_id": spec.unit_id,
                "failure_year": float(spec.base_failure_year),
                "regime": spec.regime,
                "censored": False,
                "time_to_failure_years": float(spec.base_failure_year - spec.install_year),
            })
        else:
            failure_records.append({
                "unit_id": spec.unit_id,
                "failure_year": None,
                "regime": spec.regime,
                "censored": True,
                "time_to_failure_years": float(study_end_year - spec.install_year),
            })

    df_readings = pd.DataFrame([asdict(r) for r in all_readings])
    df_failure = pd.DataFrame(failure_records)
    df_fleet = pd.DataFrame([asdict(s) for s in fleet_specs])

    return {
        "readings": df_readings,
        "failure": df_failure,
        "fleet": df_fleet,
        "meta": {
            "n_units": n_units,
            "study_end_year": study_end_year,
            "start_year": start_year,
            "seed": seed,
            "regimes_distribution": {k: int(v) for k, v in
                pd.Series([s.regime for s in fleet_specs]).value_counts().items()},
            "n_failures_observed": int((~df_failure["censored"]).sum()),
            "n_censored": int(df_failure["censored"].sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera dataset sintético de flota de transformadores")
    parser.add_argument("--out", default="data/processed", help="Directorio de salida")
    parser.add_argument("--n-units", type=int, default=200)
    parser.add_argument("--study-end-year", type=int, default=2024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = generate(args.study_end_year, args.n_units, args.seed)
    bundle["readings"].to_csv(out_dir / "dga_readings.csv", index=False)
    bundle["failure"].to_csv(out_dir / "failure_outcomes.csv", index=False)
    bundle["fleet"].to_csv(out_dir / "fleet_specs.csv", index=False)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(bundle["meta"], f, indent=2, ensure_ascii=False)

    print(f"OK - {len(bundle['readings'])} lecturas DGA, {bundle['meta']['n_failures_observed']} fallas, {bundle['meta']['n_censored']} censurados")
    print(f"Guardado en {out_dir}")


if __name__ == "__main__":
    main()