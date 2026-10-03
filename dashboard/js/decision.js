/* decision.js — Capa de decisión económica (corregida y con riesgo residual).
 *
 * Compara el VPN esperado de tres alternativas sobre un horizonte en años,
 * usando la curva de supervivencia CONDICIONAL de la unidad (años desde
 * hoy, S(0) = 1):
 *
 *   MONITOREAR  :  Σ_{k=0}^{H-1} (S(k) − S(k+1)) · Cf/(1+r)^k
 *   INTERVENIR t:  Ci/(1+r)^t + Σ_{k<t} (S(k) − S(k+1)) · Cf/(1+r)^k
 *                      + Σ_{k≥t} (S̃(k) − S̃(k+1)) · Cf/(1+r)^k
 *   REEMPLAZAR t:  Cr/(1+r)^t + Σ_{k<t} (S(k) − S(k+1)) · Cf/(1+r)^k
 *
 * Dos correcciones clave respecto a la versión anterior:
 * 1. REEMPLAZAR también asume el riesgo de falla previo al año de acción
 *    — antes lo ignoraba y «ganaba siempre» diferiendo por puro descuento.
 * 2. INTERVENIR no elimina el riesgo: reduce el hazard en (1 − φ), con
 *    φ = RESIDUAL_HAZARD = 0.30 (una reparación temprana mitiga, no
 *    renueva). La supervivencia post-intervención es
 *        S̃(k) = S(t) + 0.7 · (S(k) − S(t))   para k ≥ t.
 *    Esto crea una frontera económica real entre reparar y reemplazar.
 *
 * Supuestos restantes (demo): reemplazo elimina el riesgo dentro del
 * horizonte; sin valor residual; costo de falla independiente del año.
 */
(function () {
  "use strict";

  const RESIDUAL_HAZARD = 0.30;

  function survivalAt(curve, t) {
    if (!curve || !curve.length) return 1;
    if (t <= curve[0].t) return 1;
    const last = curve[curve.length - 1];
    if (t >= last.t) return last.s;
    for (let i = 0; i < curve.length - 1; i++) {
      if (curve[i + 1].t >= t) {
        const f = (t - curve[i].t) / (curve[i + 1].t - curve[i].t);
        return curve[i].s * (1 - f) + curve[i + 1].s * f;
      }
    }
    return last.s;
  }

  /* Devuelve, para la unidad y parámetros dados:
   *  { action, year, npv, npvMonitor, npvIntervene, npvReplace,
   *    probFail2y, saving, curves: [{t, intervene, replace, monitor}] }
   * Todos los VPN en USD (el llamador los presenta en M). */
  function decide(unit, params, horizon) {
    const H = horizon || 12;
    const r = params.rate;
    const Cf = params.costFail * 1e6;
    const Ci = params.costInt * 1e3;
    const Cr = params.costReplace * 1e6;
    const curve = unit.surv_curve;

    const pfail = (k) => Math.max(0, survivalAt(curve, k) - survivalAt(curve, k + 1));
    const pfailPost = (k, t) => {
      const St = survivalAt(curve, t);
      const sk = St + (1 - RESIDUAL_HAZARD) * (survivalAt(curve, k) - St);
      const sk1 = St + (1 - RESIDUAL_HAZARD) * (survivalAt(curve, k + 1) - St);
      return Math.max(0, sk - sk1);
    };

    let npvMonitor = 0;
    for (let k = 0; k < H; k++) npvMonitor += (pfail(k) * Cf) / Math.pow(1 + r, k);

    const curves = [];
    let bestI = { npv: Infinity, t: 0 };
    let bestR = { npv: Infinity, t: 0 };
    for (let t = 0; t <= H; t++) {
      let pre = 0;
      for (let k = 0; k < t; k++) pre += (pfail(k) * Cf) / Math.pow(1 + r, k);

      let vi = pre + Ci / Math.pow(1 + r, t);
      for (let k = t; k < H; k++) vi += (pfailPost(k, t) * Cf) / Math.pow(1 + r, k);

      const vr = pre + Cr / Math.pow(1 + r, t);

      curves.push({ t, intervene: vi, replace: vr, monitor: npvMonitor });
      if (vi < bestI.npv) bestI = { npv: vi, t };
      if (vr < bestR.npv) bestR = { npv: vr, t };
    }

    const actions = [
      { name: "MONITOREAR", npv: npvMonitor, t: 0 },
      { name: "INTERVENIR", npv: bestI.npv, t: bestI.t },
      { name: "REEMPLAZAR", npv: bestR.npv, t: bestR.t },
    ];
    actions.sort((a, b) => a.npv - b.npv);
    const best = actions[0];

    const probFail2y = Math.max(0, 1 - survivalAt(curve, 2));

    return {
      action: best.name,
      year: best.t,
      npv: best.npv,
      npvMonitor,
      npvIntervene: bestI.npv,
      npvReplace: bestR.npv,
      interveneYear: bestI.t,
      replaceYear: bestR.t,
      probFail2y,
      saving: Math.max(0, npvMonitor - best.npv), // ahorro vs. no actuar
      horizon: H,
      residualHazard: RESIDUAL_HAZARD,
      curves,
    };
  }

  /* CAPEX diferido a valor presente para una unidad cuya decisión difiere
   * el reemplazo t* años:  Cr · (1 − (1+r)^−t*).  Es el capital que la
   * empresa no desembolsa hoy con evidencia (ingreso regulado gestionado
   * en el marco VAD/VATT). */
  function capexDeferredPV(unit, params, decision, horizon) {
    const d = decision || decide(unit, params, horizon);
    if (d.action === "REEMPLAZAR") return 0;
    const Cr = params.costReplace * 1e6;
    const tStar = d.action === "INTERVENIR" ? d.interveneYear : d.horizon;
    return Cr * (1 - Math.pow(1 + params.rate, -tStar));
  }

  window.TRDecision = { decide, capexDeferredPV, survivalAt, RESIDUAL_HAZARD };
})();
