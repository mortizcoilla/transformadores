/* app.js — Orquestador del dashboard.
 *
 * Estado: flota, unidad seleccionada (pin), filtro por zona Duval,
 * búsqueda y orden. Todo cross-highlight: seleccionar una unidad
 * actualiza §02–§05 con transición; el filtro atenúa flota/hero/duval.
 */
(function () {
  "use strict";

  const state = {
    data: null,
    units: [],
    view: [],              // unidades visibles tras filtro/búsqueda
    selectedId: null,
    beamUnitId: null,      // unidad bajo el haz (si no hay pin)
    zoneFilter: null,      // zona Duval o null
    searchTerm: "",
    sortBy: "hi50_asc",
    params: { costFail: 8.0, costInt: 250, costReplace: 1.5, rate: 0.06 },
  };

  let heroChart = null;
  let fleetChart = null;
  let duvalChart = null;

  const $ = (id) => document.getElementById(id);

  /* ---------- lógica de vista ---------- */

  function sortUnits(list) {
    const s = list.slice();
    const cmp = {
      hi50_asc: (a, b) => a.hi.q50 - b.hi.q50,
      hi50_desc: (a, b) => b.hi.q50 - a.hi.q50,
      uncertainty: (a, b) => b.hi.q95 - b.hi.q05 - (a.hi.q95 - a.hi.q05),
      rul_asc: (a, b) => (a.rul.rsf ?? 31) - (b.rul.rsf ?? 31),
      age_desc: (a, b) => b.age_years - a.age_years,
      mva_desc: (a, b) => b.rated_mva - a.rated_mva,
    }[state.sortBy] || (() => 0);
    return s.sort(cmp);
  }

  function updateView() {
    let v = state.units.slice();
    if (state.zoneFilter) v = v.filter((u) => u.duval_zone === state.zoneFilter);
    if (state.searchTerm) {
      const q = state.searchTerm.toLowerCase();
      v = v.filter((u) => u.unit_id.toLowerCase().includes(q));
    }
    state.view = sortUnits(v);
  }

  function isDimmed(u) {
    if (state.zoneFilter && u.duval_zone !== state.zoneFilter) return true;
    return false;
  }

  function selectedUnit() {
    return state.units.find((u) => u.unit_id === state.selectedId) || null;
  }

  /* ---------- KPIs y meta ---------- */

  function renderMeta() {
    const m = state.data.meta;
    $("fleet-meta").textContent =
      `${m.n_units} unidades · ventana ${m.training_window} · ${m.failures_observed} fallas · ${m.censored} censuradas · CQR 90 %`;

    const us = state.view;
    $("kpi-units").textContent = us.length;
    $("kpi-failures").textContent = m.failures_observed;
    $("kpi-censored").textContent = m.censored;
    const crit = us.filter((u) => u.hi.q05 < 30).length;
    $("kpi-critical").textContent = crit;
    const meanHI = us.length ? us.reduce((a, u) => a + u.hi.q50, 0) / us.length : 0;
    $("kpi-hi").textContent = meanHI.toFixed(1);
    const cov = m.calibration;
    $("kpi-coverage").textContent =
      `${((cov.coverage_calibration || 0) * 100).toFixed(0)} % / ${((cov.coverage_validation ?? 0) * 100).toFixed(0)} %`;
  }

  /* ---------- readout del hero ---------- */

  function readoutUnit() {
    return selectedUnit() || state.units.find((u) => u.unit_id === state.beamUnitId) || state.units[0] || null;
  }

  function renderReadout() {
    const u = readoutUnit();
    if (!u) return;
    $("ro-unit").textContent = u.unit_id;
    const z = u.duval_zone;
    $("ro-zone").textContent = z || "—";
    $("ro-zone").style.color = z && TRDuval.ZONES[z] ? TRDuval.ZONES[z].color : "";
    $("ro-hi").textContent = u.hi.q50.toFixed(1);
    $("ro-hi").style.color = TRCharts.hiColor(u.hi.q50);
    $("ro-band").textContent = `[${u.hi.q05.toFixed(0)}, ${u.hi.q95.toFixed(0)}]`;
    $("ro-rul").textContent = u.rul.rsf == null ? "> 30 a" : `${u.rul.rsf.toFixed(1)} a`;
    const s2 = TRDecision.survivalAt(u.surv_curve, 2);
    const pf = Math.max(0, 1 - s2);
    $("ro-pf").textContent = `${(pf * 100).toFixed(1)} %`;
    $("ro-pf").classList.toggle("warm", pf > 0.1);
  }

  /* ---------- chips de zona ---------- */

  function renderZoneChips() {
    const counts = new Map();
    for (const u of state.units) {
      const z = u.duval_zone || "—";
      counts.set(z, (counts.get(z) || 0) + 1);
    }
    const order = [...TRDuval.ZONE_ORDER].filter((z) => counts.has(z));
    const host = $("zone-chips");
    host.innerHTML = "";

    const mk = (zone, label, count, color) => {
      const el = document.createElement("button");
      el.className = "chip" + (state.zoneFilter === zone ? " active" : "");
      el.style.setProperty("--zone-color", color || "#b9a87f");
      el.innerHTML = `${label}<span class="n">${count}</span>`;
      el.addEventListener("click", () => {
        state.zoneFilter = state.zoneFilter === zone ? null : zone;
        refreshAfterFilter();
      });
      host.appendChild(el);
    };

    mk(null, "todas", state.units.length, "#1b2a4a");
    order.forEach((z) => mk(z, z, counts.get(z), TRDuval.ZONES[z].color));
  }

  function refreshAfterFilter() {
    updateView();
    renderZoneChips();
    renderMeta();
    fleetChart && fleetChart.refreshDim();
    duvalChart && duvalChart.refreshDim();
  }

  /* ---------- selección ---------- */

  function selectUnit(id, opts) {
    state.selectedId = id;
    renderChartsForUnit();
    heroChart && heroChart.setSelected(id);
    renderReadout();
    if (opts && opts.scroll) {
      document.getElementById("salud").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function renderChartsForUnit() {
    const u = selectedUnit();
    if (!u) return;
    TRCharts.chartHI("#chart-hi", u, state.units);
    TRCharts.chartTimeSeries("#chart-ts", u);
    TRCharts.chartSurvival("#chart-surv", u);
    duvalChart = TRCharts.chartDuval("#chart-duval", state.units, u, {
      isDimmed,
      onSelect: (id) => selectUnit(id),
    });
    renderDecision();
  }

  /* ---------- decisión ---------- */

  function renderDecision() {
    const u = selectedUnit();
    if (!u) return;
    const d = TRDecision.decide(u, state.params, 12);
    TRCharts.chartDecision("#chart-decision", u, state.params);

    const act = $("rec-action");
    act.textContent = d.action;
    act.className = "rec-action";
    act.classList.add(`act-${d.action}`);
    $("rec-year").textContent = d.action === "MONITOREAR" ? "— (sin acción)" : `año ${d.year}`;
    $("rec-npv").textContent = `$${(d.npv / 1e6).toFixed(2)} M`;
    $("rec-prob").textContent = `${(d.probFail2y * 100).toFixed(1)} %`;
    $("rec-save").textContent = `$${(d.saving / 1e6).toFixed(2)} M`;

    /* CAPEX diferido de la flota visible (PV) */
    let capex = 0;
    for (const un of state.view) {
      capex += TRDecision.capexDeferredPV(un, state.params, null, 12);
    }
    $("rec-capex").textContent = `$${(capex / 1e6).toFixed(1)} M`;

    const notes = [];
    if (d.action === "MONITOREAR") notes.push("El riesgo esperado no justifica acción: postergar con evidencia es la decisión de menor costo esperado.");
    if (d.action === "INTERVENIR") notes.push(`Intervenir en el año ${d.year} minimiza el VPN: el riesgo de falla crece más rápido que el descuento del capital.`);
    if (d.action === "REEMPLAZAR") notes.push(`Reemplazar en el año ${d.year} minimiza el VPN esperado bajo los supuestos del modelo.`);
    notes.push("Horizonte 12 a · intervenir/reemplazar asume el riesgo previo al año de acción (corregido).");
    $("rec-note").textContent = notes.join(" ");

    /* flash de confirmación */
    const card = $("rec-card");
    card.classList.remove("flash");
    void card.offsetWidth; // reinicia la animación
    card.classList.add("flash");
  }

  /* ---------- controles ---------- */

  function bindControls() {
    $("search-input").addEventListener("input", function () {
      state.searchTerm = this.value.trim();
      updateView();
      renderMeta();
    });
    $("sort-by").addEventListener("change", function () {
      state.sortBy = this.value;
      rebuildHero();
    });

    const sliders = [
      { id: "cost-fail", label: "cost-fail-label", key: "costFail", fmt: (v) => `$ ${v.toFixed(1)} M` },
      { id: "cost-int", label: "cost-int-label", key: "costInt", fmt: (v) => `$ ${Math.round(v)} k` },
      { id: "cost-rep", label: "cost-rep-label", key: "costReplace", fmt: (v) => `$ ${v.toFixed(1)} M` },
      { id: "rate", label: "rate-label", key: "rate", fmt: (v) => `${(v * 100).toFixed(1)} %`, scale: 0.01 },
    ];
    sliders.forEach((s) => {
      $(s.id).addEventListener("input", function () {
        const raw = +this.value;
        state.params[s.key] = s.scale ? raw * s.scale : raw;
        $(s.label).textContent = s.fmt(state.params[s.key]);
        renderDecision();
      });
    });

    /* teclado: ↑/↓ recorren la vista ordenada, / enfoca búsqueda */
    document.addEventListener("keydown", (evt) => {
      if (evt.target && ["INPUT", "SELECT", "TEXTAREA"].includes(evt.target.tagName)) return;
      if (evt.key === "/") {
        evt.preventDefault();
        $("search-input").focus();
        return;
      }
      if (evt.key !== "ArrowDown" && evt.key !== "ArrowUp") return;
      evt.preventDefault();
      const list = state.view.length ? state.view : state.units;
      if (!list.length) return;
      let i = list.findIndex((u) => u.unit_id === state.selectedId);
      i = i < 0 ? 0 : (i + (evt.key === "ArrowDown" ? 1 : -1) + list.length) % list.length;
      selectUnit(list[i].unit_id);
    });
  }

  /* ---------- (re)construcción ---------- */

  function rebuildHero() {
    const list = state.view.length ? state.view : state.units;
    if (heroChart) heroChart.destroy();
    heroChart = TRCharts.chartHero("#chart-hero", list, {
      isDimmed,
      onBeam(u) {
        state.beamUnitId = u.unit_id;
        if (!state.selectedId) renderReadout();
      },
      onSelect(id) { selectUnit(id, { scroll: false }); },
    });
    heroChart.setSelected(state.selectedId);
  }

  function renderAll() {
    updateView();
    renderZoneChips();
    renderMeta();
    rebuildHero();
    fleetChart = TRCharts.chartFleet("#chart-fleet", state.units, {
      isDimmed,
      onSelect: (id) => selectUnit(id),
    });
    TRCharts.chartShap("#chart-shap", state.data.meta.shap_importance);

    /* selección inicial: peor HI q50 de la vista */
    const worst = state.view.slice().sort((a, b) => a.hi.q50 - b.hi.q50)[0] || state.units[0];
    state.selectedId = worst.unit_id;
    renderChartsForUnit();
    heroChart.setSelected(state.selectedId);
    renderReadout();
  }

  function init() {
    /* esquinas inferiores de las figuras (decorativas) */
    document.querySelectorAll(".fig").forEach((f) => {
      const c = document.createElement("i");
      c.className = "corner-b";
      f.appendChild(c);
    });

    d3.json("data/fleet_data.json")
      .then((data) => {
        state.data = data;
        state.units = data.units;
        renderAll();
        bindControls();

        /* re-render en resize (debounced) */
        let tId = null;
        window.addEventListener("resize", () => {
          clearTimeout(tId);
          tId = setTimeout(() => {
            rebuildHero();
            fleetChart = TRCharts.chartFleet("#chart-fleet", state.units, {
              isDimmed, onSelect: (id) => selectUnit(id),
            });
            renderChartsForUnit();
            TRCharts.chartShap("#chart-shap", state.data.meta.shap_importance);
          }, 250);
        });
      })
      .catch((err) => {
        console.error("Error cargando fleet_data.json:", err);
        document.querySelector("main").innerHTML =
          `<div style="max-width:640px;margin:60px auto;padding:20px;border:2px solid #b3261e;">
             <h2 style="font-family:'Space Grotesk',sans-serif;margin-top:0;">No se pudo cargar <code>data/fleet_data.json</code></h2>
             <p style="font-family:'JetBrains Mono',monospace;font-size:12px;">${err.message || err}</p>
             <p>Opciones: (1) ejecuta <code>python scripts/pipeline.py</code> para regenerarlo, o
             (2) sirve el dashboard por HTTP — <code>python -m http.server 8000</code> dentro de
             <code>dashboard/</code> y abre <code>http://localhost:8000</code> —
             pues <code>fetch</code> sobre <code>file://</code> está bloqueado por CORS.</p>
           </div>`;
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
