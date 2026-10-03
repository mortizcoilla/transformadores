/* charts.js — Visualizaciones D3 v7 · edición papel milimetrado.
 *
 * Cada función construye una figura y devuelve un objeto con destroy()
 * para limpiar timers antes de redibujar. Las transiciones usan
 * cubic-out 500 ms — la firma de movimiento de esta edición.
 */
(function () {
  "use strict";

  const INK = "#1b2a4a";
  const INK_SOFT = "#45526d";
  const INK_FAINT = "#8a8062";
  const RULE = "#d8ccae";
  const RULE_DARK = "#b9a87f";
  const ORANGE = "#c2571c";
  const GREEN = "#2f7d4f";
  const NAVY = "#2e5eaa";
  const RED = "#b3261e";

  const GAS_COLOR = {
    H2: "#2e5eaa",
    CH4: "#2f7d4f",
    C2H6: "#d9a441",
    C2H4: "#c2571c",
    C2H2: "#7c4dbd",
  };
  const GASES = ["H2", "CH4", "C2H6", "C2H4", "C2H2"];

  /* límites IEEE C57.104 estado 2 (ppm) para la referencia del plot DGA */
  const IEEE_S2 = { H2: 100, CH4: 120, C2H6: 65, C2H4: 50, C2H2: 1 };

  const EASE = d3.easeCubicOut;
  const DUR = 500;

  const fmt1 = (v) => (v == null ? "—" : v.toFixed(1));
  const fmtPct = (v) => `${(v * 100).toFixed(1)}%`;
  const fmtM = (v) => `$${v.toFixed(2)} M`;
  const fmtK = (v) => `$${Math.round(v).toLocaleString("es-CL")}`;

  function hiColor(q50) {
    if (q50 < 30) return RED;
    if (q50 < 60) return ORANGE;
    return GREEN;
  }

  /* ---------- tooltip compartido ---------- */
  function tip() {
    return d3.select("#tooltip");
  }
  function showTip(html, evt) {
    const t = tip().html(html).classed("on", true).node();
    const pad = 14;
    let x = evt.pageX + pad;
    let y = evt.pageY - pad - t.offsetHeight;
    if (x + t.offsetWidth > window.innerWidth - 8) x = evt.pageX - t.offsetWidth - pad;
    if (y < 8) y = evt.pageY + pad;
    t.style("left", `${x}px`).style("top", `${y}px`);
  }
  function hideTip() {
    tip().classed("on", false);
  }

  /* ---------- skeleton SVG ---------- */
  function mkSvg(selector, height, margin) {
    const host = d3.select(selector);
    host.selectAll("*").remove();
    const node = host.node();
    const W = Math.max(320, node.clientWidth || 600);
    const H = height;
    const svg = host
      .append("svg")
      .attr("viewBox", `0 0 ${W} ${H}`)
      .attr("preserveAspectRatio", "xMidYMid meet");
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    return { svg, g, W, H, iw: W - margin.left - margin.right, ih: H - margin.top - margin.bottom };
  }

  function axisText(sel) {
    sel.selectAll("text").attr("fill", INK_SOFT).style("font-family", "'JetBrains Mono', monospace").style("font-size", "9.5px");
    sel.selectAll("path, line").attr("stroke", INK_SOFT);
  }

  function unitTipHtml(u, extra) {
    return `<div class="tt-title">${u.unit_id} · ${u.rated_mva} MVA · ${u.age_years} a</div>` +
      `<div class="tt-row"><span>zona Duval</span><b>${u.duval_zone ?? "—"}</b></div>` +
      `<div class="tt-row"><span>HI q50</span><b>${fmt1(u.hi.q50)}</b></div>` +
      `<div class="tt-row"><span>banda 90 %</span><b>[${fmt1(u.hi.q05)}, ${fmt1(u.hi.q95)}]</b></div>` +
      `<div class="tt-row"><span>RUL RSF</span><b>${u.rul.rsf == null ? "&gt; 30 a" : `${fmt1(u.rul.rsf)} a`}</b></div>` +
      (extra || "");
  }

  /* ================================================================
   * FIG 00 — HERO: flota sobre el eje HI + haz de barrido
   * ================================================================ */
  function chartHero(selector, units, opts) {
    let destroyed = false;
    let paused = false;
    let beamUnit = null;

    const m = { top: 26, right: 30, bottom: 34, left: 40 };
    const sk = mkSvg(selector, 240, m);
    const { g, iw, ih } = sk;

    const x = d3.scaleLinear().domain([0, 100]).range([0, iw]);

    /* zonas de estado */
    const zones = [
      { x0: 0, x1: 30, fill: "rgba(179,38,30,0.07)", lbl: "crítico" },
      { x0: 30, x1: 60, fill: "rgba(194,87,28,0.07)", lbl: "degradado" },
      { x0: 60, x1: 100, fill: "rgba(47,125,79,0.07)", lbl: "estable" },
    ];
    zones.forEach((z) => {
      g.append("rect").attr("x", x(z.x0)).attr("width", x(z.x1) - x(z.x0))
        .attr("y", 0).attr("height", ih).attr("fill", z.fill);
      g.append("text").attr("x", (x(z.x0) + x(z.x1)) / 2).attr("y", 10)
        .attr("text-anchor", "middle").attr("class", "axis-title")
        .attr("fill", INK_FAINT).style("font-size", "8.5px").text(z.lbl);
    });

    /* ejes */
    g.append("g").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(10).tickSize(3)).call(axisText);
    g.append("text").attr("x", iw / 2).attr("y", ih + 30).attr("text-anchor", "middle")
      .attr("class", "axis-title").attr("fill", INK_FAINT).text("Health Index · q50");

    /* beeswarm: apila por columna de píxel */
    const pxCol = new Map();
    const rowH = 12.5;
    const layout = units
      .slice()
      .sort((a, b) => a.hi.q50 - b.hi.q50)
      .map((u) => {
        const col = Math.round(x(u.hi.q50));
        const row = pxCol.get(col) || 0;
        pxCol.set(col, row + 1);
        return { u, row };
      });

    const gw = g.append("g").attr("class", "swarm");

    const whiskers = gw.selectAll("g.unit-w").data(layout, (d) => d.u.unit_id).join("g")
      .attr("class", "unit-w")
      .attr("transform", (d) => `translate(${x(d.u.hi.q50)},${ih - 8 - d.row * rowH})`);

    whiskers.append("line")
      .attr("y1", (d) => x(d.u.hi.q05) - x(d.u.hi.q50))
      .attr("y2", (d) => x(d.u.hi.q95) - x(d.u.hi.q50))
      .attr("stroke", NAVY).attr("stroke-width", 1).attr("stroke-opacity", 0.4);

    whiskers.append("circle")
      .attr("r", 3.4)
      .attr("fill", (d) => hiColor(d.u.hi.q50))
      .attr("stroke", "#f5eedc").attr("stroke-width", 1.2)
      .style("cursor", "pointer")
      .on("mousemove", (evt, d) => {
        paused = true;
        beamUnit = d.u;
        showTip(unitTipHtml(d.u), evt);
        highlight(d.u.unit_id);
        opts.onBeam && opts.onBeam(d.u);
      })
      .on("mouseleave", () => {
        paused = false;
        hideTip();
        highlight(null);
      })
      .on("click", (evt, d) => {
        evt.stopPropagation();
        opts.onSelect && opts.onSelect(d.u.unit_id);
      });

    /* entrada animada */
    whiskers.attr("opacity", 0).transition().duration(700).ease(EASE).delay((d, i) => (i * 1100) / layout.length)
      .attr("opacity", 1);

    function highlight(id) {
      whiskers.select("circle")
        .attr("stroke", (d) => (id && d.u.unit_id === id ? INK : "#f5eedc"))
        .attr("stroke-width", (d) => (id && d.u.unit_id === id ? 2 : 1.2))
        .attr("r", (d) => (id && d.u.unit_id === id ? 5 : 3.4));
      whiskers.select("line").attr("stroke-opacity", (d) => (id && d.u.unit_id === id ? 0.95 : 0.4));
    }

    /* haz de barrido */
    const beam = g.append("g").attr("class", "beam");
    beam.append("line")
      .attr("y1", -6).attr("y2", ih)
      .attr("stroke", ORANGE).attr("stroke-width", 1.6).attr("stroke-dasharray", "5,3");
    beam.append("path")
      .attr("d", "M -5 -12 L 5 -12 L 0 -4 Z")
      .attr("fill", ORANGE);
    beam.attr("opacity", 0).transition().duration(400).attr("opacity", 1);

    let lastReported = null;
    const timer = d3.timer((elapsed) => {
      if (destroyed) return true;
      const phase = (elapsed / 16000) % 1;
      const hi = phase * 100;
      beam.attr("transform", `translate(${x(hi)},0)`);
      if (!paused) {
        let best = null;
        let bd = Infinity;
        for (const u of units) {
          const d = Math.abs(u.hi.q50 - hi);
          if (d < bd) { bd = d; best = u; }
        }
        if (best && best.unit_id !== lastReported) {
          lastReported = best.unit_id;
          beamUnit = best;
          opts.onBeam && opts.onBeam(best);
        }
      }
    }, 200);

    return {
      destroy() { destroyed = true; timer.stop(); },
      setSelected(id) { highlight(id); },
    };
  }

  /* ================================================================
   * FIG 01 — FLOTA: scatter HI vs RUL
   * ================================================================ */
  function chartFleet(selector, units, opts) {
    const m = { top: 24, right: 24, bottom: 44, left: 52 };
    const sk = mkSvg(selector, 380, m);
    const { g, iw, ih } = sk;

    const x = d3.scaleLinear().domain([0, 100]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 30]).range([ih, 0]);
    const r = d3.scaleSqrt().domain(d3.extent(units, (u) => u.rated_mva)).range([3.2, 11]);

    /* banda crítica */
    g.append("rect").attr("x", x(0)).attr("width", x(30) - x(0)).attr("y", 0).attr("height", ih)
      .attr("fill", "rgba(179,38,30,0.06)");
    g.append("text").attr("x", x(15)).attr("y", 12).attr("text-anchor", "middle")
      .style("font-family", "'JetBrains Mono', monospace").style("font-size", "8.5px")
      .attr("fill", RED).attr("opacity", 0.8).text("q50 < 30 · crítico");

    /* gridlines */
    const gy = g.append("g").selectAll("line").data(y.ticks(6)).join("line")
      .attr("class", "gridline")
      .attr("x1", 0).attr("x2", iw).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
    const gx = g.append("g").selectAll("line").data(x.ticks(10)).join("line")
      .attr("class", "gridline")
      .attr("x1", (d) => x(d)).attr("x2", (d) => x(d)).attr("y1", 0).attr("y2", ih);

    /* ejes */
    g.append("g").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(10).tickSize(3)).call(axisText);
    g.append("g").call(d3.axisLeft(y).ticks(6).tickSize(3)).call(axisText);
    g.append("text").attr("x", iw / 2).attr("y", ih + 36).attr("text-anchor", "middle")
      .attr("class", "axis-title").attr("fill", INK_FAINT).text("Health Index · q50");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -ih / 2).attr("y", -38)
      .attr("text-anchor", "middle").attr("class", "axis-title").attr("fill", INK_FAINT)
      .text("RUL condicional · años");

    /* unidades censuradas por grilla (RUL > 30): marcas en el borde */
    const capped = units.filter((u) => u.rul.rsf == null);
    const cap = g.append("g").selectAll("path").data(capped).join("path")
      .attr("d", "M -4 -5 L 4 -5 L 0 2 Z")
      .attr("transform", (d) => `translate(${x(Math.min(d.hi.q50, 99))},${y(30) - 1})`)
      .attr("fill", (d) => zoneFill(d))
      .attr("opacity", 0.5);

    const dots = g.append("g").selectAll("circle").data(units.filter((u) => u.rul.rsf != null)).join("circle")
      .attr("cx", (d) => x(d.hi.q50))
      .attr("cy", (d) => y(d.rul.rsf))
      .attr("r", (d) => r(d.rated_mva))
      .attr("fill", (d) => zoneFill(d, 0.62))
      .attr("stroke", (d) => zoneFill(d, 1))
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .on("mousemove", (evt, d) => showTip(unitTipHtml(d), evt))
      .on("mouseleave", hideTip)
      .on("click", (evt, d) => { evt.stopPropagation(); opts.onSelect && opts.onSelect(d.unit_id); });

    dots.attr("opacity", 0)
      .transition().duration(DUR).ease(EASE)
      .delay((d, i) => (i * 900) / units.length)
      .attr("opacity", (d) => (opts.isDimmed(d) ? 0.15 : 1));

    /* anillo rojo para críticas por q05 (criterio pesimista) */
    const rings = g.append("g").selectAll("circle").data(units.filter((u) => u.hi.q05 < 30 && u.rul.rsf != null)).join("circle")
      .attr("cx", (d) => x(d.hi.q50)).attr("cy", (d) => y(d.rul.rsf))
      .attr("r", (d) => r(d.rated_mva) + 3)
      .attr("fill", "none").attr("stroke", RED).attr("stroke-width", 1).attr("stroke-dasharray", "2,2")
      .attr("pointer-events", "none")
      .attr("opacity", (d) => (opts.isDimmed(d) ? 0.1 : 0.7));

    function zoneFill(u, alpha) {
      const z = window.TRDuval.ZONES[u.duval_zone];
      return z ? z.color : INK_FAINT;
    }

    return {
      refreshDim() {
        dots.transition().duration(300).attr("opacity", (d) => (opts.isDimmed(d) ? 0.15 : 1));
        rings.transition().duration(300).attr("opacity", (d) => (opts.isDimmed(d) ? 0.1 : 0.7));
      },
    };
  }

  /* ================================================================
   * FIG 02 — HEALTH INDEX de la unidad con banda calibrada
   * ================================================================ */
  function chartHI(selector, unit, allUnits) {
    const m = { top: 40, right: 30, bottom: 40, left: 46 };
    const sk = mkSvg(selector, 250, m);
    const { g, iw, ih } = sk;

    const x = d3.scaleLinear().domain([0, 100]).range([0, iw]);

    /* histograma de flota al fondo */
    const bins = d3.bin().domain([0, 100]).thresholds(25)(allUnits.map((u) => u.hi.q50));
    const hy = d3.scaleLinear().domain([0, d3.max(bins, (b) => b.length)]).range([ih, ih * 0.25]);
    g.append("g").selectAll("rect").data(bins).join("rect")
      .attr("x", (b) => x(b.x0) + 0.5).attr("width", (b) => Math.max(0, x(b.x1) - x(b.x0) - 1))
      .attr("y", (b) => hy(b.length)).attr("height", (b) => ih - hy(b.length))
      .attr("fill", INK).attr("opacity", 0.10);

    /* umbrales */
    [[30, RED, "30 · crítico"], [60, ORANGE, "60 · degradado"]].forEach(([v, c, lbl]) => {
      g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", 0).attr("y2", ih)
        .attr("stroke", c).attr("stroke-width", 1).attr("stroke-dasharray", "3,3").attr("opacity", 0.65);
      g.append("text").attr("x", x(v) + 4).attr("y", ih - 6)
        .style("font-family", "'JetBrains Mono', monospace").style("font-size", "8.5px")
        .attr("fill", c).attr("opacity", 0.8).text(lbl);
    });

    /* ejes */
    g.append("g").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(10).tickSize(3)).call(axisText);
    g.append("text").attr("x", iw / 2).attr("y", ih + 32).attr("text-anchor", "middle")
      .attr("class", "axis-title").attr("fill", INK_FAINT).text("Health Index · flota = histograma");

    /* banda de la unidad */
    const cy = ih * 0.32;
    const grp = g.append("g");

    const band = grp.append("rect")
      .attr("x", x(unit.hi.q05))
      .attr("width", Math.max(2, x(unit.hi.q95) - x(unit.hi.q05)))
      .attr("y", cy - 13).attr("height", 26)
      .attr("fill", NAVY).attr("opacity", 0.22)
      .attr("rx", 4)
      .style("cursor", "crosshair")
      .on("mousemove", (evt) => showTip(
        `<div class="tt-title">${unit.unit_id} · banda conformal 90 %</div>` +
        `<div class="tt-row"><span>q05 (pesimista)</span><b>${fmt1(unit.hi.q05)}</b></div>` +
        `<div class="tt-row"><span>q50 (mediana)</span><b>${fmt1(unit.hi.q50)}</b></div>` +
        `<div class="tt-row"><span>q95 (optimista)</span><b>${fmt1(unit.hi.q95)}</b></div>` +
        `<div class="tt-row"><span>amplitud</span><b>${fmt1(unit.hi.q95 - unit.hi.q05)}</b></div>`, evt))
      .on("mouseleave", hideTip);

    const stem = grp.append("line")
      .attr("x1", x(unit.hi.q50)).attr("x2", x(unit.hi.q50))
      .attr("y1", cy - 17).attr("y2", cy + 17)
      .attr("stroke", INK).attr("stroke-width", 2.5);

    const mk = grp.selectAll("g.q").data([
      { k: "q05", v: unit.hi.q05, dy: -1 },
      { k: "q50", v: unit.hi.q50, dy: -1 },
      { k: "q95", v: unit.hi.q95, dy: -1 },
    ]).join("g");

    mk.append("circle").attr("cx", (d) => x(d.v)).attr("cy", cy)
      .attr("r", (d) => (d.k === "q50" ? 6 : 4.2))
      .attr("fill", (d) => (d.k === "q50" ? INK : "#f5eedc"))
      .attr("stroke", INK).attr("stroke-width", 2);

    mk.append("text").attr("x", (d) => x(d.v)).attr("y", cy - 24)
      .attr("text-anchor", "middle")
      .style("font-family", "'JetBrains Mono', monospace").style("font-size", "10px").style("font-weight", "700")
      .attr("fill", INK)
      .text((d) => `${d.k} ${d.v.toFixed(1)}`);

    grp.append("text").attr("x", x(unit.hi.q50)).attr("y", cy + 32).attr("text-anchor", "middle")
      .style("font-family", "'Space Grotesk', sans-serif").style("font-size", "11px").style("font-weight", "700")
      .attr("fill", hiColor(unit.hi.q50))
      .text(`${unit.unit_id} · ${unit.rated_mva} MVA · ${unit.age_years} a · zona ${unit.duval_zone ?? "—"}`);

    /* transición de entrada */
    band.attr("width", 0).transition().duration(DUR).ease(EASE)
      .attr("width", Math.max(2, x(unit.hi.q95) - x(unit.hi.q05)));

    return {};
  }

  /* ================================================================
   * FIG 03 — TRIÁNGULO DE DUVAL con zonas canónicas
   * ================================================================ */
  function chartDuval(selector, units, selected, opts) {
    const m = { top: 22, right: 16, bottom: 30, left: 34 };
    const hostH = 430;
    const sk = mkSvg(selector, hostH, m);
    const { g, iw, ih } = sk;

    /* triángulo equilátero inscrito */
    const side = Math.min(iw, ih / 0.866);
    const W = side, H = side * 0.866;
    const ox = (iw - W) / 2, oy = 6;
    const tg = g.append("g").attr("transform", `translate(${ox},${oy})`);

    const D = window.TRDuval;

    /* zonas */
    Object.entries(D.ZONES).forEach(([key, z]) => {
      const poly = tg.append("polygon")
        .attr("points", D.polyToPoints(z.poly, W, H))
        .attr("fill", z.color).attr("fill-opacity", 0.16)
        .attr("stroke", z.color).attr("stroke-opacity", 0.55).attr("stroke-width", 1)
        .style("cursor", "help")
        .on("mousemove", (evt) => {
          const n = units.filter((u) => u.duval_zone === key).length;
          showTip(`<div class="tt-title">Zona ${key}</div><div style="max-width:240px;white-space:normal;">${z.descr}</div><div class="tt-row" style="margin-top:4px;"><span>unidades</span><b>${n}</b></div>`, evt);
          poly.attr("fill-opacity", 0.34);
        })
        .on("mouseleave", () => poly.attr("fill-opacity", 0.16));

      /* etiqueta en el centroide (con offsets manuales para zonas delgadas) */
      const off = { PD: [10, 10], D1: [0, 0], D2: [0, 0], DT: [0, 0], T1: [0, 4], T2: [0, 0], T3: [0, 0] }[key];
      const pts = z.poly.map(([a, b, c]) => D.project({ ch4: a, c2h2: b, c2h4: c }, W, H));
      const cx = d3.mean(pts, (p) => p.x) + off[0];
      const cyy = d3.mean(pts, (p) => p.y) + off[1];
      tg.append("text").attr("x", cx).attr("y", cyy + 3)
        .attr("text-anchor", "middle")
        .style("font-family", "'Space Grotesk', sans-serif").style("font-size", "11px").style("font-weight", "700")
        .attr("fill", z.color).attr("opacity", 0.95)
        .text(key)
        .style("pointer-events", "none");
    });

    /* contorno del triángulo */
    tg.append("polygon")
      .attr("points", D.polyToPoints([[100, 0, 0], [0, 100, 0], [0, 0, 100]], W, H))
      .attr("fill", "none").attr("stroke", INK).attr("stroke-width", 1.6)
      .style("pointer-events", "none");

    /* vértices */
    const vLabels = [
      { t: "CH₄", x: W / 2, y: -8, anchor: "middle" },
      { t: "C₂H₂", x: -6, y: H + 14, anchor: "start" },
      { t: "C₂H₄", x: W + 6, y: H + 14, anchor: "end" },
    ];
    vLabels.forEach((v) => tg.append("text")
      .attr("x", v.x).attr("y", v.y).attr("text-anchor", v.anchor)
      .style("font-family", "'JetBrains Mono', monospace").style("font-size", "10.5px").style("font-weight", "700")
      .attr("fill", INK).text(v.t));

    /* nube de flota */
    const cloud = tg.append("g").selectAll("circle").data(units).join("circle")
      .attr("cx", (u) => D.project({ ch4: 100 * u.ppm_latest.CH4, c2h2: 100 * u.ppm_latest.C2H2, c2h4: 100 * u.ppm_latest.C2H4 }, W, H).x)
      .attr("cy", (u) => D.project({ ch4: 100 * u.ppm_latest.CH4, c2h2: 100 * u.ppm_latest.C2H2, c2h4: 100 * u.ppm_latest.C2H4 }, W, H).y)
      .attr("r", 2.8)
      .attr("fill", (u) => (D.ZONES[u.duval_zone] ? D.ZONES[u.duval_zone].color : INK_FAINT))
      .attr("opacity", (u) => (opts.isDimmed(u) ? 0.08 : 0.5))
      .style("cursor", "pointer")
      .on("mousemove", (evt, u) => showTip(unitTipHtml(u,
        `<div class="tt-row"><span>%CH₄</span><b>${duvalPct(u, "CH4")}</b></div>` +
        `<div class="tt-row"><span>%C₂H₂</span><b>${duvalPct(u, "C2H2")}</b></div>` +
        `<div class="tt-row"><span>%C₂H₄</span><b>${duvalPct(u, "C2H4")}</b></div>`), evt))
      .on("mouseleave", hideTip)
      .on("click", (evt, u) => { evt.stopPropagation(); opts.onSelect && opts.onSelect(u.unit_id); });

    function duvalPct(u, gas) {
      const s = u.ppm_latest.CH4 + u.ppm_latest.C2H2 + u.ppm_latest.C2H4;
      return `${((100 * u.ppm_latest[gas]) / s).toFixed(1)} %`;
    }

    /* unidad seleccionada: halo pulsante */
    if (selected) {
      const p = D.project({
        ch4: 100 * selected.ppm_latest.CH4,
        c2h2: 100 * selected.ppm_latest.C2H2,
        c2h4: 100 * selected.ppm_latest.C2H4,
      }, W, H);
      const halo = tg.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", 9)
        .attr("fill", "none").attr("stroke", ORANGE).attr("stroke-width", 1.5);
      const anim = halo.append("animate")
        .attr("attributeName", "r").attr("values", "8;14;8").attr("dur", "1.8s")
        .attr("repeatCount", "indefinite");
      const anim2 = halo.append("animate")
        .attr("attributeName", "stroke-opacity").attr("values", "0.9;0.2;0.9").attr("dur", "1.8s")
        .attr("repeatCount", "indefinite");
      const dot = tg.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", 6)
        .attr("fill", ORANGE).attr("stroke", "#f5eedc").attr("stroke-width", 2)
        .style("cursor", "pointer")
        .on("mousemove", (evt) => showTip(unitTipHtml(selected,
          `<div class="tt-row"><span>%CH₄</span><b>${duvalPct(selected, "CH4")}</b></div>` +
          `<div class="tt-row"><span>%C₂H₂</span><b>${duvalPct(selected, "C2H2")}</b></div>` +
          `<div class="tt-row"><span>%C₂H₄</span><b>${duvalPct(selected, "C2H4")}</b></div>`), evt))
        .on("mouseleave", hideTip);
      dot.attr("opacity", 0).transition().duration(DUR).ease(EASE).attr("opacity", 1);
    }

    return {
      refreshDim() {
        cloud.transition().duration(300).attr("opacity", (u) => (opts.isDimmed(u) ? 0.08 : 0.5));
      },
    };
  }

  /* ================================================================
   * FIG 04 — SERIE TEMPORAL DGA (log)
   * ================================================================ */
  function chartTimeSeries(selector, unit) {
    const m = { top: 22, right: 64, bottom: 40, left: 52 };
    const sk = mkSvg(selector, 340, m);
    const { g, iw, ih } = sk;

    const ts = unit.ts.map((d) => ({ ...d, t: new Date(d.date) }));
    if (ts.length < 2) return {};

    const x = d3.scaleTime().domain(d3.extent(ts, (d) => d.t)).range([0, iw]);
    const yMax = d3.max(ts, (d) => Math.max(...GASES.map((gg) => d[gg])));
    const y = d3.scaleLog().domain([0.5, Math.max(2, yMax * 1.6)]).range([ih, 0]);

    /* gridlines horizontales */
    g.append("g").selectAll("line").data(y.ticks(5)).join("line")
      .attr("class", "gridline")
      .attr("x1", 0).attr("x2", iw).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));

    g.append("g").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6).tickSize(3).tickFormat(d3.timeFormat("%Y")))
      .call(axisText);
    g.append("g").call(d3.axisLeft(y).ticks(5, (v) => (v >= 10 ? v : v.toFixed(1))).tickSize(3)).call(axisText);
    g.append("text").attr("x", iw / 2).attr("y", ih + 34).attr("text-anchor", "middle")
      .attr("class", "axis-title").attr("fill", INK_FAINT).text("año de muestra");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -ih / 2).attr("y", -40)
      .attr("text-anchor", "middle").attr("class", "axis-title").attr("fill", INK_FAINT).text("concentración · ppm");

    /* límites IEEE estado 2 como marcas en el margen derecho, sin colisiones */
    const limitLabels = GASES
      .map((gas) => ({ gas, lim: IEEE_S2[gas] }))
      .filter((d) => d.lim >= y.domain()[0] && d.lim <= y.domain()[1])
      .map((d) => ({ ...d, ly: y(d.lim) }))
      .sort((a, b) => a.ly - b.ly);
    const MIN_GAP = 11;
    for (let i = 1; i < limitLabels.length; i++) {
      if (limitLabels[i].ly - limitLabels[i - 1].ly < MIN_GAP) {
        limitLabels[i].ly = limitLabels[i - 1].ly + MIN_GAP;
      }
    }
    limitLabels.forEach((d) => {
      g.append("line").attr("x1", iw - 6).attr("x2", iw + 6)
        .attr("y1", y(d.lim)).attr("y2", d.ly)
        .attr("stroke", GAS_COLOR[d.gas]).attr("stroke-width", 0.8).attr("stroke-opacity", 0.6);
      g.append("text").attr("x", iw + 9).attr("y", d.ly + 3)
        .style("font-family", "'JetBrains Mono', monospace").style("font-size", "7.5px")
        .attr("fill", GAS_COLOR[d.gas]).text(`${d.gas} ${d.lim}`);
    });

    /* líneas por gas + punto final */
    GASES.forEach((gas) => {
      const path = d3.line()
        .x((d) => x(d.t))
        .y((d) => y(Math.max(d[gas], 0.5)))
        .curve(d3.curveMonotoneX);
      g.append("path").datum(ts)
        .attr("fill", "none").attr("stroke", GAS_COLOR[gas]).attr("stroke-width", 1.8)
        .attr("d", path)
        .attr("opacity", 0)
        .transition().duration(700).ease(EASE).delay(120)
        .attr("opacity", 1);

      const last = ts[ts.length - 1];
      g.append("circle").attr("cx", x(last.t)).attr("cy", y(Math.max(last[gas], 0.5)))
        .attr("r", 3).attr("fill", GAS_COLOR[gas]);
    });

    /* crosshair */
    const focus = g.append("g").style("pointer-events", "none").attr("opacity", 0);
    const vline = focus.append("line").attr("y1", 0).attr("y2", ih).attr("stroke", INK).attr("stroke-width", 0.8).attr("stroke-dasharray", "3,3");
    const fdots = focus.selectAll("circle.f").data(GASES).join("circle")
      .attr("class", "f").attr("r", 3.6).attr("fill", (gg) => GAS_COLOR[gg]).attr("stroke", "#f5eedc").attr("stroke-width", 1.2);

    sk.svg.on("mousemove", (evt) => {
      const [mx] = d3.pointer(evt, g.node());
      const tInv = x.invert(Math.max(0, Math.min(iw, mx)));
      let bestD = null, bd = Infinity;
      for (const d of ts) {
        const dd = Math.abs(d.t - tInv);
        if (dd < bd) { bd = dd; bestD = d; }
      }
      if (!bestD) return;
      const px = x(bestD.t);
      focus.attr("opacity", 1).attr("transform", `translate(${px},0)`);
      vline.attr("x1", 0).attr("x2", 0);
      fdots.attr("cx", 0).attr("cy", (gg) => y(Math.max(bestD[gg], 0.5)));
      showTip(
        `<div class="tt-title">${bestD.date}</div>` +
        GASES.map((gg) => `<div class="tt-row"><span>${ggSub(gg)}</span><b>${bestD[gg].toFixed(1)} ppm</b></div>`).join("") +
        `<div class="tt-row"><span>carga</span><b>${bestD.load_pct.toFixed(0)} %</b></div>`, evt);
    });
    sk.svg.on("mouseleave", () => { focus.attr("opacity", 0); hideTip(); });

    function ggSub(gas) {
      return { H2: "H₂", CH4: "CH₄", C2H6: "C₂H₆", C2H4: "C₂H₄", C2H2: "C₂H₂" }[gas];
    }

    return {};
  }

  /* ================================================================
   * FIG 05 — SUPERVIVENCIA CONDICIONAL (RSF + Weibull)
   * ================================================================ */
  function chartSurvival(selector, unit) {
    const m = { top: 24, right: 92, bottom: 44, left: 52 };
    const sk = mkSvg(selector, 360, m);
    const { g, iw, ih } = sk;

    const x = d3.scaleLinear().domain([0, 30]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 1.02]).range([ih, 0]);

    g.append("g").selectAll("line").data(y.ticks(5)).join("line")
      .attr("class", "gridline").attr("x1", 0).attr("x2", iw)
      .attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
    g.append("g").selectAll("line").data(x.ticks(6)).join("line")
      .attr("class", "gridline").attr("y1", 0).attr("y2", ih)
      .attr("x1", (d) => x(d)).attr("x2", (d) => x(d));

    g.append("g").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6).tickSize(3)).call(axisText);
    g.append("g").call(d3.axisLeft(y).ticks(5, d3.format(".1f")).tickSize(3)).call(axisText);
    g.append("text").attr("x", iw / 2).attr("y", ih + 36).attr("text-anchor", "middle")
      .attr("class", "axis-title").attr("fill", INK_FAINT).text("años desde hoy · r");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -ih / 2).attr("y", -38)
      .attr("text-anchor", "middle").attr("class", "axis-title").attr("fill", INK_FAINT)
      .text("S(a + r) / S(a)");

    /* zona p(falla ≤ 2 años) */
    const pf2 = Math.max(0, 1 - (unit.surv_curve.find((p) => p.t >= 2) || { s: 1 }).s);
    g.append("rect").attr("x", x(0)).attr("width", x(2) - x(0)).attr("y", 0).attr("height", ih)
      .attr("fill", "rgba(194,87,28,0.09)");
    g.append("text").attr("x", x(1)).attr("y", ih - 10).attr("text-anchor", "middle")
      .style("font-family", "'JetBrains Mono', monospace").style("font-size", "8.5px")
      .attr("fill", ORANGE).text(`p ≈ ${(pf2 * 100).toFixed(1)} %`);

    /* línea S = 0.5 */
    g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(0.5)).attr("y2", y(0.5))
      .attr("stroke", INK_FAINT).attr("stroke-width", 0.8).attr("stroke-dasharray", "1,3");

    const lineRSF = d3.line().x((d) => x(d.t)).y((d) => y(d.s)).curve(d3.curveStepAfter);
    const lineWei = d3.line().x((d) => x(d.t)).y((d) => y(d.s)).curve(d3.curveMonotoneX);

    /* banda de desacuerdo entre modelos */
    const areaBetween = d3.area()
      .x((d) => x(d.t))
      .y0((d) => y(Math.min(d.s, d.w)))
      .y1((d) => y(Math.max(d.s, d.w)))
      .curve(d3.curveMonotoneX);
    const joined = unit.surv_curve.map((p, i) => ({ t: p.t, s: p.s, w: unit.weibull_curve[i].s }));
    g.append("path").datum(joined).attr("fill", INK).attr("opacity", 0.08)
      .attr("stroke", "none").attr("d", areaBetween);

    /* curvas */
    g.append("path").datum(unit.weibull_curve).attr("fill", "none")
      .attr("stroke", GREEN).attr("stroke-width", 1.8).attr("stroke-dasharray", "6,4")
      .attr("d", lineWei)
      .attr("opacity", 0).transition().duration(DUR).ease(EASE).attr("opacity", 1);

    const rsfPath = g.append("path").datum(unit.surv_curve).attr("fill", "none")
      .attr("stroke", NAVY).attr("stroke-width", 2.6)
      .attr("d", lineRSF);
    const totalLen = rsfPath.node().getTotalLength();
    rsfPath.attr("stroke-dasharray", `${totalLen} ${totalLen}`)
      .attr("stroke-dashoffset", totalLen)
      .transition().duration(900).ease(EASE)
      .attr("stroke-dashoffset", 0);

    /* medianas */
    if (unit.rul.rsf != null) {
      g.append("line").attr("x1", x(unit.rul.rsf)).attr("x2", x(unit.rul.rsf))
        .attr("y1", y(0.5)).attr("y2", ih).attr("stroke", NAVY).attr("stroke-width", 1).attr("stroke-dasharray", "3,3");
      g.append("text").attr("x", x(unit.rul.rsf) + 5).attr("y", y(0.5) - 6)
        .style("font-family", "'JetBrains Mono', monospace").style("font-size", "9px").style("font-weight", "700")
        .attr("fill", NAVY).text(`RSF ${fmt1(unit.rul.rsf)} a`);
    } else {
      g.append("text").attr("x", x(24)).attr("y", y(0.5) - 6)
        .attr("text-anchor", "end")
        .style("font-family", "'JetBrains Mono', monospace").style("font-size", "9px")
        .attr("fill", NAVY).text("RSF: mediana > 30 a");
    }
    if (unit.rul.weibull != null && unit.rul.weibull <= 30) {
      g.append("line").attr("x1", x(unit.rul.weibull)).attr("x2", x(unit.rul.weibull))
        .attr("y1", y(0.5)).attr("y2", ih).attr("stroke", GREEN).attr("stroke-width", 1).attr("stroke-dasharray", "3,3");
      g.append("text").attr("x", x(unit.rul.weibull) + 5).attr("y", y(0.5) + 14)
        .style("font-family", "'JetBrains Mono', monospace").style("font-size", "9px").style("font-weight", "700")
        .attr("fill", GREEN).text(`Weib ${fmt1(unit.rul.weibull)} a`);
    }

    /* leyenda inline derecha */
    const leg = [
      { c: NAVY, txt: "RSF", dash: null },
      { c: GREEN, txt: "Weibull AFT", dash: "6,4" },
    ];
    leg.forEach((l, i) => {
      g.append("line").attr("x1", iw + 10).attr("x2", iw + 28).attr("y1", 10 + i * 16).attr("y2", 10 + i * 16)
        .attr("stroke", l.c).attr("stroke-width", 2.4).attr("stroke-dasharray", l.dash);
      g.append("text").attr("x", iw + 32).attr("y", 13 + i * 16)
        .style("font-family", "'JetBrains Mono', monospace").style("font-size", "8.5px")
        .attr("fill", l.c).text(l.txt);
    });

    /* crosshair interactivo */
    const focus = g.append("g").style("pointer-events", "none").attr("opacity", 0);
    const vl = focus.append("line").attr("y1", 0).attr("y2", ih).attr("stroke", INK).attr("stroke-width", 0.8).attr("stroke-dasharray", "3,3");
    const f1 = focus.append("circle").attr("r", 3.6).attr("fill", NAVY).attr("stroke", "#f5eedc").attr("stroke-width", 1.2);
    const f2 = focus.append("circle").attr("r", 3.6).attr("fill", GREEN).attr("stroke", "#f5eedc").attr("stroke-width", 1.2);

    sk.svg.on("mousemove", (evt) => {
      const [mx] = d3.pointer(evt, g.node());
      const tInv = Math.max(0, Math.min(30, x.invert(mx)));
      const sR = window.TRDecision.survivalAt(unit.surv_curve, tInv);
      const sW = window.TRDecision.survivalAt(unit.weibull_curve, tInv);
      focus.attr("opacity", 1).attr("transform", `translate(${x(tInv)},0)`);
      vl.attr("x1", 0).attr("x2", 0);
      f1.attr("cx", 0).attr("cy", y(sR));
      f2.attr("cx", 0).attr("cy", y(sW));
      showTip(`<div class="tt-title">r = ${tInv.toFixed(1)} años</div>` +
        `<div class="tt-row"><span>S(r) RSF</span><b>${sR.toFixed(3)}</b></div>` +
        `<div class="tt-row"><span>S(r) Weibull</span><b>${sW.toFixed(3)}</b></div>` +
        `<div class="tt-row"><span>p(falla)</span><b>${((1 - sR) * 100).toFixed(1)} %</b></div>`, evt);
    });
    sk.svg.on("mouseleave", () => { focus.attr("opacity", 0); hideTip(); });

    return {};
  }

  /* ================================================================
   * FIG 06 — CURVA DE DECISIÓN ECONÓMICA
   * ================================================================ */
  function chartDecision(selector, unit, params) {
    const m = { top: 24, right: 24, bottom: 44, left: 62 };
    const sk = mkSvg(selector, 380, m);
    const { g, iw, ih } = sk;

    const d = window.TRDecision.decide(unit, params, 12);
    const pts = d.curves;

    const x = d3.scaleLinear().domain([0, 12]).range([0, iw]);
    const yMax = d3.max([d3.max(pts, (p) => p.intervene), d3.max(pts, (p) => p.replace), d.npvMonitor]) / 1e6;
    const y = d3.scaleLinear().domain([0, yMax * 1.12]).range([ih, 0]);

    g.append("g").selectAll("line").data(y.ticks(5)).join("line")
      .attr("class", "gridline").attr("x1", 0).attr("x2", iw)
      .attr("y1", (v) => y(v)).attr("y2", (v) => y(v));
    g.append("g").selectAll("line").data(x.ticks(6)).join("line")
      .attr("class", "gridline").attr("y1", 0).attr("y2", ih)
      .attr("x1", (v) => x(v)).attr("x2", (v) => x(v));

    g.append("g").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6).tickSize(3)).call(axisText);
    g.append("g").call(d3.axisLeft(y).ticks(5).tickFormat((v) => `$${v.toFixed(1)}M`).tickSize(3)).call(axisText);
    g.append("text").attr("x", iw / 2).attr("y", ih + 36).attr("text-anchor", "middle")
      .attr("class", "axis-title").attr("fill", INK_FAINT).text("año de decisión · t");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -ih / 2).attr("y", -48)
      .attr("text-anchor", "middle").attr("class", "axis-title").attr("fill", INK_FAINT).text("VPN esperado");

    const li = d3.line().x((p) => x(p.t)).y((p) => y(p.intervene / 1e6)).curve(d3.curveMonotoneX);
    const lr = d3.line().x((p) => x(p.t)).y((p) => y(p.replace / 1e6)).curve(d3.curveMonotoneX);

    g.append("path").datum(pts).attr("fill", "none").attr("stroke", NAVY).attr("stroke-width", 2.4).attr("d", li)
      .attr("opacity", 0).transition().duration(DUR).ease(EASE).attr("opacity", 1);
    g.append("path").datum(pts).attr("fill", "none").attr("stroke", RED).attr("stroke-width", 2)
      .attr("stroke-dasharray", "6,4").attr("d", lr)
      .attr("opacity", 0).transition().duration(DUR).ease(EASE).attr("opacity", 1);

    /* monitorear: horizontal */
    g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(d.npvMonitor / 1e6)).attr("y2", y(d.npvMonitor / 1e6))
      .attr("stroke", GREEN).attr("stroke-width", 1.8).attr("stroke-dasharray", "2,4");

    /* óptimo */
    const bestX = d.action === "MONITOREAR" ? 0 : d.year;
    const opt = g.append("path")
      .attr("d", "M 0 -7 L 7 0 L 0 7 L -7 0 Z")
      .attr("transform", `translate(${x(bestX)},${y(d.npv / 1e6)})`)
      .attr("fill", ORANGE).attr("stroke", "#f5eedc").attr("stroke-width", 1.5);
    opt.append("animate").attr("attributeName", "opacity").attr("values", "1;0.35;1").attr("dur", "1.6s").attr("repeatCount", "indefinite");
    g.append("text").attr("x", x(bestX) + 10).attr("y", y(d.npv / 1e6) - 8)
      .style("font-family", "'JetBrains Mono', monospace").style("font-size", "9px").style("font-weight", "700")
      .attr("fill", ORANGE).text(`${d.action} · año ${d.year} · ${fmtM(d.npv / 1e6)}`);

    /* leyenda */
    const leg = [
      { c: NAVY, txt: "intervenir en t", dash: null },
      { c: RED, txt: "reemplazar en t", dash: "6,4" },
      { c: GREEN, txt: "monitorear (H = 12)", dash: "2,4" },
    ];
    leg.forEach((l, i) => {
      g.append("line").attr("x1", 10).attr("x2", 28).attr("y1", 12 + i * 15).attr("y2", 12 + i * 15)
        .attr("stroke", l.c).attr("stroke-width", 2.2).attr("stroke-dasharray", l.dash);
      g.append("text").attr("x", 32).attr("y", 15 + i * 15)
        .style("font-family", "'JetBrains Mono', monospace").style("font-size", "8.5px")
        .attr("fill", l.c).text(l.txt);
    });

    /* hover por año */
    sk.svg.on("mousemove", (evt) => {
      const [mx] = d3.pointer(evt, g.node());
      const t = Math.round(Math.max(0, Math.min(12, x.invert(mx))));
      const p = pts[t];
      if (!p) return;
      showTip(`<div class="tt-title">actuar en el año ${t}</div>` +
        `<div class="tt-row"><span>VPN intervenir</span><b>${fmtM(p.intervene / 1e6)}</b></div>` +
        `<div class="tt-row"><span>VPN reemplazar</span><b>${fmtM(p.replace / 1e6)}</b></div>` +
        `<div class="tt-row"><span>VPN monitorear</span><b>${fmtM(p.monitor / 1e6)}</b></div>`, evt);
    });
    sk.svg.on("mouseleave", hideTip);

    return { decision: d };
  }

  /* ================================================================
   * FIG 07 — SHAP
   * ================================================================ */
  const RENAMES = {
    last_H2: "H₂ última", max_H2: "H₂ máx", mean_H2: "H₂ media", growth_H2: "H₂ tasa",
    last_CH4: "CH₄ última", max_CH4: "CH₄ máx", mean_CH4: "CH₄ media", growth_CH4: "CH₄ tasa",
    last_C2H6: "C₂H₆ última", max_C2H6: "C₂H₆ máx", mean_C2H6: "C₂H₆ media", growth_C2H6: "C₂H₆ tasa", pct_C2H6: "C₂H₆ % molar",
    last_C2H4: "C₂H₄ última", max_C2H4: "C₂H₄ máx", mean_C2H4: "C₂H₄ media", growth_C2H4: "C₂H₄ tasa", pct_C2H4: "C₂H₄ % molar",
    last_C2H2: "C₂H₂ última", max_C2H2: "C₂H₂ máx", mean_C2H2: "C₂H₂ media", growth_C2H2: "C₂H₂ tasa", pct_C2H2: "C₂H₂ % molar",
    pct_H2: "H₂ % molar", pct_CH4: "CH₄ % molar",
    n_samples: "nº muestras", last_load_pct: "carga última", mean_load_pct: "carga media", max_load_pct: "carga máx",
    last_amb_temp_c: "t.ambiente última", mean_amb_temp_c: "t.ambiente media",
    max_sc_year: "cortocirc. máx/año", sum_sc: "cortocirc. acumulado",
    last_days_int: "días s/intervención", rated_mva: "potencia (MVA)", voltage_kv: "tensión (kV)",
    install_year: "año instalación", age_years: "edad (años)",
  };
  const GAS_RE = /^(?:last|max|mean|growth|pct)_(H2|CH4|C2H6|C2H4|C2H2)$/;

  function chartShap(selector, importance) {
    const top = importance.slice(0, 12).map((d) => ({
      feature: d.feature,
      name: RENAMES[d.feature] || d.feature,
      importance: d.importance,
      gas: (d.feature.match(GAS_RE) || [])[1] || null,
    }));
    const m = { top: 10, right: 56, bottom: 34, left: 138 };
    const H = Math.max(260, top.length * 24 + m.top + m.bottom);
    const sk = mkSvg(selector, H, m);
    const { g, iw, ih } = sk;

    const x = d3.scaleLinear().domain([0, d3.max(top, (d) => d.importance) * 1.12]).range([0, iw]);
    const y = d3.scaleBand().domain(top.map((d) => d.name)).range([0, ih]).padding(0.28);

    g.append("g").selectAll("line").data(x.ticks(5)).join("line")
      .attr("class", "gridline").attr("y1", 0).attr("y2", ih)
      .attr("x1", (v) => x(v)).attr("x2", (v) => x(v));

    const bars = g.selectAll("rect").data(top).join("rect")
      .attr("x", 0).attr("y", (d) => y(d.name)).attr("height", y.bandwidth())
      .attr("width", 0)
      .attr("fill", (d) => (d.gas ? GAS_COLOR[d.gas] : INK_FAINT))
      .attr("opacity", 0.9)
      .style("cursor", "help")
      .on("mousemove", (evt, d) => showTip(
        `<div class="tt-title">${d.name}</div>` +
        `<div class="tt-row"><span>feature</span><b>${d.feature}</b></div>` +
        `<div class="tt-row"><span>|SHAP| medio</span><b>${d.importance.toFixed(3)}</b></div>` +
        `<div class="tt-row"><span>tipo</span><b>${d.gas ? "gas disuelto (DGA)" : "operacional / placa"}</b></div>`, evt))
      .on("mouseleave", hideTip);

    bars.transition().duration(DUR).ease(EASE).delay((d, i) => i * 40)
      .attr("width", (d) => x(d.importance));

    g.selectAll("text.val").data(top).join("text")
      .attr("class", "val")
      .attr("x", (d) => x(d.importance) + 5).attr("y", (d) => y(d.name) + y.bandwidth() / 2 + 3.5)
      .style("font-family", "'JetBrains Mono', monospace").style("font-size", "9.5px")
      .attr("fill", INK_SOFT)
      .attr("opacity", 0)
      .text((d) => d.importance.toFixed(2))
      .transition().delay(450).duration(300).attr("opacity", 1);

    const ax = g.append("g").call(d3.axisLeft(y).tickSize(0)).call(axisText);
    ax.selectAll("text").style("font-size", "10px").attr("fill", INK);
    ax.select(".domain").remove();

    g.append("g").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(5).tickSize(3)).call(axisText);
    g.append("text").attr("x", iw / 2).attr("y", ih + 30).attr("text-anchor", "middle")
      .attr("class", "axis-title").attr("fill", INK_FAINT).text("|SHAP| medio · puntos de HI");
  }

  /* ---------- export ---------- */
  window.TRCharts = {
    chartHero, chartFleet, chartHI, chartDuval, chartTimeSeries,
    chartSurvival, chartDecision, chartShap,
    hiColor, GAS_COLOR,
  };
})();
