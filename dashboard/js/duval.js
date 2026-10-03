/* duval.js — Triángulo de Duval 1 con coordenadas canónicas.
 *
 * Zonas según IEC 60599:2015 / IEEE C57.104 (mismos polígonos que usa
 * scripts/pipeline.py — fuente única de verdad duplicada a propósito:
 * si cambian, cambian en ambos).
 *
 * Convención bariéntrica: (ch4, c2h2, c2h4) en %, suma 100.
 * Orientación de dibujo: CH4 vértice superior, C2H2 inferior-izquierda,
 * C2H4 inferior-derecha (orientación clásica de las figuras Duval).
 */
(function () {
  "use strict";

  const ZONES = {
    PD: {
      name: "PD",
      descr: "Descargas parciales (corona). Baja energía en cavidades gaseosas; suele ser el primer signo de envejecimiento del aislamiento.",
      poly: [[100, 0, 0], [98, 0, 2], [98, 2, 0]],
      color: "#7a9e2e",
    },
    D1: {
      name: "D1",
      descr: "Descarga eléctrica de baja energía. Chispazo en conexiones deficientes o potenciales flotantes.",
      poly: [[0, 100, 0], [0, 77, 23], [64, 13, 23], [87, 13, 0]],
      color: "#4a7bd0",
    },
    D2: {
      name: "D2",
      descr: "Descarga eléctrica de alta energía (arco). Produce gran volumen de gas y degradación rápida del aislamiento.",
      poly: [[0, 77, 23], [0, 29, 71], [31, 29, 40], [47, 13, 40], [64, 13, 23]],
      color: "#2e5eaa",
    },
    DT: {
      name: "DT",
      descr: "Falla mixta térmica + eléctrica. Coexisten descargas y sobrecalentamiento.",
      poly: [[0, 29, 71], [0, 15, 85], [35, 15, 50], [46, 4, 50], [96, 4, 0], [87, 13, 0], [47, 13, 40], [31, 29, 40]],
      color: "#c2571c",
    },
    T1: {
      name: "T1",
      descr: "Falla térmica < 300 °C. Sobrecarga prolongada o refrigeración inadecuada.",
      poly: [[96, 4, 0], [80, 0, 20], [98, 0, 2], [98, 2, 0]],
      color: "#d9a441",
    },
    T2: {
      name: "T2",
      descr: "Falla térmica 300–700 °C. Puntos calientes localizados en núcleo o devanados.",
      poly: [[46, 4, 50], [50, 0, 50], [80, 0, 20], [76, 4, 20]],
      color: "#cf7c2a",
    },
    T3: {
      name: "T3",
      descr: "Falla térmica > 700 °C. Sobrecalentamiento severo del núcleo, arco sostenido.",
      poly: [[0, 15, 85], [0, 0, 100], [50, 0, 50], [35, 15, 50]],
      color: "#b3261e",
    },
  };

  const ZONE_ORDER = ["PD", "D1", "D2", "DT", "T1", "T2", "T3"];

  function pointInPoly(x, y, poly2d) {
    let inside = false;
    for (let i = 0, n = poly2d.length; i < n; i++) {
      const [x1, y1] = poly2d[i];
      const [x2, y2] = poly2d[(i + 1) % n];
      if ((y1 > y) !== (y2 > y)) {
        const xInt = x1 + ((y - y1) * (x2 - x1)) / (y2 - y1);
        if (x < xInt) inside = !inside;
      }
    }
    return inside;
  }

  /* Clasifica una muestra {CH4, C2H4, C2H2} (ppm) → nombre de zona o null. */
  function classify(ppm) {
    const s = ppm.CH4 + ppm.C2H4 + ppm.C2H2;
    if (s < 1) return null; // sin firma apreciable
    const p = {
      ch4: (100 * ppm.CH4) / s,
      c2h2: (100 * ppm.C2H2) / s,
      c2h4: (100 * ppm.C2H4) / s,
    };
    for (const z of ZONE_ORDER) {
      const poly2d = ZONES[z].poly.map(([a, b]) => [a, b]); // (ch4, c2h2)
      if (pointInPoly(p.ch4, p.c2h2, poly2d)) return z;
    }
    return "DT"; // borde exacto
  }

  /* Proyección bariéntrica → cartesianas dentro de un triángulo equilátero
   * con vértices CH4=(cx, 0), C2H2=(0, H), C2H4=(W, H). */
  function project(p, width, height) {
    const s = p.ch4 + p.c2h2 + p.c2h4;
    const a = s > 0 ? p.ch4 / s : 0;
    const b = s > 0 ? p.c2h2 / s : 0;
    const c = s > 0 ? p.c2h4 / s : 0;
    return {
      x: c * width + b * 0 + a * (width / 2),
      y: c * height + b * height + a * 0,
    };
  }

  function polyToPoints(poly, width, height) {
    return poly
      .map(([a, b, c]) => project({ ch4: a, c2h2: b, c2h4: c }, width, height))
      .map(pt => `${pt.x.toFixed(2)},${pt.y.toFixed(2)}`)
      .join(" ");
  }

  window.TRDuval = { ZONES, ZONE_ORDER, classify, project, polyToPoints };
})();
