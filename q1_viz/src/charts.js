/* Hand-rolled SVG marks per the dataviz spec: thin bars, 4px rounded data-ends,
   2px surface gaps, hairline solid grid, selective direct labels, hover tooltips,
   text in ink tokens, legends for >=2 series. No chart library. */

const NS = "http://www.w3.org/2000/svg";
const INK = { primary: "var(--text-primary)", secondary: "var(--text-secondary)", muted: "var(--text-muted)" };
const SERIES = { 1: "var(--series-1)", 2: "var(--series-2)", 3: "var(--series-3)", deemph: "var(--series-deemph)" };
const SEQ = ["var(--seq-100)", "var(--seq-250)", "var(--seq-450)", "var(--seq-600)"];

export function svgEl(tag, attrs = {}, text = null) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text != null) node.textContent = text;
  return node;
}

function tooltip() { return document.getElementById("tooltip"); }

export function bindTip(node, title, rows) {
  node.style.cursor = "default";
  node.addEventListener("pointerenter", (event) => {
    const tip = tooltip();
    const body = rows.map(([k, v]) => `<div class="t-row"><span>${k}</span><b>${v}</b></div>`).join("");
    tip.innerHTML = `<div class="t-title">${title}</div>${body}`;
    tip.hidden = false;
    moveTip(event);
  });
  node.addEventListener("pointermove", moveTip);
  node.addEventListener("pointerleave", () => { tooltip().hidden = true; });
}

function moveTip(event) {
  const tip = tooltip();
  const pad = 14;
  let x = event.clientX + pad, y = event.clientY + pad;
  const rect = tip.getBoundingClientRect();
  if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

function roundedEndBar(x, y, width, height, r, dir = "right") {
  const radius = Math.min(r, width, height / 2);
  if (dir === "right") {
    return `M${x},${y} h${Math.max(width - radius, 0)} a${radius},${radius} 0 0 1 ${radius},${radius}` +
      ` v${height - 2 * radius} a${radius},${radius} 0 0 1 ${-radius},${radius} h${-Math.max(width - radius, 0)} z`;
  }
  return `M${x + width},${y} h${-Math.max(width - radius, 0)} a${radius},${radius} 0 0 0 ${-radius},${radius}` +
    ` v${height - 2 * radius} a${radius},${radius} 0 0 0 ${radius},${radius} h${Math.max(width - radius, 0)} z`;
}

export function legend(container, entries) {
  const box = document.createElement("div");
  box.className = "legend";
  for (const { label, color, line } of entries) {
    const key = document.createElement("span");
    key.className = "key";
    const swatch = document.createElement("span");
    swatch.className = line ? "line-key" : "swatch";
    swatch.style.background = color;
    key.append(swatch, document.createTextNode(label));
    box.append(key);
  }
  container.append(box);
}

function valueLabel(svg, x, y, text, anchor = "start", weight = 600) {
  svg.append(svgEl("text", {
    x, y, "font-size": 11, "font-weight": weight, fill: INK.secondary,
    "text-anchor": anchor, "dominant-baseline": "middle",
  }, text));
}

/** Horizontal single/dual-series bars. items: {label, value[, value2]} */
export function barChart(container, { items, series = 1, series2 = null, unit = "", max = null,
                                      labelWidth = 150, height = null, tipTitle = (d) => d.label }) {
  const dual = series2 !== null;
  const barH = 16, gapInner = 2, groupGap = 10;
  const rowH = dual ? barH * 2 + gapInner + groupGap : barH + groupGap;
  const heightAll = height ?? items.length * rowH + 24;
  const width = 760, plotX = labelWidth, plotW = width - labelWidth - 70;
  const peak = max ?? Math.max(...items.map((d) => Math.max(d.value, dual ? d.value2 : 0)), 1);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${heightAll}`, role: "img" });

  for (let t = 0; t <= 4; t++) {
    const gx = plotX + (plotW * t) / 4;
    svg.append(svgEl("line", { x1: gx, y1: 6, x2: gx, y2: heightAll - 18, stroke: "var(--grid)", "stroke-width": 1 }));
    valueLabel(svg, gx, heightAll - 8, String(Math.round((peak * t) / 4)), "middle", 400);
  }
  svg.append(svgEl("line", { x1: plotX, y1: 6, x2: plotX, y2: heightAll - 18, stroke: "var(--baseline)", "stroke-width": 1 }));

  items.forEach((item, index) => {
    const y = 8 + index * rowH;
    valueLabel(svg, plotX - 10, y + barH / 2, item.label, "end", 500);
    const w = Math.max((item.value / peak) * plotW, item.value > 0 ? 2 : 0);
    const rect = svgEl("path", { d: roundedEndBar(plotX, y, w, barH, 4), fill: SERIES[series] });
    bindTip(rect, tipTitle(item), [["数值", `${item.value}${unit}`], ...(item.tip || [])]);
    svg.append(rect);
    valueLabel(svg, plotX + w + 8, y + barH / 2, `${item.value}${unit}`);
    if (dual) {
      const y2 = y + barH + gapInner;
      const w2 = Math.max((item.value2 / peak) * plotW, item.value2 > 0 ? 2 : 0);
      const rect2 = svgEl("path", { d: roundedEndBar(plotX, y2, w2, barH, 4), fill: SERIES.deemph });
      bindTip(rect2, tipTitle(item), [[series2, `${item.value2}${unit}`]]);
      svg.append(rect2);
      if (item.value2 !== item.value) valueLabel(svg, plotX + w2 + 8, y2 + barH / 2, `${item.value2}${unit}`);
    }
  });
  container.append(svg);
  return svg;
}

/** Two-series horizontal bars with hue identity (slots 1-2). groups: {label, a, b} */
export function groupedBars(container, { groups, nameA, nameB, unit = "", labelWidth = 170, valueKeyA = "a", valueKeyB = "b" }) {
  return barChart(container, {
    items: groups.map((g) => ({ label: g.label, value: g[valueKeyA], value2: g[valueKeyB], tip: [[nameB, `${g[valueKeyB]}${unit}`]] })),
    series: 1, series2: nameB, unit, labelWidth,
    tipTitle: (d) => d.label,
  });
}

/** Confusion matrix as table + sequential wash (values always visible = table twin). */
export function heatTable(container, { rows, columns, matrix, rowTitle = "人工", colTitle = "自动" }) {
  const peak = Math.max(...matrix.flat(), 1);
  const wash = (value) => {
    if (value === 0) return "transparent";
    const ratio = value / peak;
    if (ratio <= 0.25) return SEQ[0];
    if (ratio <= 0.5) return SEQ[1];
    if (ratio <= 0.75) return SEQ[2];
    return SEQ[3];
  };
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.append(Object.assign(document.createElement("th"), { textContent: `${rowTitle} \\ ${colTitle}` }));
  for (const col of columns) headRow.append(Object.assign(document.createElement("th"), { textContent: col }));
  thead.append(headRow);
  table.append(thead);
  const tbody = document.createElement("tbody");
  rows.forEach((rowName, i) => {
    const tr = document.createElement("tr");
    tr.append(Object.assign(document.createElement("td"), { textContent: rowName }));
    columns.forEach((colName, j) => {
      const value = matrix[i][j];
      const td = document.createElement("td");
      td.textContent = value;
      td.style.background = wash(value);
      td.style.color = value / peak > 0.6 ? "var(--surface-1)" : "var(--text-primary)";
      td.style.cursor = "default";
      bindTip(td, `${rowName} → ${colName}`, [["样本数", value]]);
      tr.append(td);
    });
    tbody.append(tr);
  });
  table.append(tbody);
  container.append(table);
}

/** Alignment timeline: manual + MFA word intervals, waveform, confidence strip. */
export function timelineChart(container, timeline) {
  const width = 1000, height = 300;
  const duration = timeline.duration_s || Math.max(...timeline.words.map((w) => w.mfa_end_s || 0), 1);
  const plotX = 18, plotW = width - 36;
  const t2x = (t) => plotX + (t / duration) * plotW;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });

  // rows: y=28 manual, y=76 MFA, waveform 120..210, confidence 232, axis 262
  valueLabel(svg, plotX, 16, "人工边界 (check5.xlsx)", "start", 600);
  valueLabel(svg, plotX, 64, "MFA 词级对齐", "start", 600);
  valueLabel(svg, plotX, 112, "音频包络", "start", 600);
  valueLabel(svg, plotX, 224, "对齐置信度", "start", 600);

  for (const word of timeline.words) {
    if (word.manual_start_s != null && word.manual_end_s != null) {
      const x = t2x(word.manual_start_s), w = Math.max(t2x(word.manual_end_s) - x, 3);
      const rect = svgEl("path", { d: roundedEndBar(x, 28, w, 18, 4), fill: SERIES[2] });
      bindTip(rect, `人工 · ${word.word}`, [
        ["开始", `${word.manual_start_s.toFixed(3)} s`], ["结束", `${word.manual_end_s.toFixed(3)} s`]]);
      svg.append(rect);
    }
    if (word.mfa_start_s != null && word.mfa_end_s != null) {
      const x = t2x(word.mfa_start_s), w = Math.max(t2x(word.mfa_end_s) - x, 3);
      const rect = svgEl("path", { d: roundedEndBar(x, 76, w, 18, 4), fill: SERIES[1] });
      bindTip(rect, `MFA · ${word.word}`, [
        ["开始", `${word.mfa_start_s.toFixed(3)} s`], ["结束", `${word.mfa_end_s.toFixed(3)} s`],
        ["人工误差", word.human_boundary_error_s != null ? `${Math.round(word.human_boundary_error_s * 1000)} ms` : "—"],
        ["置信度", word.alignment_confidence]]);
      svg.append(rect);
      if (w > word.word.length * 6.2) {
        svg.append(svgEl("text", {
          x: x + 5, y: 85, "font-size": 10, fill: "#ffffff", "dominant-baseline": "middle",
        }, word.word));
      }
    }
    const gradeColor = { HIGH: "var(--seq-600)", MEDIUM: "var(--seq-450)", LOW: "var(--seq-250)", UNAVAILABLE: "var(--unavailable)" };
    if (word.mfa_start_s != null) {
      const x = t2x(word.mfa_start_s), w = Math.max(t2x(word.mfa_end_s) - x, 3);
      const strip = svgEl("rect", { x, y: 232, width: w, height: 10, rx: 3, fill: gradeColor[word.alignment_confidence] || "var(--unavailable)" });
      bindTip(strip, `${word.word} · ${word.alignment_confidence}`, [["MFA–ASR 分歧等级", word.alignment_confidence]]);
      svg.append(strip);
    }
  }

  const envelope = timeline.waveform?.envelope || [];
  if (envelope.length) {
    const mid = 165, amp = 42;
    svg.append(svgEl("rect", { x: plotX, y: mid - amp, width: plotW, height: amp * 2, fill: "var(--text-muted)", opacity: 0.06 }));
    const step = plotW / envelope.length;
    envelope.forEach((value, index) => {
      const x = plotX + index * step + step / 2;
      svg.append(svgEl("line", {
        x1: x, y1: mid - value * amp, x2: x, y2: mid + value * amp,
        stroke: "var(--text-muted)", "stroke-width": Math.max(step - 0.6, 0.8), opacity: 0.75,
      }));
    });
    svg.append(svgEl("line", { x1: plotX, y1: mid, x2: plotX + plotW, y2: mid, stroke: "var(--baseline)", "stroke-width": 1 }));
  }

  for (let t = 0; t <= Math.floor(duration); t++) {
    const x = t2x(t);
    svg.append(svgEl("line", { x1: x, y1: 246, x2: x, y2: 252, stroke: "var(--baseline)", "stroke-width": 1 }));
    valueLabel(svg, x, 264, `${t}s`, "middle", 400);
  }
  svg.append(svgEl("line", { x1: plotX, y1: 252, x2: plotX + plotW, y2: 252, stroke: "var(--baseline)", "stroke-width": 1 }));
  container.append(svg);
  legend(container, [
    { label: "MFA 词边界", color: SERIES[1] },
    { label: "人工词边界", color: SERIES[2] },
    { label: "置信度 LOW / MEDIUM / HIGH", color: "var(--seq-250)" },
    { label: "无对齐 (UNAVAILABLE)", color: "var(--unavailable)" },
  ]);
}

/** iBUG-68 schematic built from parametric arcs (layout 示意, counts exact:
    jaw 17 + brows 10 + nose 9 + eyes 12 + mouth 20 = 68). */
function buildFace68() {
  const points = [];
  for (let i = 0; i <= 16; i++) {
    const angle = Math.PI * (0.15 + (0.7 * i) / 16);
    points.push([0.5 - 0.36 * Math.cos(angle), 0.42 + 0.52 * Math.sin(angle)]);
  }
  for (let i = 0; i < 5; i++) points.push([0.28 + i * 0.06, 0.27 - 0.04 * Math.sin((i / 4) * Math.PI)]);
  for (let i = 0; i < 5; i++) points.push([0.52 + i * 0.06, 0.27 - 0.04 * Math.sin((i / 4) * Math.PI)]);
  for (let i = 0; i <= 3; i++) points.push([0.5, 0.36 + i * 0.05]);
  points.push([0.43, 0.55], [0.465, 0.565], [0.5, 0.57], [0.535, 0.565], [0.57, 0.55]);
  for (const centerX of [0.36, 0.64]) {
    for (let i = 0; i < 6; i++) {
      const angle = (i / 6) * 2 * Math.PI;
      points.push([centerX + 0.075 * Math.cos(angle), 0.42 + 0.035 * Math.sin(angle)]);
    }
  }
  for (let i = 0; i < 12; i++) {
    const angle = (i / 12) * 2 * Math.PI;
    points.push([0.5 + 0.115 * Math.cos(angle), 0.68 + 0.055 * Math.sin(angle)]);
  }
  for (let i = 0; i < 8; i++) {
    const angle = Math.PI + (i / 7) * Math.PI;
    points.push([0.5 + 0.115 * Math.cos(angle), 0.68 + 0.04 * Math.sin(angle)]);
  }
  return points;
}
const FACE68 = buildFace68();

export function face68(container, schemaBlocks) {
  const width = 340, height = 300;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  svg.append(svgEl("rect", { x: 0, y: 0, width, height, fill: "none" }));
  FACE68.forEach(([x, y], index) => {
    const dot = svgEl("circle", {
      cx: 40 + x * 250, cy: 30 + y * 220, r: 2.6,
      fill: SERIES[1], stroke: "var(--surface-1)", "stroke-width": 1.2,
    });
    bindTip(dot, `OpenFace 点 ${index}`, [["拓扑", "iBUG-68"], ["几何维度", "x, y (136 维)"], ["归一化", "眼中心平移 + 眼距尺度"]]);
    svg.append(dot);
  });
  container.append(svg);
  const table = document.createElement("table");
  table.innerHTML = "<thead><tr><th>特征块</th><th>帧级</th><th>词级</th></tr></thead>";
  const tbody = document.createElement("tbody");
  for (const block of schemaBlocks) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${block.feature_name}</td><td>${block.frame_dimension}</td><td>${block.word_dimension}</td>`;
    tbody.append(tr);
  }
  table.append(tbody);
  container.append(table);
}

/** Routing outcome bars with status colors + icon-label mitigation. */
export function routingBars(container, routing) {
  const icon = { good: "✓", warning: "△", critical: "✕" };
  const items = [
    { label: `${icon.good} 安全放行 · TP`, value: routing.safe_allowed_tp, color: "var(--status-good)" },
    { label: `${icon.critical} 危险放行 · FP`, value: routing.unsafe_allowed_fp, color: "var(--status-critical)" },
    { label: `${icon.warning} 保守拦截 · FN`, value: routing.safe_held_fn, color: "var(--status-warning)" },
    { label: `${icon.good} 正确拦截 · TN`, value: routing.unsafe_held_tn, color: "var(--status-good)" },
  ];
  const barH = 16, rowH = 28, width = 420, labelWidth = 150, plotW = width - labelWidth - 50;
  const heightAll = items.length * rowH + 20;
  const peak = Math.max(...items.map((d) => d.value), 1);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${heightAll}`, role: "img" });
  items.forEach((item, index) => {
    const y = 8 + index * rowH;
    valueLabel(svg, labelWidth - 10, y + barH / 2, item.label, "end", 500);
    const w = Math.max((item.value / peak) * plotW, item.value > 0 ? 2 : 0);
    const rect = svgEl("path", { d: roundedEndBar(labelWidth, y, w, barH, 4), fill: item.color });
    bindTip(rect, item.label.replace(/[✓△✕]\s*/, ""), [["样本数", item.value]]);
    svg.append(rect);
    valueLabel(svg, labelWidth + w + 8, y + barH / 2, String(item.value));
  });
  container.append(svg);
}
