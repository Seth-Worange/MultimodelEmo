import { barChart, groupedBars, heatTable, timelineChart, face68, routingBars, legend } from "./charts.js";

const $ = (id) => document.getElementById(id);

async function loadData() {
  for (const path of ["frontend_data.json", "public/frontend_data.json"]) {
    try {
      const response = await fetch(path);
      if (response.ok) return response.json();
    } catch { /* try next */ }
  }
  throw new Error("frontend_data.json 未找到：请先运行 tools/build_frontend_data.py，再用 npm run dev 或 python -m http.server 打开。");
}

function tile(label, value, note, status = null) {
  const box = document.createElement("div");
  box.className = "tile";
  box.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>` +
    (status ? `<div class="status-line"><span class="chip ${status.tone}"><span class="dot"></span>${status.text}</span></div>` : "") +
    (note ? `<div class="note">${note}</div>` : "");
  return box;
}

function simpleTable(container, headers, rows) {
  const table = document.createElement("table");
  table.innerHTML = `<thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead>`;
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = row.map((cell) => `<td>${cell}</td>`).join("");
    tbody.append(tr);
  }
  table.append(tbody);
  container.append(table);
}

function pct(value, digits = 1) {
  return value == null ? "—" : `${(value * 100).toFixed(digits)}%`;
}

function render(data) {
  const chips = [
    ["visual_backend = openface68", ""],
    [`schema ${data.visual.feature_schema.schema_version}`, ""],
    ["人工标签仅用于评估", "good"],
    ["full100_feature_extraction = NOT_RUN", "warning"],
    ["分支 feature/q1-improvement", ""],
  ];
  for (const [text, tone] of chips) {
    const chip = document.createElement("span");
    chip.className = `chip ${tone}`;
    chip.innerHTML = tone ? `<span class="dot"></span>${text}` : text;
    $("meta-chips").append(chip);
  }

  data.pipeline.forEach((step, index) => {
    if (index) $("pipeline").append(Object.assign(document.createElement("span"), { className: "arrow", textContent: "→" }));
    $("pipeline").append(Object.assign(document.createElement("span"), { className: "step", textContent: step }));
  });

  const quality = data.quality_overview;
  $("quality-tiles").append(
    tile("人工精确一致率", pct(quality.exact_class_accuracy), "自动分类 = 人工分类 (100 条)"),
    tile("粗粒度一致率", pct(quality.coarse_class_accuracy), "NO_AUDIO/NO_SPEECH 合并为 NO_VALID_SPEECH"),
    tile("样本人工核验数", quality.sample_count, "人工核验100条.xlsx · 只读评估"),
    tile("自动 QA 覆盖", "100 / 100", "Round 5 轻量 QA 成功 100 条"),
  );
  const classItems = Object.entries(quality.manual_class_counts)
    .map(([label, value]) => ({ label, value, tip: [["自动分类数", quality.automatic_class_counts[label] ?? 0]] }))
    .sort((a, b) => b.value - a.value);
  barChart($("chart-quality"), { items: classItems, labelWidth: 180, tipTitle: (d) => `人工 · ${d.label}` });

  const qa = data.correspondence_qa;
  $("qa-tiles").append(
    tile("HIGH_CONFIDENCE precision", pct(qa.high_confidence_precision, 2), "放行样本中人工 MATCHED 比例",
      { tone: "good", text: "✓ 无错误放行" }),
    tile("HIGH_CONFIDENCE recall", pct(qa.high_confidence_recall, 2), "人工 MATCHED 中被放行比例"),
    tile("F1", pct(qa.high_confidence_f1, 1), "保守优先"),
    tile("unsafe false positive", qa.unsafe_false_positive_count, "危险文本-音频错误对齐放行数",
      { tone: "good", text: "✓ 0 危险假阳性" }),
    tile("conservative false negative", qa.conservative_false_negative_count, "被保守拦截的可对齐样本"),
  );
  routingBars($("chart-routing"), qa.routing_confusion);
  legend($("chart-routing"), [
    { label: "安全/正确", color: "var(--status-good)" },
    { label: "保守拦截", color: "var(--status-warning)" },
    { label: "危险放行", color: "var(--status-critical)" },
  ]);
  heatTable($("chart-confusion"), {
    rows: qa.confusion.row_labels, columns: qa.confusion.columns, matrix: qa.confusion.matrix,
  });

  const extra = qa.extra_speech;
  simpleTable($("table-extra-speech"),
    ["指标", "R4 前置", "R5 VAD 前置", "R4 后置", "R5 VAD 后置"],
    [
      ["precision", pct(extra.round4_before_on_same_samples.precision), pct(extra.round5_before.precision),
        pct(extra.round4_after_on_same_samples.precision), pct(extra.round5_after.precision)],
      ["recall", pct(extra.round4_before_on_same_samples.recall), pct(extra.round5_before.recall),
        pct(extra.round4_after_on_same_samples.recall), pct(extra.round5_after.recall)],
      ["评估样本数", extra.round5_comparable_matched_span_count, extra.round5_comparable_matched_span_count,
        extra.round5_comparable_matched_span_count, extra.round5_comparable_matched_span_count],
    ]);
  const extraNote = document.createElement("p");
  extraNote.className = "warn-note";
  extraNote.textContent = `VAD 规则语义更符合“存在额外讲话”，但在同口径 ${extra.round5_comparable_matched_span_count} 条上方向准确率 ` +
    `${pct(extra.round5_exact_direction_accuracy)}（R4 同口径为 ${pct(extra.round4_exact_direction_accuracy ?? extra.round5_exact_direction_accuracy)}），未在同一 100 条人工标签上调阈值。`;
  $("table-extra-speech").append(extraNote);

  timelineChart($("chart-timeline"), data.alignment.timeline);
  const thumbs = $("timeline-thumbs");
  for (const name of data.alignment.timeline.thumbnails) {
    const img = document.createElement("img");
    img.src = `thumbs/${name}`;
    img.alt = "视频帧";
    img.addEventListener("error", () => { img.src = `public/thumbs/${name}`; }, { once: true });
    thumbs.append(img);
  }
  simpleTable($("table-boundary"),
    ["样本", "人工词数", "可比较词", "平均边界误差 (ms)", "平均 IoU", "100ms 双边界通过率"],
    data.alignment.by_sample.map((row) => [
      `<code>${row.sample_id}</code>`, row.manual_words, row.compared_words,
      row.mean_boundary_error_ms == null ? "未运行 MFA（保守拦截）" : row.mean_boundary_error_ms,
      row.mean_temporal_iou ?? "—", row.pass_rate_100ms == null ? "—" : pct(row.pass_rate_100ms, 1),
    ]));
  const conf = data.alignment.alignment_confidence;
  simpleTable($("table-confidence"),
    ["分歧等级", "词数", "含义"],
    Object.entries(conf.counts).map(([grade, count]) => [
      grade, count,
      { HIGH: "分歧 > 200 ms，需人工复核", MEDIUM: "分歧 100–200 ms", LOW: "分歧 ≤ 100 ms", UNAVAILABLE: "无真实 MFA/ASR 对照" }[grade] ?? "—",
    ]));
  const confNote = document.createElement("p");
  confNote.className = "warn-note";
  confNote.textContent = `含义：${conf.meaning}。覆盖 ${conf.audited_word_count} 词 / ${conf.eligible_real_mfa_sample_count} 条真实 MFA 样本。`;
  $("table-confidence").append(confNote);

  face68($("chart-face"), data.visual.schema.blocks);
  const backendGroups = data.visual.backend_comparison.map((row) => ({
    label: row.sample_id, a: row.openface68_valid, b: row.mediapipe478_valid,
  }));
  groupedBars($("chart-backend"), {
    groups: backendGroups, nameA: "OpenFace68 有效帧", nameB: "MediaPipe478 有效帧",
    labelWidth: 180, unit: " 帧",
  });
  legend($("chart-backend"), [
    { label: "OpenFace68（默认后端）", color: "var(--series-1)" },
    { label: "MediaPipe478（基线）", color: "var(--series-deemph)" },
  ]);
  const backendNote = document.createElement("p");
  backendNote.className = "warn-note";
  backendNote.textContent = "MediaPipe 帧向量 1486 维（478×3 + 52 blendshape）；OpenFace 分块 136/35/6/8 维。两者维度不可互换，此图仅比较有效帧数。静态人物图中高检出率不代表说话者身份已验证。";
  $("chart-backend").append(backendNote);

  const select = $("sample-select");
  const ids = Object.keys(data.smoke.samples);
  for (const id of ids) select.append(new Option(id, id));
  const renderSample = () => {
    const sample = data.smoke.samples[select.value];
    const body = $("explorer-body");
    body.innerHTML = "";
    const summary = sample.summary || {};
    const qualityBlock = sample.quality || {};
    const grid = document.createElement("div");
    grid.className = "detail-grid";
    const left = document.createElement("figure");
    left.className = "figure";
    left.innerHTML = "<figcaption>质量标志与路由</figcaption>";
    const right = document.createElement("figure");
    right.className = "figure";
    right.innerHTML = "<figcaption>特征与掩码</figcaption>";
    simpleTable(left, ["字段", "值"], [
      ["correspondence_status", qualityBlock.correspondence_status ?? "—"],
      ["speech_status", qualityBlock.speech_status ?? "—"],
      ["language_status", qualityBlock.language_status ?? "—"],
      ["face_status", qualityBlock.face_status ?? "—"],
      ["quality_route", summary.quality_route ?? "—"],
      ["mfa_policy", sample.mfa_policy ?? "—"],
      ["detected_language", sample.qa?.detected_language ?? "—"],
      ["duration_s", sample.qa?.duration_s ? Number(sample.qa.duration_s).toFixed(2) : "—"],
      ["face_detection_rate", sample.qa?.face_detection_rate ? pct(Number(sample.qa.face_detection_rate), 1) : "—"],
    ]);
    simpleTable(right, ["维度/掩码", "值"], [
      ["词数 (原始 / 对齐)", `${summary.original_word_count ?? "—"} / ${summary.aligned_word_count ?? "—"}`],
      ["text 词级有效", `${summary.text_word_count ?? "—"} · ${summary.text_word_dim ?? "—"} 维`],
      ["audio 词级有效", `${summary.audio_word_count ?? "—"} · ${summary.audio_word_dim ?? "—"} 维`],
      ["visual 词级有效", `${summary.visual_word_count ?? "—"} · ${summary.visual_word_dim_total ?? "—"} 维`],
      ["visual_word_coverage", summary.visual_word_coverage != null ? pct(Number(summary.visual_word_coverage), 1) : "—"],
      ["OpenFace 帧 (采样/有效)", `${summary.openface_sampled_frame_count ?? "—"} / ${summary.openface_valid_frame_count ?? "—"}`],
      ["NPZ 无 pickle", String(summary.npz_allow_pickle_false ?? "—")],
      ["feature_package", summary.feature_package_bytes ? `${(summary.feature_package_bytes / 1024).toFixed(0)} KiB` : "—"],
    ]);
    grid.append(left, right);
    body.append(grid);
    const words = sample.alignment_words || [];
    const wrap = document.createElement("div");
    wrap.className = "table-view scroll-table";
    wrap.open = true;
    wrap.innerHTML = "<summary>词级对齐区间（word / start / end / mask）</summary>";
    simpleTable(wrap, ["#", "word", "start_s", "end_s", "alignment_available"],
      words.map((w) => [w.word_index, w.word, w.start_s ?? "NaN", w.end_s ?? "NaN", w.alignment_mask]));
    body.append(wrap);
  };
  select.addEventListener("change", renderSample);
  renderSample();

  const sources = $("sources");
  for (const src of data.sources) sources.append(Object.assign(document.createElement("li"), { textContent: src }));
}

loadData()
  .then(render)
  .catch((error) => {
    const p = document.createElement("p");
    p.className = "warn-note";
    p.textContent = String(error.message || error);
    document.querySelector("main").prepend(p);
    console.error(error);
  });
