/* Self-Trading Explorer — renders results/ (bundled in data.js) as interactive SVG charts.
   No dependencies. Every number comes from window.DML_DATA. */
(function () {
  "use strict";
  const D = window.DML_DATA;
  if (!D) {
    document.querySelector("main").innerHTML =
      '<section class="panel"><h2>No data found</h2><p class="sub">Run <code>python webapp/export_data.py</code> after <code>python run_pipeline.py</code> to create <code>webapp/data.js</code>.</p></section>';
    return;
  }

  // ───────────────────────── helpers ─────────────────────────
  const $ = (s, el = document) => el.querySelector(s);
  const NS = "http://www.w3.org/2000/svg";
  const MINUS = "−";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmt = (x, d = 2) => (x == null || isNaN(x) ? "n/a" : (x < 0 ? MINUS : "") + Math.abs(x).toFixed(d));
  const fmtS = (x, d = 2) => (x > 0 ? "+" : "") + fmt(x, d);
  const int = (x) => Math.round(x).toLocaleString("en-US");
  const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
  const pval = (p) => (p == null ? "n/a" : p < 0.001 ? "< 0.001" : p.toFixed(3));
  const ci = (lo, hi, d = 2) => (lo == null ? "n/a" : `[${fmt(lo, d)}, ${fmt(hi, d)}]`);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const ease = (t) => 1 - Math.pow(1 - t, 3);
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const blockOf = (p) => (p <= 53 ? "early" : p <= 72 ? "mid" : "late");
  const blockColor = (b) => css(b === "early" ? "--early" : b === "late" ? "--late" : "--mid");
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const dshort = (s) => { const d = new Date(s + "T00:00:00"); return `${d.getDate()} ${MONTHS[d.getMonth()]}`; };

  function el(tag, attrs = {}, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, str, attrs = {}) {
    const t = el("text", { x, y, ...attrs }, parent);
    t.textContent = str;
    return t;
  }
  function scale(d0, d1, r0, r1) {
    const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
    f.inv = (r) => d0 + ((r - r0) / (r1 - r0)) * (d1 - d0);
    return f;
  }
  function ticks(min, max, n = 5) {
    const span = max - min, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function tween(ms, fn, done) {
    if (reduced) { fn(1); done && done(); return; }
    const t0 = performance.now();
    (function frame(now) {
      const t = clamp((now - t0) / ms, 0, 1);
      fn(ease(t));
      if (t < 1) requestAnimationFrame(frame); else done && done();
    })(t0);
  }
  function svg(container, w, h) {
    container.innerHTML = "";
    return el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" }, container);
  }
  const tip = $("#tooltip");
  function showTip(evt, html) {
    tip.innerHTML = html;
    tip.style.opacity = 1;
    const pad = 14, r = tip.getBoundingClientRect();
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  const hideTip = () => (tip.style.opacity = 0);
  const row = (k, v) => `<div class="row"><span>${k}</span><span>${v}</span></div>`;

  function segmented(target, onChange) {
    const seg = document.querySelector(`.seg[data-target="${target}"]`);
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      onChange(b.dataset.v);
    });
  }

  // ───────────────────────── derived data ─────────────────────────
  const primary = D.models.find((m) => /primary/.test(m.model_name));
  const periods = D.periods.slice().sort((a, b) => a.period - b.period);
  const blockRow = (re) => D.blocks.find((b) => re.test(b.block));
  const early = blockRow(/31/), mid = blockRow(/54/), late = blockRow(/73/);
  const totalTrades = periods.reduce((s, p) => s + p.self_trades, 0);
  const sumBy = (f, rows) => rows.reduce((s, r) => s + f(r), 0);

  $("#hero-rows").textContent = int(D.sample.final_rows);
  $("#hero-traders").textContent = int(D.sample.final_traders);

  // ───────────────────────── 0 · pipeline replay ─────────────────────────
  const STAGES = ["Load", "Features", "DML", "Inference", "Forest", "Diagnostics"];
  const stagesEl = $("#stages");
  stagesEl.innerHTML = STAGES.map((s) => `<div class="stage"><div class="bar"><i></i></div>${s}</div>`).join("");
  const s = D.sample, de = D.descriptives;
  const fi = D.forest.feature_importances_named || {};
  const ah2 = D.feedback.instrument_T_lag2;
  const allNeg = D.grid.every((g) => g.ate < 0);
  const script = [
    [0, `<span class="muted">[load]</span> Reading XTrader panel …`],
    [0, `  ${int(s.total)} trader-weeks after the country filter`],
    [1, `<span class="muted">[features]</span> Lags, tenure, audience-gap repair, pre-window moderators`],
    [1, `  dropped ${int(s.dropped_missing_lags)} first weeks (no lags), ${int(s.dropped_pre_join)} pre-join weeks`],
    [1, `  analysis sample: <span class="hl">${int(s.final_rows)}</span> trader-weeks · ${int(s.final_traders)} traders · periods 31–75`],
    [1, `  ${pct(de.share_no_trades)} of trader-weeks have no trades; ${pct(de.share_traders_never_self_trade)} of traders never self-trade`],
    [2, `<span class="muted">[dml]</span> Two-way demeaning (trader + week) by alternating projections … <span class="ok">converged</span>`],
    [2, `  LinearDML · HistGBR nuisances · 5 trader-grouped folds`],
    [2, `  nuisance R²: outcome ${pct(primary.nuisance_r2_y)}, treatment ${pct(primary.nuisance_r2_t)}`],
    [2, `  θ = <span class="hl">${fmt(primary.ate)}</span>  trader-clustered CI ${ci(primary.ate_ci_lower, primary.ate_ci_upper)}`],
    [3, `<span class="muted">[inference]</span> Wild cluster bootstrap over ${primary.n_time_clusters} weeks, 9,999 Webb draws …`],
    [3, `  effective number of weeks: <span class="warn">${primary.effective_time_clusters.toFixed(1)}</span>`],
    [3, `  week-bootstrap CI <span class="hl">${ci(primary.ate_ci_lower_week_wild, primary.ate_ci_upper_week_wild)}</span>, p = ${pval(primary.ate_pvalue_week_wild)}`],
    [3, `  decomposing into ${periods.length} week contributions · break after period <span class="hl">${D.regime_break.break_after_period}</span>`],
    [3, `  periods 31–53: ${fmt(early.ate)} · periods 73–75: ${fmt(late.ate)}`],
    [4, `<span class="muted">[forest]</span> CausalForestDML on half A (${int(D.forest.n_fit_traders)} traders), tested on held-out half B`],
    [4, `  importance on calendar period: <span class="warn">${(fi.period || 0).toFixed(2)}</span> · GATES equal-effects p ${pval(D.gates.wald_equal_pvalue)}`],
    [5, `<span class="muted">[robustness]</span> 24-spec grid … ${allNeg ? '<span class="ok">all 24 negative</span>' : "mixed signs"}`],
    [5, `  lead test: week-level t = ${D.lead.linear_fe_lead_t_week.toFixed(1)}, p = ${pval(D.lead.linear_fe_lead_pvalue_week_wild)} → <span class="warn">strict exogeneity rejected</span>`],
    [5, `  Anderson–Hsiao (T in t−2): ${fmtS(ah2.theta)} ${ci(ah2.ci_lower_twoway, ah2.ci_upper_twoway, 1)}`],
    [5, `  placebo (${D.placebo.length} permutations): ${D.placebo.map((p) => fmt(p.ate)).join(", ")}`],
    [5, `<span class="ok">✓ done.</span> results/ written · scroll down to explore`],
  ];
  const logEl = $("#log"), statusEl = $("#console-status"), runBtn = $("#run-btn");
  let running = false;

  function setStage(i, frac) {
    [...stagesEl.children].forEach((st, k) => {
      const w = k < i ? 100 : k === i ? frac * 100 : 0;
      st.querySelector("i").style.width = w + "%";
      st.classList.toggle("done", k < i || (k === i && frac >= 1));
      st.classList.toggle("active", k === i && frac < 1);
    });
  }
  function finishReplay() {
    running = false;
    runBtn.disabled = false;
    runBtn.innerHTML = '<span class="play"></span>Run again';
    statusEl.textContent = "exit 0";
    statusEl.className = "console-status done";
    setStage(STAGES.length, 1);
    revealKPIs();
  }
  function runReplay(fast) {
    if (running) return;
    running = true;
    runBtn.disabled = true;
    statusEl.textContent = "running";
    statusEl.className = "console-status running";
    logEl.innerHTML = '<span class="muted">$ python run_pipeline.py</span>\n';
    let i = 0;
    const counts = STAGES.map((_, k) => script.filter((l) => l[0] === k).length);
    const seen = STAGES.map(() => 0);
    (function next() {
      if (i >= script.length) { finishReplay(); return; }
      const [st, line] = script[i++];
      seen[st]++;
      setStage(st, seen[st] / counts[st]);
      logEl.innerHTML = logEl.innerHTML.replace('<span class="cursor"></span>', "") + line + '\n<span class="cursor"></span>';
      logEl.scrollTop = logEl.scrollHeight;
      setTimeout(next, fast || reduced ? 0 : line.startsWith("<span") ? 520 : 260);
    })();
  }
  runBtn.addEventListener("click", () => runReplay(false));
  $("#skip-btn").addEventListener("click", () => {
    if (!running) runReplay(true);
    document.getElementById("kpis").scrollIntoView({ behavior: "smooth" });
  });

  let kpisShown = false;
  function countUp(node, target, d, suffix = "") {
    tween(1100, (t) => (node.textContent = fmt(target * t, d) + suffix));
  }
  function revealKPIs() {
    if (kpisShown) return;
    kpisShown = true;
    document.querySelectorAll(".kpi").forEach((k, i) => setTimeout(() => k.classList.add("show"), i * 120));
    const q = (k) => document.querySelector(`[data-kpi="${k}"]`), qs = (k) => document.querySelector(`[data-kpi-sub="${k}"]`);
    countUp(q("pooled"), primary.ate, 2);
    qs("pooled").textContent = `CI ${ci(primary.ate_ci_lower_week_wild, primary.ate_ci_upper_week_wild)} · p ${pval(primary.ate_pvalue_week_wild)}`;
    countUp(q("early"), early.ate, 2);
    qs("early").textContent = `CI ${ci(early.ci_lower_week_wild, early.ci_upper_week_wild)}`;
    countUp(q("late"), late.ate, 2);
    qs("late").textContent = `CI ${ci(late.ci_lower, late.ci_upper)} (trader)`;
    tween(1100, (t) => (q("geff").textContent = (primary.effective_time_clusters * t).toFixed(1)));
    qs("geff").textContent = `of ${primary.n_time_clusters} weeks in the window`;
  }
  // reveal KPIs anyway if the user scrolls past without running
  new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setTimeout(revealKPIs, 600)), { threshold: 0.6 })
    .observe($("#kpis"));

  // ───────────────────────── 1 · activity ─────────────────────────
  const W = 1000;
  function blockBands(g, x, top, bottom, labels = true) {
    const ranges = [["early", 31, 53, "Jan – mid-Jun"], ["mid", 54, 72, "near-empty"], ["late", 73, 75, "final 3 wks"]];
    for (const [b, a, z, lab] of ranges) {
      const x0 = x(a - 0.5), x1 = x(z + 0.5);
      el("rect", { x: x0, y: top, width: x1 - x0, height: bottom - top, fill: blockColor(b), opacity: 0.06 }, g);
      if (labels) text(g, (x0 + x1) / 2, top + 14, lab, { "text-anchor": "middle", fill: blockColor(b), "font-weight": 600, "font-size": 12 });
    }
  }
  function periodAxis(g, x, y) {
    for (const p of periods) {
      if (p.period % 5 === 0 || p.period === 31 || p.period === 75) {
        text(g, x(p.period), y + 16, p.period, { "text-anchor": "middle" });
        text(g, x(p.period), y + 30, dshort(p.start_date), { "text-anchor": "middle", "font-size": 10, fill: css("--faint") });
      }
    }
  }
  function drawActivity(metric) {
    const H = 360, m = { l: 64, r: 16, t: 26, b: 48 };
    const root = svg($("#chart-activity"), W, H);
    const x = scale(30.5, 75.5, m.l, W - m.r);
    const bw = (x(1) - x(0)) * 0.78;
    const val = (p) => metric === "trades" ? p.self_trades : metric === "active" ? p.share_active * 100 : p.raw_self_gain_per_self_trade;
    let lo = 0, hi = Math.max(...periods.map(val));
    if (metric === "pnl") { lo = Math.min(-90, ...periods.map(val)); hi = 30; }
    const yLog = metric === "trades";
    const y = yLog ? ((v) => { const f = scale(0, Math.log10(hi * 1.3), H - m.b, m.t + 10); return f(Math.log10(Math.max(v, 1))); })
                   : scale(lo, hi * 1.08, H - m.b, m.t + 10);
    blockBands(root, x, m.t, H - m.b);
    const grid = el("g", { class: "grid" }, root);
    const tv = yLog ? [1, 10, 100, 1000, 10000].filter((v) => v <= hi * 1.3) : ticks(lo, hi * 1.08, 5);
    for (const v of tv) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, grid);
      text(root, m.l - 8, y(v) + 4, yLog ? int(v) : metric === "active" ? v + "%" : fmt(v, 0), { "text-anchor": "end" });
    }
    text(root, 14, (H - m.b + m.t) / 2, metric === "trades" ? "Self-trades (log scale)" : metric === "active" ? "% of trader-weeks with any trade" : "Raw P&L per self-trade",
      { transform: `rotate(-90 14 ${(H - m.b + m.t) / 2})`, "text-anchor": "middle" });
    const base = metric === "pnl" ? y(0) : H - m.b;
    if (metric === "pnl") el("line", { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), class: "zero" }, root);
    const bars = periods.map((p) => {
      const r = el("rect", { x: x(p.period) - bw / 2, width: bw, y: base, height: 0, rx: 2, fill: blockColor(blockOf(p.period)),
        opacity: metric === "pnl" && p.self_trades < 100 ? 0.35 : 0.95 }, root);
      r.addEventListener("mousemove", (e) => showTip(e,
        `<b>Period ${p.period}</b> · ${dshort(p.start_date)} – ${dshort(p.end_date)} 2013` +
        row("Self-trades", int(p.self_trades)) + row("Trader-weeks active", pct(p.share_active)) +
        row("Raw P&L / self-trade", fmt(p.raw_self_gain_per_self_trade)) + row("Weight in pooled θ", pct(p.weight))));
      r.addEventListener("mouseleave", hideTip);
      return [r, p];
    });
    tween(900, (t) => bars.forEach(([r, p], i) => {
      const tt = clamp(t * 1.6 - i / periods.length * 0.6, 0, 1);
      const v = val(p), yy = metric === "pnl" ? y(v * tt) : y(yLog ? Math.pow(10, Math.log10(Math.max(v, 1)) * tt) : v * tt);
      r.setAttribute("y", Math.min(yy, base)); r.setAttribute("height", Math.abs(base - yy));
    }));
    periodAxis(root, x, H - m.b);
    if (metric === "pnl") text(root, W - m.r, m.t + 30, "faded bars: weeks with < 100 self-trades", { "text-anchor": "end", "font-size": 11, fill: css("--faint") });
  }
  segmented("activity", drawActivity);
  const shareOf = (a, z) => sumBy((p) => p.self_trades, periods.filter((p) => p.period >= a && p.period <= z)) / totalTrades;
  $("#data-facts").innerHTML = [
    [pct(de.share_no_trades), "of trader-weeks have no trades at all"],
    [pct(shareOf(73, 75)), "of all self-trades fall in the last 3 weeks (29 Oct – 18 Nov)"],
    [int(mid.self_trades), "self-trades in the 19 weeks between 18 Jun and 28 Oct"],
  ].map(([b, s]) => `<div class="fact"><b>${b}</b><span>${s}</span></div>`).join("");

  // ───────────────────────── 2 · CI comparison ─────────────────────────
  const ciModels = D.models.filter((m) => m.ate_ci_lower_week_wild != null).map((m) => ({
    name: m.model_name.replace("LinearDML", "DML").replace("trader + week FE (primary)", "trader + week FE ★").replace(" (clustered)", ""),
    m,
  }));
  const ciKeys = {
    trader: (m) => [m.ate_ci_lower, m.ate_ci_upper, m.ate_pvalue],
    twoway: (m) => [m.ate_ci_lower_twoway, m.ate_ci_upper_twoway, m.ate_pvalue_twoway],
    wild: (m) => [m.ate_ci_lower_week_wild, m.ate_ci_upper_week_wild, m.ate_pvalue_week_wild],
  };
  const ciNotes = {
    trader: "Trader-clustered: every interval excludes zero by a wide margin. This assumes a market-wide shock in one week does not move all traders together.",
    twoway: "Trader × week clustering allows common weekly shocks. Every interval now includes zero.",
    wild: `Week wild cluster bootstrap (primary): reliable with few effective clusters. The primary CI is ${ci(primary.ate_ci_lower_week_wild, primary.ate_ci_upper_week_wild)}, p = ${pval(primary.ate_pvalue_week_wild)}: borderline.`,
  };
  let ciState = null;
  function drawCI(kind) {
    const H = 300, m = { l: 210, r: 190, t: 20, b: 40 };
    const lo = -22, hi = 4;
    const x = scale(lo, hi, m.l, W - m.r);
    const rowH = (H - m.t - m.b) / ciModels.length;
    let parts = ciState;
    if (!parts) {
      const root = svg($("#chart-ci"), W, H);
      const grid = el("g", { class: "grid" }, root);
      for (const v of ticks(lo, hi, 6)) {
        el("line", { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }, grid);
        text(root, x(v), H - m.b + 18, fmt(v, 0), { "text-anchor": "middle" });
      }
      el("line", { x1: x(0), x2: x(0), y1: m.t - 6, y2: H - m.b, class: "zero" }, root);
      text(root, (m.l + W - m.r) / 2, H - 4, "Effect of one extra self-trade on weekly gain", { "text-anchor": "middle" });
      parts = ciModels.map((c, i) => {
        const cy = m.t + rowH * (i + 0.5);
        const star = /★/.test(c.name);
        text(root, m.l - 14, cy + 4, c.name, { "text-anchor": "end", class: star ? "lbl-strong" : null });
        const band = el("rect", { y: cy - 9, height: 18, rx: 9, fill: css("--accent"), opacity: star ? 0.32 : 0.18 }, root);
        el("circle", { cx: x(c.m.ate), cy, r: star ? 7 : 5.5, fill: css("--accent"), stroke: css("--card"), "stroke-width": 2 }, root);
        const lab = text(root, W - m.r + 14, cy + 4, "", { class: "mono" });
        return { c, band, lab, lo: x(c.m.ate), hi: x(c.m.ate) };
      });
      ciState = parts;
    }
    const from = parts.map((p) => [p.lo, p.hi]);
    const to = parts.map((p) => { const [a, b] = ciKeys[kind](p.c.m); return [x(clamp(a, lo, hi)), x(clamp(b, lo, hi))]; });
    parts.forEach((p) => { const [a, b, pv] = ciKeys[kind](p.c.m); p.lab.textContent = `${ci(a, b)}  p ${pval(pv)}`; p.lab.setAttribute("fill", a < 0 && b < 0 ? css("--text") : css("--muted")); });
    tween(800, (t) => parts.forEach((p, i) => {
      p.lo = from[i][0] + (to[i][0] - from[i][0]) * t;
      p.hi = from[i][1] + (to[i][1] - from[i][1]) * t;
      p.band.setAttribute("x", p.lo); p.band.setAttribute("width", Math.max(p.hi - p.lo, 2));
    }));
    $("#ci-note").textContent = ciNotes[kind];
  }
  segmented("ci", drawCI);

  // ───────────────────────── 3 · decomposition ─────────────────────────
  const wsum = (f) => sumBy(f, periods);
  const weightSets = {
    variance_weighted: periods.map((p) => p.weight),
    trade_weighted: periods.map((p) => p.self_trades / totalTrades),
    equal_week: periods.map(() => 1 / periods.length),
  };
  let decompState = null;
  function drawDecomp(kind) {
    const H = 520, m = { l: 64, r: 20, t: 18, b: 48 };
    const topH = 150, gap = 36;
    const x = scale(30.5, 75.5, m.l, W - m.r);
    const bw = (x(1) - x(0)) * 0.78;
    const yW = scale(0, 0.45, m.t + topH, m.t + 10);
    const y0 = m.t + topH + gap, y1 = H - m.b;
    const yC = scale(-45, 10, y1, y0);
    if (!decompState) {
      const root = svg($("#chart-decomp"), W, H);
      blockBands(root, x, m.t, y1, false);
      const g = el("g", { class: "grid" }, root);
      for (const v of [0.1, 0.2, 0.3, 0.4]) { el("line", { x1: m.l, x2: W - m.r, y1: yW(v), y2: yW(v) }, g); text(root, m.l - 8, yW(v) + 4, pct(v, 0), { "text-anchor": "end" }); }
      for (const v of ticks(-45, 10, 6)) { el("line", { x1: m.l, x2: W - m.r, y1: yC(v), y2: yC(v) }, g); text(root, m.l - 8, yC(v) + 4, fmt(v, 0), { "text-anchor": "end" }); }
      text(root, m.l, m.t + 4, "Weight of each week in the average", { "font-weight": 600, fill: css("--text") });
      text(root, m.l, y0 - 8, "Week contribution (effect per extra self-trade, 95% CI)", { "font-weight": 600, fill: css("--text") });
      el("line", { x1: m.l, x2: W - m.r, y1: yC(0), y2: yC(0), class: "zero" }, root);
      // regime break means
      const rb = D.regime_break;
      el("line", { x1: x(30.5), x2: x(rb.break_after_period + 0.5), y1: yC(rb.mean_before), y2: yC(rb.mean_before), stroke: css("--early"), "stroke-width": 3, opacity: 0.55 }, root);
      el("line", { x1: x(rb.break_after_period + 0.5), x2: x(75.5), y1: yC(rb.mean_after), y2: yC(rb.mean_after), stroke: css("--late"), "stroke-width": 3, opacity: 0.55 }, root);
      el("line", { x1: x(rb.break_after_period + 0.5), x2: x(rb.break_after_period + 0.5), y1: y0, y2: y1, stroke: css("--text"), "stroke-dasharray": "2 4", opacity: 0.5 }, root);
      text(root, x(rb.break_after_period + 0.5) + 6, y0 + 16, `break after period ${rb.break_after_period} (17 Jun)`, { "font-size": 11, fill: css("--text") });
      text(root, x(42), yC(rb.mean_before) - 8, `regime mean ${fmt(rb.mean_before, 1)}`, { "text-anchor": "middle", "font-size": 11, fill: css("--early") });
      const bars = periods.map((p) => el("rect", { x: x(p.period) - bw / 2, width: bw, y: yW(0), height: 0, rx: 2, fill: blockColor(blockOf(p.period)) }, root));
      const aggLine = el("line", { x1: m.l, x2: W - m.r, stroke: css("--text"), "stroke-width": 2, "stroke-dasharray": "7 5" }, root);
      const aggLab = text(root, x(33.5), 0, "", { "text-anchor": "start", class: "lbl-strong" });
      const looLine = el("line", { x1: m.l, x2: W - m.r, stroke: css("--accent-2"), "stroke-width": 2, opacity: 0 }, root);
      for (const p of periods) {
        const cx = x(p.period), c = blockColor(blockOf(p.period));
        const loC = clamp(p.slope_ci_lower, -45, 10), hiC = clamp(p.slope_ci_upper, -45, 10);
        el("line", { x1: cx, x2: cx, y1: yC(loC), y2: yC(hiC), stroke: c, "stroke-width": 1.5, opacity: 0.8 }, root);
        const r = 3 + Math.sqrt(p.weight) * 22;
        const dot = el("circle", { cx, cy: yC(clamp(p.slope, -45, 10)), r, fill: c, opacity: 0.9, stroke: css("--card"), "stroke-width": 1.5 }, root);
        const hit = el("rect", { x: cx - (x(1) - x(0)) / 2, y: m.t, width: x(1) - x(0), height: y1 - m.t, fill: "transparent" }, root);
        hit.addEventListener("mousemove", (e) => {
          dot.setAttribute("stroke", css("--text"));
          looLine.setAttribute("y1", yC(p.estimate_leave_week_out)); looLine.setAttribute("y2", yC(p.estimate_leave_week_out)); looLine.setAttribute("opacity", 0.9);
          showTip(e, `<b>Period ${p.period}</b> · ${dshort(p.start_date)} – ${dshort(p.end_date)}` +
            row("Weight (DML)", pct(p.weight)) + row("Contribution", fmt(p.slope)) + row("95% CI", ci(p.slope_ci_lower, p.slope_ci_upper)) +
            row("Self-trades", int(p.self_trades)) + row("θ without this week", `<span style="color:${css("--accent-2")}">${fmt(p.estimate_leave_week_out)}</span>`));
        });
        hit.addEventListener("mouseleave", () => { hideTip(); dot.setAttribute("stroke", css("--card")); looLine.setAttribute("opacity", 0); });
      }
      periodAxis(root, x, y1);
      decompState = { bars, aggLine, aggLab, cur: periods.map(() => 0), curAgg: 0 };
    }
    const st = decompState, target = weightSets[kind], from = st.cur.slice(), fromAgg = st.curAgg, toAgg = D.aggregates[kind];
    tween(900, (t) => {
      st.bars.forEach((b, i) => {
        const v = from[i] + (target[i] - from[i]) * t; st.cur[i] = v;
        b.setAttribute("y", yW(v)); b.setAttribute("height", yW(0) - yW(v));
      });
      st.curAgg = fromAgg + (toAgg - fromAgg) * t;
      st.aggLine.setAttribute("y1", yC(st.curAgg)); st.aggLine.setAttribute("y2", yC(st.curAgg));
      st.aggLab.setAttribute("y", yC(st.curAgg) - 7);
      st.aggLab.textContent = `average = ${fmt(st.curAgg)}`;
    });
  }
  segmented("agg", drawDecomp);
  const bCard = (cls, title, b, note, tag) =>
    `<div class="block-card ${cls}"><span>${title}</span><b>${fmt(b.ate)}</b>` +
    `<span class="ci">trader ${ci(b.ci_lower, b.ci_upper)}</span>` +
    `<span class="ci">week bootstrap ${b.ci_lower_week_wild == null ? "n/a (3 weeks)" : ci(b.ci_lower_week_wild, b.ci_upper_week_wild)}</span>` +
    `<span>${int(b.self_trades)} self-trades · ${note}</span><br><span class="tag ${tag[0]}">${tag[1]}</span></div>`;
  $("#block-cards").innerHTML =
    bCard("early", "Periods 31–53 · 8 Jan – 17 Jun", early, "all 23 weeks negative", ["ok", "survives week-level inference"]) +
    bCard("", "Periods 54–72 · 18 Jun – 28 Oct", mid, "too thin to interpret", ["no", "not interpretable"]) +
    bCard("late", "Periods 73–75 · 29 Oct – 18 Nov", late, `${pct(shareOf(73, 75), 0)} of all self-trades`, ["warn", "not distinguishable from zero"]);

  // ───────────────────────── 4 · P&L scatter ─────────────────────────
  const busy = periods.filter((p) => p.self_trades >= 100);
  function corr(a, b) {
    const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
    return sab / Math.sqrt(saa * sbb);
  }
  function drawPnL() {
    const Wp = 640, H = 460, m = { l: 60, r: 20, t: 20, b: 54 };
    const root = svg($("#chart-pnl"), Wp, H);
    const x = scale(-90, 10, m.l, Wp - m.r), y = scale(-45, 10, H - m.b, m.t);
    const g = el("g", { class: "grid" }, root);
    for (const v of ticks(-90, 10, 5)) { el("line", { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }, g); text(root, x(v), H - m.b + 18, fmt(v, 0), { "text-anchor": "middle" }); }
    for (const v of ticks(-45, 10, 5)) { el("line", { x1: m.l, x2: Wp - m.r, y1: y(v), y2: y(v) }, g); text(root, m.l - 8, y(v) + 4, fmt(v, 0), { "text-anchor": "end" }); }
    el("line", { x1: x(-45), y1: y(-45), x2: x(10), y2: y(10), stroke: css("--faint"), "stroke-dasharray": "5 5" }, root);
    text(root, x(-40), y(-40) - 8, "model = raw", { fill: css("--faint"), "font-size": 11, transform: `rotate(-${Math.atan((y(-45) - y(10)) / (x(10) - x(-45))) * 180 / Math.PI} ${x(-40)} ${y(-40) - 8})` });
    text(root, (m.l + Wp - m.r) / 2, H - 10, "Raw average P&L per self-trade (no model)", { "text-anchor": "middle" });
    text(root, 14, (H - m.b + m.t) / 2, "Model week contribution", { transform: `rotate(-90 14 ${(H - m.b + m.t) / 2})`, "text-anchor": "middle" });
    const dots = busy.map((p) => {
      const c = el("circle", { cx: x(clamp(p.raw_self_gain_per_self_trade, -90, 10)), cy: y(0), r: 0, fill: blockColor(blockOf(p.period)), opacity: 0.85, stroke: css("--card"), "stroke-width": 1.5 }, root);
      c.addEventListener("mousemove", (e) => showTip(e, `<b>Period ${p.period}</b> · ${dshort(p.start_date)}` + row("Raw P&L / trade", fmt(p.raw_self_gain_per_self_trade)) + row("Contribution", fmt(p.slope)) + row("Self-trades", int(p.self_trades))));
      c.addEventListener("mouseleave", hideTip);
      return [c, p];
    });
    tween(1000, (t) => dots.forEach(([c, p]) => { c.setAttribute("cy", y(clamp(p.slope, -45, 10) * t)); c.setAttribute("r", (4 + Math.sqrt(p.self_trades) / 9) * t); }));
  }
  drawPnL();
  const r = corr(busy.map((p) => p.raw_self_gain_per_self_trade), busy.map((p) => p.slope));
  $("#pnl-corr").innerHTML = `<b>${r.toFixed(2)}</b>correlation across the ${busy.length} weeks with ≥ 100 self-trades`;
  const blockAgg = (a, z) => {
    const rows = periods.filter((p) => p.period >= a && p.period <= z);
    return [sumBy((p) => p.slope * p.weight, rows) / sumBy((p) => p.weight, rows),
            sumBy((p) => p.raw_self_gain_per_self_trade * p.self_trades, rows) / sumBy((p) => p.self_trades, rows),
            sumBy((p) => p.self_trades, rows)];
  };
  const [e1, e2, e3] = blockAgg(31, 53), [l1, l2, l3] = blockAgg(73, 75);
  $("#pnl-table").innerHTML = `<tr><th>Block</th><th class="num">Model</th><th class="num">Raw P&L / trade</th><th class="num">Trades</th></tr>` +
    `<tr><td><span style="color:${css("--early")}">●</span> 31–53</td><td class="num">${fmt(e1)}</td><td class="num">${fmt(e2)}</td><td class="num">${int(e3)}</td></tr>` +
    `<tr><td><span style="color:${css("--late")}">●</span> 73–75</td><td class="num">${fmt(l1)}</td><td class="num">${fmt(l2)}</td><td class="num">${int(l3)}</td></tr>`;
  const ck = (re) => D.checks.find((c) => re.test(c.check));
  const cg = ck(/copied-trade gain only \(raw\)/), cc = ck(/copied-trade count/);
  $("#crowding").innerHTML = `<b>No crowding out.</b> One extra self-trade changes copied-trade gains by ${fmt(cg.ate)} ${ci(cg.ci_lower, cg.ci_upper)} ` +
    `and the number of copied trades by ${fmtS(cc.ate)} (week-bootstrap p = ${pval(cc.pvalue_week_wild)}).`;

  // ───────────────────────── 5 · heterogeneity ─────────────────────────
  function drawGates() {
    const Wg = 640, H = 400, m = { l: 60, r: 20, t: 30, b: 70 };
    const root = svg($("#chart-gates"), Wg, H);
    const groups = D.gates.groups;
    const y = scale(-35, 5, H - m.b, m.t);
    const x0 = m.l, bwid = (Wg - m.l - m.r) / groups.length;
    const g = el("g", { class: "grid" }, root);
    for (const v of ticks(-35, 5, 8)) { el("line", { x1: m.l, x2: Wg - m.r, y1: y(v), y2: y(v) }, g); text(root, m.l - 8, y(v) + 4, fmt(v, 0), { "text-anchor": "end" }); }
    el("line", { x1: m.l, x2: Wg - m.r, y1: y(0), y2: y(0), class: "zero" }, root);
    const mix = (s) => {
      const a = css("--early"), b = css("--late");
      const h = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
      const [r1, g1, b1] = h(a), [r2, g2, b2] = h(b);
      return `rgb(${r1 + (r2 - r1) * s | 0},${g1 + (g2 - g1) * s | 0},${b1 + (b2 - b1) * s | 0})`;
    };
    const bars = groups.map((q, i) => {
      const cx = x0 + bwid * (i + 0.5), w = bwid * 0.56;
      const b = el("rect", { x: cx - w / 2, width: w, y: y(0), height: 0, rx: 6, fill: mix(q.share_self_trades_late_block) }, root);
      el("line", { x1: cx, x2: cx, y1: y(q.ci_lower), y2: y(q.ci_upper), stroke: css("--text"), "stroke-width": 2 }, root);
      text(root, cx, H - m.b + 20, `Q${q.group}`, { "text-anchor": "middle", class: "lbl-strong" });
      text(root, cx, H - m.b + 38, `${pct(q.share_self_trades_late_block, 0)} in final 3 wks`, { "text-anchor": "middle", "font-size": 11 });
      text(root, cx, (q.gate < 0 ? y(q.gate) + 18 : y(q.gate) - 8), fmt(q.gate, 1), { "text-anchor": "middle", class: "mono", fill: css("--text") });
      b.addEventListener("mousemove", (e) => showTip(e, `<b>Quartile ${q.group}</b>` + row("Held-out effect", fmt(q.gate)) + row("95% CI", ci(q.ci_lower, q.ci_upper)) + row("Self-trades in periods 73–75", pct(q.share_self_trades_late_block))));
      b.addEventListener("mouseleave", hideTip);
      return [b, q];
    });
    text(root, m.l, 16, "Held-out effect by quartile of the forest's predicted effect", { "font-weight": 600, fill: css("--text") });
    text(root, (m.l + Wg - m.r) / 2, H - 8, "colour: blue = trades early in the window, orange = trades in the final 3 weeks", { "text-anchor": "middle", "font-size": 11, fill: css("--faint") });
    tween(900, (t) => bars.forEach(([b, q]) => { const yy = y(q.gate * t); b.setAttribute("y", Math.min(yy, y(0))); b.setAttribute("height", Math.abs(y(0) - yy)); }));
  }
  drawGates();
  function drawImportance() {
    const names = { period: "Calendar week", join_cohort_weeks: "Join cohort", followers_start: "Followers (wk 30)", copiers_start: "Copiers (wk 30)" };
    const items = Object.entries(fi).sort((a, b) => b[1] - a[1]);
    const Wi = 420, H = 30 + items.length * 34, m = { l: 130, r: 50 };
    const root = svg($("#chart-importance"), Wi, H);
    const x = scale(0, 1, m.l, Wi - m.r);
    items.forEach(([k, v], i) => {
      const yy = 10 + i * 34;
      text(root, m.l - 10, yy + 15, names[k] || k, { "text-anchor": "end" });
      el("rect", { x: m.l, y: yy + 3, width: x(1) - m.l, height: 16, rx: 8, fill: css("--grid") }, root);
      const b = el("rect", { x: m.l, y: yy + 3, width: 0, height: 16, rx: 8, fill: k === "period" ? css("--late") : css("--accent") }, root);
      text(root, Wi - m.r + 8, yy + 15, v.toFixed(3), { class: "mono" });
      tween(1000, (t) => b.setAttribute("width", Math.max((x(v) - m.l) * t, 0)));
    });
  }
  drawImportance();
  const modNames = { join_cohort_weeks: "Join cohort", copiers_start: "Copiers (wk 30)", followers_start: "Followers (wk 30)" };
  const mods = D.moderators.terms.filter((t) => !/^block/.test(t.term));
  $("#mod-table").innerHTML = `<tr><th>Moderator (SD)</th><th class="num">Coef.</th><th class="num">p trader</th><th class="num">p two-way</th></tr>` +
    mods.map((t) => `<tr><td>${modNames[t.term] || t.term}</td><td class="num">${fmtS(t.coef)}</td><td class="num">${pval(t.pvalue)}</td><td class="num">${pval(t.pvalue_twoway)}</td></tr>`).join("") +
    `<tr><td colspan="4" style="color:var(--faint);font-size:12px">With week-block intercepts. Joint test p = ${pval(D.moderators.joint_moderators_pvalue_twoway)} (two-way).</td></tr>`;

  // ───────────────────────── 6 · robustness ─────────────────────────
  const scales = [...new Set(D.grid.map((g) => `${g.treatment_type} / ${g.outcome}`))];
  const order = ["count / winsorized", "count / asinh", "log1p / winsorized", "log1p / asinh"].filter((s) => scales.includes(s));
  $("#spec-tabs").innerHTML = order.map((s, i) => `<button class="${i ? "" : "on"}" data-v="${s}">${s}</button>`).join("");
  function drawSpec(sc) {
    const rows = D.grid.filter((g) => `${g.treatment_type} / ${g.outcome}` === sc).sort((a, b) => a.ate - b.ate);
    const H = 330, m = { l: 70, r: 20, t: 20, b: 80 };
    const root = svg($("#chart-spec"), W, H);
    const lo = Math.min(...rows.map((r) => r.ci_lower_week_wild)), hi = Math.max(0, ...rows.map((r) => r.ci_upper_week_wild));
    const pad = (hi - lo) * 0.08;
    const y = scale(lo - pad, hi + pad, H - m.b, m.t);
    const g = el("g", { class: "grid" }, root);
    for (const v of ticks(lo - pad, hi + pad, 6)) { el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, g); text(root, m.l - 8, y(v) + 4, Math.abs(v) < 1 && v !== 0 ? fmt(v, 2) : fmt(v, 0), { "text-anchor": "end" }); }
    el("line", { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), class: "zero" }, root);
    const step = (W - m.l - m.r) / rows.length;
    const lrn = { histgbr: "HistGBR", lightgbm: "LightGBM", lasso: "Lasso" };
    rows.forEach((r, i) => {
      const cx = m.l + step * (i + 0.5);
      const sig = r.pvalue_week_wild < 0.05;
      const band = el("rect", { x: cx - 13, width: 26, y: y(0), height: 0, rx: 13, fill: css("--accent"), opacity: sig ? 0.3 : 0.14 }, root);
      el("line", { x1: cx, x2: cx, y1: y(r.ci_lower), y2: y(r.ci_upper), stroke: css("--text"), "stroke-width": 3 }, root);
      el("circle", { cx, cy: y(r.ate), r: 6, fill: css("--accent"), stroke: css("--card"), "stroke-width": 2 }, root);
      text(root, cx, H - m.b + 20, lrn[r.nuisance_learner] || r.nuisance_learner, { "text-anchor": "middle", "font-size": 11.5 });
      text(root, cx, H - m.b + 36, r.sample === "all" ? "all traders" : "present ≥ 23 wks", { "text-anchor": "middle", "font-size": 10.5, fill: css("--faint") });
      text(root, cx, H - m.b + 54, sig ? "p < 0.05" : `p ${pval(r.pvalue_week_wild)}`, { "text-anchor": "middle", "font-size": 10.5, fill: sig ? css("--good") : css("--faint") });
      tween(700 + i * 60, (t) => { const a = y(0) + (y(r.ci_upper_week_wild) - y(0)) * t, b = y(0) + (y(r.ci_lower_week_wild) - y(0)) * t; band.setAttribute("y", Math.min(a, b)); band.setAttribute("height", Math.abs(b - a)); });
      const hit = el("rect", { x: cx - step / 2, y: m.t, width: step, height: H - m.b - m.t, fill: "transparent" }, root);
      hit.addEventListener("mousemove", (e) => showTip(e, `<b>${lrn[r.nuisance_learner]} · ${r.sample}</b>` + row("Estimate", fmt(r.ate, 3)) + row("Trader CI", ci(r.ci_lower, r.ci_upper, 3)) + row("Week-bootstrap CI", ci(r.ci_lower_week_wild, r.ci_upper_week_wild, 3)) + row("Week-bootstrap p", pval(r.pvalue_week_wild)) + row("Effective weeks", r.effective_weeks.toFixed(1))));
      hit.addEventListener("mouseleave", hideTip);
    });
  }
  segmented("spec", drawSpec);
  drawSpec(order[0]);

  function drawChecks() {
    const rows = D.checks.filter((c) => !/Ratio/.test(c.check) && !/copied-trade count/.test(c.check) && !/copied-trade gain/.test(c.check));
    const short = (c) => c.replace("Primary: two-way FE, winsorized outcome", "★ Primary: two-way FE")
      .replace(/Self-trades > p99\.9.*$/, "Extreme weeks trimmed (> p99.9)")
      .replace("First-difference DML, nuisances on controls in t and t-1 (exact FD)", "Exact first-difference DML")
      .replace("First-difference DML, nuisances on differenced controls", "First-difference DML (approx.)")
      .replace("Correlated random effects (Mundlak) DML", "Correlated random effects DML")
      .replace("Dynamic: + lagged outcome (Nickell-biased, comparison only)", "+ lagged outcome (comparison)")
      .replace("No FE (pooled levels + country dummies)", "No fixed effects")
      .replace("Trader FE only (no week FE)", "Trader FE only")
      .replace("Raw outcome (total_gain, not winsorized)", "Raw (not winsorized) outcome")
      .replace("Primary on first-difference sample (same rows)", "Primary on first-difference rows")
      .replace("Outcome: self-trade gain only (raw)", "Self-trade gain only (raw)")
      .replace("Outcome: self-trade gain only (winsorized)", "Self-trade gain only");
    const H = 50 + rows.length * 30, m = { l: 290, r: 200, t: 10, b: 36 };
    const root = svg($("#chart-checks"), W, H);
    const lo = -30, hi = 5, x = scale(lo, hi, m.l, W - m.r);
    const g = el("g", { class: "grid" }, root);
    for (const v of ticks(lo, hi, 7)) { el("line", { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }, g); text(root, x(v), H - m.b + 18, fmt(v, 0), { "text-anchor": "middle" }); }
    el("line", { x1: x(0), x2: x(0), y1: m.t, y2: H - m.b, class: "zero" }, root);
    rows.forEach((c, i) => {
      const cy = m.t + 15 + i * 30, star = /Primary: two-way/.test(c.check);
      text(root, m.l - 12, cy + 4, short(c.check), { "text-anchor": "end", class: star ? "lbl-strong" : null });
      el("rect", { x: x(clamp(c.ci_lower_week_wild, lo, hi)), width: x(clamp(c.ci_upper_week_wild, lo, hi)) - x(clamp(c.ci_lower_week_wild, lo, hi)), y: cy - 7, height: 14, rx: 7, fill: css("--accent"), opacity: 0.2 }, root);
      el("line", { x1: x(c.ci_lower), x2: x(c.ci_upper), y1: cy, y2: cy, stroke: css("--text"), "stroke-width": 3 }, root);
      el("circle", { cx: x(c.ate), cy, r: 5, fill: css("--accent") }, root);
      text(root, W - m.r + 12, cy + 4, `${fmt(c.ate)}  p ${pval(c.pvalue_week_wild)}`, { class: "mono" });
    });
  }
  drawChecks();

  // ───────────────────────── 7 · feedback ─────────────────────────
  function drawFeedback() {
    const ah1 = D.feedback.instrument_T_lag1;
    const items = [
      ["Primary DML (two-way CI)", primary.ate, primary.ate_ci_lower_twoway, primary.ate_ci_upper_twoway, css("--accent"), "assumes strict exogeneity"],
      ["Anderson–Hsiao, T in t−1", ah1.theta, ah1.ci_lower_twoway, ah1.ci_upper_twoway, css("--mid"), `invalid under feedback · F = ${ah1.first_stage_F.toFixed(0)}`],
      ["Anderson–Hsiao, T in t−2", ah2.theta, ah2.ci_lower_twoway, ah2.ci_upper_twoway, css("--late"), `valid under feedback · F = ${ah2.first_stage_F.toFixed(0)}`],
    ];
    const Wf = 640, H = 300, m = { l: 190, r: 20, t: 30, b: 50 };
    const root = svg($("#chart-feedback"), Wf, H);
    const x = scale(-25, 25, m.l, Wf - m.r);
    const g = el("g", { class: "grid" }, root);
    for (const v of ticks(-25, 25, 5)) { el("line", { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }, g); text(root, x(v), H - m.b + 18, fmt(v, 0), { "text-anchor": "middle" }); }
    el("line", { x1: x(0), x2: x(0), y1: m.t - 10, y2: H - m.b, class: "zero" }, root);
    text(root, (m.l + Wf - m.r) / 2, H - 10, "Effect of one extra self-trade (95% CI, trader × week)", { "text-anchor": "middle" });
    items.forEach(([name, est, lo, hi, col, note], i) => {
      const cy = m.t + 30 + i * 72;
      text(root, m.l - 12, cy + 2, name, { "text-anchor": "end", class: "lbl-strong" });
      text(root, m.l - 12, cy + 18, note, { "text-anchor": "end", "font-size": 11, fill: css("--faint") });
      const bar = el("line", { x1: x(est), x2: x(est), y1: cy, y2: cy, stroke: col, "stroke-width": 10, "stroke-linecap": "round", opacity: 0.45 }, root);
      el("circle", { cx: x(est), cy, r: 7, fill: col, stroke: css("--card"), "stroke-width": 2 }, root);
      text(root, x(est), cy - 14, `${fmtS(est)}  ${ci(lo, hi, 1)}`, { "text-anchor": "middle", class: "mono", fill: css("--text") });
      tween(900 + i * 250, (t) => { bar.setAttribute("x1", x(est + (lo - est) * t)); bar.setAttribute("x2", x(est + (hi - est) * t)); });
    });
  }
  drawFeedback();
  const L = D.lead;
  $("#feedback-side").innerHTML =
    `<div class="stat-big"><b>t = ${L.linear_fe_lead_t_week.toFixed(1)}</b>next week's self-trades predicting this week's gain (week-level inference, p = ${pval(L.linear_fe_lead_pvalue_week_wild)})</div>` +
    `<table class="tbl"><tr><th>Lead test</th><th class="num">Value</th></tr>` +
    `<tr><td>Coefficient on lead</td><td class="num">${fmt(L.linear_fe_lead_coef)}</td></tr>` +
    `<tr><td>t, trader-clustered</td><td class="num">${L.linear_fe_lead_t.toFixed(1)}</td></tr>` +
    `<tr><td>t, week-clustered</td><td class="num">${L.linear_fe_lead_t_week.toFixed(1)}</td></tr>` +
    `<tr><td>Effective weeks</td><td class="num">${L.effective_time_clusters.toFixed(1)}</td></tr></table>` +
    `<div class="callout" style="border-left-color:var(--late)"><b>Reading.</b> The feedback-robust estimate spans both signs, so the data cannot separate a causal effect from feedback. The placebo (${D.placebo.map((p) => fmt(p.ate)).join(", ")}) shows the pipeline returns a null when the effect is zero by construction; it does not test identification.</div>`;

  // ───────────────────────── verdict + footer ─────────────────────────
  $("#v-yes").innerHTML = [
    `Self-trades lost about <b>${fmt(Math.abs(early.ate), 1)}</b> units at the margin from January to mid-June 2013, robust to week-level inference.`,
    `In the final three weeks, where ${pct(shareOf(73, 75), 0)} of self-trades occur, the marginal trade was <b>break-even</b> (${fmt(late.ate)}).`,
    `Self-trading did <b>not crowd out</b> copy-trading, in gains or in number of trades.`,
  ].map((x) => `<li>${x}</li>`).join("");
  $("#v-no").innerHTML = [
    `A single, time-invariant <b>causal effect</b> of self-trading: the pooled ${fmt(primary.ate)} is borderline (p = ${pval(primary.ate_pvalue_week_wild)}) and depends on the weighting (${fmt(D.aggregates.variance_weighted)} to ${fmt(D.aggregates.equal_week)}).`,
    `Claims that some <b>types of trader</b> benefit: differences between traders are calendar timing.`,
    `<b>Policy recommendations</b>: strict exogeneity fails and the feedback-robust estimate is uninformative.`,
  ].map((x) => `<li>${x}</li>`).join("");
  const man = D.manifest;
  const last = man && man.runs[man.latest_run];
  $("#provenance").textContent = `Results from run ${last ? last.run_started : "n/a"} · package ${last ? last.package_version : "n/a"} · commit ${last ? last.git_commit : "n/a"} · bundled ${D.generated}.`;

  // initial renders + scroll reveal
  drawActivity("trades");
  drawCI("trader");
  drawDecomp("variance_weighted");
  document.querySelectorAll(".panel, .verdict").forEach((p) => p.classList.add("reveal"));
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.08 });
  document.querySelectorAll(".reveal").forEach((p) => io.observe(p));

  // re-colour charts when the system theme flips
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => location.reload());
})();
