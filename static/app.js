const LABELS = ["NAME", "ORG", "DES", "PHONE", "EMAIL", "WEB"];
const LABEL_TEXT = { NAME: "Name", ORG: "Company", DES: "Title", PHONE: "Phone", EMAIL: "E-mail", WEB: "Website" };
const SOURCE_TEXT = { ner: "model", rules: "rule", layout: "layout", llm: "LLM", user: "edited" };

const $ = (s) => document.querySelector(s);
const canvas = $("#canvas"), ctx = canvas.getContext("2d");
let state = { id: null, img: null, corners: [], drag: -1 };

function showError(msg) { const e = $("#error"); e.textContent = msg; e.hidden = !msg; }
function step(n) {
  [1, 2, 3].forEach((i) => { $("#s" + i).hidden = i !== n; });
  document.querySelectorAll(".steps li").forEach((li) => {
    const i = Number(li.dataset.step);
    li.classList.toggle("on", i === n);
    li.classList.toggle("done", i < n);
  });
  showError("");
}

async function detect(form) {
  showError("");
  const res = await fetch("/api/detect", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) return showError(data.error || "Upload failed.");
  state.id = data.id;
  state.corners = data.corners.map(([x, y]) => ({ x, y }));
  const msg = $("#detect-msg");
  msg.textContent = data.found ? "Card found. Check the corners." : "No card edge found. Drag the handles onto the card's corners.";
  msg.classList.toggle("warn", !data.found);
  const img = new Image();
  img.onload = () => { state.img = img; canvas.width = img.width; canvas.height = img.height; draw(); step(2); };
  img.src = data.image;
}

function handleRadius() { return Math.max(14, canvas.width / 70); }

function draw() {
  const { img, corners } = state;
  if (!img) return;
  ctx.drawImage(img, 0, 0);
  ctx.save();
  ctx.fillStyle = "rgba(10,12,16,.45)";
  ctx.beginPath();
  ctx.rect(0, 0, canvas.width, canvas.height);
  corners.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.closePath();
  ctx.fill("evenodd");
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#D9481C";
  ctx.strokeStyle = accent;
  ctx.lineWidth = Math.max(3, canvas.width / 400);
  ctx.beginPath();
  corners.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.closePath();
  ctx.stroke();
  const r = handleRadius();
  corners.forEach((p) => {
    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = Math.max(3, r / 4); ctx.stroke();
  });
  ctx.restore();
}

function toCanvas(e) {
  const rect = canvas.getBoundingClientRect();
  return { x: (e.clientX - rect.left) * (canvas.width / rect.width), y: (e.clientY - rect.top) * (canvas.height / rect.height) };
}
canvas.addEventListener("pointerdown", (e) => {
  const p = toCanvas(e), r = handleRadius() * 2.2;
  state.drag = state.corners.findIndex((c) => Math.hypot(c.x - p.x, c.y - p.y) < r);
  if (state.drag >= 0) canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (state.drag < 0) return;
  const p = toCanvas(e);
  state.corners[state.drag] = { x: Math.min(Math.max(p.x, 0), canvas.width), y: Math.min(Math.max(p.y, 0), canvas.height) };
  draw();
});
canvas.addEventListener("pointerup", () => { state.drag = -1; });

async function extract() {
  const btn = $("#run"), stage = $("#stage");
  btn.disabled = true; btn.textContent = "Reading the card…"; stage.classList.add("busy");
  try {
    const res = await fetch("/api/extract", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: state.id, corners: state.corners.map((c) => [c.x, c.y]) }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Extraction failed.");
    $("#card").src = data.card;
    $("#card-cap").textContent = `${data.words} words read · ${data.engine}`;
    renderRows(data.fields);
    step(3);
  } catch (err) { showError(err.message); }
  finally { btn.disabled = false; btn.textContent = "Extract details"; stage.classList.remove("busy"); }
}

function rowEl(f) {
  const row = document.createElement("div");
  row.className = "row";
  const sel = document.createElement("select");
  sel.setAttribute("aria-label", "Field type");
  LABELS.forEach((l) => sel.add(new Option(LABEL_TEXT[l], l, false, l === f.label)));
  const paint = () => sel.style.setProperty("--c", `var(--${sel.value.toLowerCase()})`);
  paint();
  const input = document.createElement("input");
  input.value = f.value; input.setAttribute("aria-label", LABEL_TEXT[f.label] || "Value");
  const src = document.createElement("span");
  src.className = "src";
  src.innerHTML = `<span>${SOURCE_TEXT[f.source] || f.source}</span><button type="button" aria-label="Remove field">×</button>`;
  const mark = () => { row.dataset.source = "user"; src.firstChild.textContent = "edited"; };
  sel.onchange = () => { paint(); mark(); };
  input.oninput = mark;
  src.querySelector("button").onclick = () => row.remove();
  row.dataset.source = f.source;
  row.append(sel, input, src);
  return row;
}

function renderRows(fields) {
  const rows = $("#rows");
  rows.innerHTML = "";
  const order = (f) => LABELS.indexOf(f.label);
  [...fields].sort((a, b) => order(a) - order(b)).forEach((f) => rows.append(rowEl(f)));
}

function currentFields() {
  return [...document.querySelectorAll("#rows .row")].map((r) => ({
    label: r.querySelector("select").value, value: r.querySelector("input").value, source: r.dataset.source,
  }));
}

async function exportAs(fmt) {
  const res = await fetch(`/api/export/${fmt}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ fields: currentFields() }),
  });
  if (!res.ok) return showError("Export failed.");
  const url = URL.createObjectURL(await res.blob());
  const a = Object.assign(document.createElement("a"), { href: url, download: `contact.${fmt}` });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

$("#file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const form = new FormData(); form.append("image", file); detect(form);
  e.target.value = "";
});
const drop = $("#drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) { const form = new FormData(); form.append("image", file); detect(form); }
});
document.querySelectorAll(".sample").forEach((b) => b.addEventListener("click", () => {
  const form = new FormData(); form.append("sample", b.dataset.sample); detect(form);
}));
$("#run").onclick = extract;
$("#back1").onclick = () => step(1);
$("#back2").onclick = () => step(1);
$("#add").onclick = () => $("#rows").append(rowEl({ label: "NAME", value: "", source: "user" }));
document.querySelectorAll(".export button").forEach((b) => (b.onclick = () => exportAs(b.dataset.fmt)));
