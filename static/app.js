const LABELS = ["NAME", "ORG", "DES", "PHONE", "EMAIL", "WEB"];
const LABEL_TEXT = { NAME: "Name", ORG: "Company", DES: "Title", PHONE: "Phone", EMAIL: "E-mail", WEB: "Website" };
const SOURCE_TEXT = { ner: "model", rules: "rule", layout: "layout", llm: "LLM", user: "edited" };

const $ = (s) => document.querySelector(s);
const canvas = $("#canvas"), ctx = canvas.getContext("2d");
let state = { id: null, img: null, corners: [], drag: -1, mode: "card", text: "", pdf: null, invoice: null };

const INVOICE_FIELDS = [
  ["seller", "Seller"], ["invoice_number", "Invoice no."], ["date", "Date"], ["net", "Net"],
  ["vat_rate", "VAT rate %"], ["vat", "VAT"], ["total", "Total"], ["currency", "Currency"],
  ["vat_id", "VAT ID"], ["iban", "IBAN"],
];

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === mode)));
}

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
  const noun = { card: "Card", invoice: "Page", archive: "Page" }[state.mode];
  msg.textContent = data.found ? `${noun} found. Check the corners.` : "No document edge found. Drag the handles onto the corners.";
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
      body: JSON.stringify({ id: state.id, mode: state.mode, corners: state.corners.map((c) => [c.x, c.y]) }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Extraction failed.");
    $("#card").src = data.card;
    $("#engine").textContent = data.engine;
    ["card", "invoice", "archive"].forEach((m) => { $("#r-" + m).hidden = m !== data.mode; });
    if (data.mode === "card") {
      $("#card-cap").textContent = `${data.words} words read · ${data.engine}`;
      renderRows(data.fields);
    } else if (data.mode === "invoice") {
      $("#card-cap").textContent = data.engine;
      renderInvoice(data.invoice);
    } else {
      $("#card-cap").textContent = `${data.text.split(/\s+/).filter(Boolean).length} words read · ${data.engine}`;
      $("#arch-text").value = data.text.trim();
      state.pdf = data.pdf;
    }
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

function renderInvoice(inv) {
  state.invoice = inv;
  const rows = $("#inv-rows");
  rows.innerHTML = "";
  INVOICE_FIELDS.forEach(([key, label]) => {
    const row = document.createElement("div");
    row.className = "row ro";
    const id = "inv-" + key;
    row.innerHTML = `<label for="${id}"></label><input id="${id}">`;
    row.querySelector("label").textContent = label;
    const input = row.querySelector("input");
    input.value = inv[key] ?? "";
    input.dataset.key = key;
    rows.append(row);
  });
  const list = $("#checks");
  list.innerHTML = "";
  inv.checks.forEach((c) => {
    const li = document.createElement("li");
    li.className = c.ok ? "" : "bad";
    li.innerHTML = "<div><span></span><small></small></div>";
    li.querySelector("span").textContent = c.name;
    li.querySelector("small").textContent = c.detail;
    list.append(li);
  });
}

function invoiceValues() {
  const out = {};
  document.querySelectorAll("#inv-rows input").forEach((i) => { out[i.dataset.key] = i.value.trim() || null; });
  return out;
}

function saveText(text, name, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
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
  setMode(b.dataset.mode);
  const form = new FormData(); form.append("sample", b.dataset.sample); detect(form);
}));
document.querySelectorAll(".mode").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
$("#inv-json").onclick = () => saveText(JSON.stringify(invoiceValues(), null, 2), "invoice.json", "application/json");
$("#inv-csv").onclick = () => {
  const v = invoiceValues();
  const esc = (x) => `"${String(x ?? "").replace(/"/g, '""')}"`;
  saveText(Object.keys(v).map(esc).join(",") + "\n" + Object.values(v).map(esc).join(",") + "\n", "invoice.csv", "text/csv");
};
$("#arch-txt").onclick = () => saveText($("#arch-text").value, "document.txt", "text/plain");
$("#arch-pdf").onclick = async () => {
  if (!state.pdf) return;
  const res = await fetch(state.pdf);
  if (!res.ok) return showError("This PDF has expired. Scan the page again.");
  const url = URL.createObjectURL(await res.blob());
  const a = Object.assign(document.createElement("a"), { href: url, download: "document.pdf" });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
};
$("#run").onclick = extract;
$("#back1").onclick = () => step(1);
$("#back2").onclick = () => step(1);
$("#add").onclick = () => $("#rows").append(rowEl({ label: "NAME", value: "", source: "user" }));
document.querySelectorAll(".export button").forEach((b) => (b.onclick = () => exportAs(b.dataset.fmt)));
