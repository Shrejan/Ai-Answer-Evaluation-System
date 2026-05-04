"""
review_labels.py  —  OCR Label Reviewer Pro
============================================
pip install flask pandas
python review_labels.py
Open  http://localhost:5000

Shortcuts:
  ← →  (arrows)       previous / next item
  Enter                save label & stay
  Tab                  save label & go to next
  Esc                  revert field
  Ctrl+D / Cmd+D       mark item as DONE

All changes are saved instantly to labels_to_review.csv.
"""

import os, json
import pandas as pd
from flask import Flask, request, jsonify, send_file

CSV_PATH  = "./labels_to_review.csv"
PORT      = 5000

app = Flask(__name__)

# ── load / prep dataframe ─────────────────────────────────
df = pd.read_csv(CSV_PATH)

# Ensure required columns exist
for col, default in [("human_label",""),("reviewed","No"),("confidence_pct",0)]:
    if col not in df.columns:
        df[col] = default

df["human_label"]    = df["human_label"].fillna("")
df["reviewed"]       = df["reviewed"].fillna("No")
df["confidence_pct"] = pd.to_numeric(df["confidence_pct"], errors="coerce").fillna(0)

# Create a stable unique ID (image_path must be unique)
if "image_path" not in df.columns:
    raise RuntimeError("CSV must contain 'image_path' column")
uid_series = df["image_path"].astype(str)
if uid_series.duplicated().any():
    raise RuntimeError("Duplicate image_path values found – cannot use as unique ID")
df["uid"] = uid_series

# Sort by confidence (lowest first)
df = df.sort_values("confidence_pct", ascending=True).reset_index(drop=True)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>OCR Reviewer Pro</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<style>
/* ══ RESET & TOKENS ════════════════════════════════════════ */
*,::before,::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:       #0d0f14;
  --surface:  #13161e;
  --panel:    #191d28;
  --border:   #252a38;
  --accent:   #00e5c0;
  --accent2:  #ff4d6d;
  --warn:     #ffd166;
  --muted:    #4a5068;
  --text:     #e8eaf2;
  --text2:    #8890aa;
  --radius:   10px;
  --mono:     'Space Mono', monospace;
  --sans:     'Syne', sans-serif;
  --trans:    .18s cubic-bezier(.4,0,.2,1);
}

html,body{height:100%;overflow:hidden}
body{
  background:var(--bg);
  color:var(--text);
  font-family:var(--sans);
  display:flex; flex-direction:column;
}

/* ══ TOP BAR ═══════════════════════════════════════════════ */
#topbar{
  height:56px; flex-shrink:0;
  background:var(--surface);
  border-bottom:1px solid var(--border);
  display:flex; align-items:center;
  padding:0 20px; gap:16px;
  position:relative; z-index:50;
}
#logo{
  font-family:var(--sans); font-weight:800; font-size:17px;
  color:var(--accent); letter-spacing:-.5px; white-space:nowrap;
}
#logo span{color:var(--text2); font-weight:400}

.sep{width:1px;height:28px;background:var(--border);flex-shrink:0}

#progress-wrap{flex:1; min-width:120px}
#prog-bar-bg{
  height:4px; background:var(--border); border-radius:2px; overflow:hidden;
}
#prog-bar{
  height:100%; background:var(--accent);
  border-radius:2px; transition:width .4s ease;
}
#prog-label{font-size:11px;color:var(--text2);margin-top:4px;font-family:var(--mono)}

#stat-chips{display:flex;gap:8px;align-items:center}
.chip{
  font-family:var(--mono); font-size:11px;
  padding:3px 9px; border-radius:20px;
  border:1px solid var(--border); color:var(--text2);
}
.chip.c-green{border-color:#00e5c044;color:var(--accent);background:#00e5c011}
.chip.c-red  {border-color:#ff4d6d44;color:var(--accent2);background:#ff4d6d11}
.chip.c-warn {border-color:#ffd16644;color:var(--warn);background:#ffd16611}

#topbar-actions{display:flex;gap:8px;align-items:center}
.btn{
  font-family:var(--sans); font-size:12px; font-weight:600;
  padding:6px 14px; border-radius:6px; cursor:pointer;
  border:1px solid var(--border); background:transparent;
  color:var(--text2); transition:var(--trans); white-space:nowrap;
}
.btn:hover{border-color:var(--accent);color:var(--accent);background:#00e5c011}
.btn.primary{
  background:var(--accent); color:#000; border-color:var(--accent);
}
.btn.primary:hover{background:#00c8a8}
.btn.danger{border-color:var(--accent2);color:var(--accent2)}
.btn.danger:hover{background:#ff4d6d22}

/* ══ LAYOUT ════════════════════════════════════════════════ */
#body{
  flex:1; display:flex; overflow:hidden;
}

/* ── SIDEBAR ── */
#sidebar{
  width:260px; flex-shrink:0;
  background:var(--surface);
  border-right:1px solid var(--border);
  display:flex; flex-direction:column;
  overflow:hidden;
}

#sidebar-head{
  padding:14px 16px;
  border-bottom:1px solid var(--border);
  display:flex; flex-direction:column; gap:8px;
}
#sidebar-head label{font-size:11px;color:var(--text2);font-family:var(--mono)}

#search{
  width:100%; background:var(--panel); border:1px solid var(--border);
  color:var(--text); font-family:var(--mono); font-size:12px;
  padding:7px 10px; border-radius:6px; outline:none;
  transition:var(--trans);
}
#search:focus{border-color:var(--accent)}
#search::placeholder{color:var(--muted)}

.filter-row{display:flex;gap:6px}
.filter-btn{
  flex:1; font-family:var(--mono); font-size:10px; font-weight:700;
  padding:5px 4px; border-radius:5px; cursor:pointer;
  border:1px solid var(--border); background:transparent;
  color:var(--text2); transition:var(--trans); text-align:center;
}
.filter-btn:hover{border-color:var(--accent);color:var(--accent)}
.filter-btn.active{background:var(--accent);color:#000;border-color:var(--accent)}

#item-list{
  flex:1; overflow-y:auto; padding:6px;
}
#item-list::-webkit-scrollbar{width:4px}
#item-list::-webkit-scrollbar-track{background:transparent}
#item-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

.list-item{
  display:flex; align-items:center; gap:8px;
  padding:8px 10px; border-radius:7px; cursor:pointer;
  border:1px solid transparent; margin-bottom:3px;
  transition:var(--trans);
}
.list-item:hover{background:var(--panel);border-color:var(--border)}
.list-item.active{background:var(--panel);border-color:var(--accent)!important}
.list-item.done{opacity:.5}

.li-thumb{
  width:44px; height:22px; flex-shrink:0;
  background:white; border-radius:3px; overflow:hidden;
  display:flex; align-items:center; justify-content:center;
}
.li-thumb img{max-width:100%;max-height:100%;object-fit:contain}

.li-info{flex:1;min-width:0}
.li-name{font-size:11px;color:var(--text);white-space:nowrap;
          overflow:hidden;text-overflow:ellipsis;font-family:var(--mono)}
.li-sub{font-size:10px;color:var(--text2);margin-top:1px;
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}

.li-badge{
  flex-shrink:0; font-family:var(--mono); font-size:9px; font-weight:700;
  width:32px; text-align:center; padding:2px 0; border-radius:4px;
}
.badge-hi{background:#00e5c022;color:var(--accent)}
.badge-md{background:#ffd16622;color:var(--warn)}
.badge-lo{background:#ff4d6d22;color:var(--accent2)}
.badge-ok{background:#00e5c033;color:var(--accent);border:1px solid var(--accent)44}

/* ── MAIN EDITOR ── */
#editor{
  flex:1; display:flex; flex-direction:column;
  overflow:hidden; background:var(--bg);
}

#editor-header{
  padding:16px 24px 0;
  display:flex; align-items:center; gap:12px; flex-shrink:0;
}
#editor-title{font-size:13px;font-family:var(--mono);color:var(--text2)}
#editor-title strong{color:var(--text)}

.nav-btn{
  font-family:var(--mono); font-size:12px;
  padding:5px 12px; border-radius:5px; cursor:pointer;
  border:1px solid var(--border); background:transparent;
  color:var(--text2); transition:var(--trans);
}
.nav-btn:hover{border-color:var(--accent);color:var(--accent)}
#shortcut-hint{margin-left:auto;font-family:var(--mono);font-size:10px;color:var(--muted)}

/* Image panel */
#img-panel{
  flex-shrink:0; padding:16px 24px;
  display:flex; align-items:flex-start; gap:20px;
}
#img-frame{
  flex:1; background:white;
  border-radius:var(--radius);
  border:2px solid var(--border);
  overflow:hidden;
  display:flex; align-items:center; justify-content:center;
  min-height:120px; max-height:240px;
  position:relative; cursor:zoom-in;
  transition:border-color var(--trans);
}
#img-frame:hover{border-color:var(--accent)}
#main-img{
  max-width:100%; max-height:240px;
  object-fit:contain; display:block;
  transition:transform var(--trans);
}
#img-frame.zoomed{cursor:zoom-out; max-height:340px}
#img-frame.zoomed #main-img{max-height:340px}

#img-meta{
  width:180px; flex-shrink:0;
  display:flex; flex-direction:column; gap:10px;
}
.meta-card{
  background:var(--panel); border:1px solid var(--border);
  border-radius:var(--radius); padding:12px;
}
.meta-card h4{font-size:10px;color:var(--text2);font-family:var(--mono);
               text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
.meta-row{display:flex;justify-content:space-between;align-items:center;
           margin-bottom:5px; font-size:12px}
.meta-row:last-child{margin-bottom:0}
.meta-key{color:var(--text2);font-family:var(--mono)}
.meta-val{color:var(--text);font-weight:600;font-family:var(--mono)}

.conf-bar-bg{height:6px;background:var(--border);border-radius:3px;margin-top:6px;overflow:hidden}
.conf-bar-fill{height:100%;border-radius:3px;transition:width .4s ease}

/* KV editor */
#kv-panel{
  flex:1; overflow:hidden;
  padding:0 24px 16px;
  display:flex; flex-direction:column; gap:0;
}

#kv-header{
  display:grid;
  grid-template-columns: 180px 1fr 100px;
  gap:1px; background:var(--border);
  border-radius:var(--radius) var(--radius) 0 0;
  overflow:hidden; flex-shrink:0;
}
.kv-head-cell{
  background:var(--panel);
  padding:9px 14px;
  font-size:10px; font-family:var(--mono);
  text-transform:uppercase; letter-spacing:.1em;
  color:var(--text2); font-weight:700;
}

#kv-body{
  flex:1; overflow-y:auto;
  border:1px solid var(--border); border-top:none;
  border-radius:0 0 var(--radius) var(--radius);
  background:var(--surface);
}
#kv-body::-webkit-scrollbar{width:4px}
#kv-body::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

.kv-row{
  display:grid;
  grid-template-columns: 180px 1fr 100px;
  gap:1px; background:var(--border);
  border-bottom:1px solid var(--border);
  transition:background var(--trans);
}
.kv-row:last-child{border-bottom:none}
.kv-row:hover{filter:brightness(1.08)}

.kv-key{
  background:var(--panel);
  padding:11px 14px;
  font-size:12px; font-family:var(--mono);
  color:var(--text2);
  display:flex; align-items:center;
}
.kv-val{
  background:var(--surface);
  padding:6px 10px;
  display:flex; align-items:center;
}
.kv-actions{
  background:var(--panel);
  padding:6px 10px;
  display:flex; align-items:center; gap:6px;
}

/* The editable text field */
.txt-field{
  width:100%; background:transparent;
  border:1px solid transparent;
  color:var(--text); font-family:var(--mono); font-size:13px;
  padding:6px 8px; border-radius:5px; outline:none;
  transition:var(--trans); resize:none; line-height:1.5;
}
.txt-field:hover{border-color:var(--border);background:var(--panel)}
.txt-field:focus{border-color:var(--accent);background:var(--panel)}
.txt-field.readonly{color:var(--text2);cursor:default}
.txt-field.readonly:hover,.txt-field.readonly:focus{
  border-color:transparent;background:transparent}

.save-mini{
  font-size:10px; font-family:var(--mono); font-weight:700;
  padding:4px 10px; border-radius:4px; cursor:pointer;
  border:1px solid var(--accent); color:var(--accent);
  background:transparent; transition:var(--trans); white-space:nowrap;
}
.save-mini:hover{background:var(--accent);color:#000}
.save-mini.saved{
  border-color:var(--muted); color:var(--muted); cursor:default;
}

/* ── BOTTOM STATUS BAR ── */
#statusbar{
  height:30px; flex-shrink:0;
  background:var(--surface);
  border-top:1px solid var(--border);
  display:flex; align-items:center;
  padding:0 20px; gap:16px;
  font-family:var(--mono); font-size:10px; color:var(--muted);
}
#save-status{margin-left:auto;color:var(--accent)}

/* ── LIGHTBOX ── */
#lightbox{
  display:none; position:fixed; inset:0; z-index:200;
  background:#000c; align-items:center; justify-content:center;
}
#lightbox.open{display:flex}
#lb-img{max-width:90vw;max-height:90vh;object-fit:contain;border-radius:8px}

/* ── TOAST ── */
#toast{
  position:fixed; bottom:44px; right:20px; z-index:300;
  background:var(--accent); color:#000;
  font-family:var(--mono); font-size:12px; font-weight:700;
  padding:9px 18px; border-radius:6px;
  opacity:0; transform:translateY(8px);
  transition:opacity .2s, transform .2s; pointer-events:none;
}
#toast.show{opacity:1;transform:translateY(0)}

/* ── EMPTY STATE ── */
#empty{
  display:none; flex:1; align-items:center; justify-content:center;
  flex-direction:column; gap:12px; color:var(--muted);
  font-family:var(--mono);
}
#empty svg{opacity:.3}
</style>
</head>
<body>

<!-- TOP BAR -->
<div id="topbar">
  <div id="logo">OCR<span>Review</span></div>
  <div class="sep"></div>
  <div id="progress-wrap">
    <div id="prog-bar-bg"><div id="prog-bar" style="width:0%"></div></div>
    <div id="prog-label">Loading…</div>
  </div>
  <div class="sep"></div>
  <div id="stat-chips">
    <div class="chip c-green" id="chip-done">✓ 0</div>
    <div class="chip c-warn"  id="chip-pend">⏳ 0</div>
    <div class="chip c-red"   id="chip-low">⚠ 0</div>
  </div>
  <div class="sep"></div>
  <div id="topbar-actions">
    <button class="btn primary" onclick="exportCSV()">⬇ Export CSV</button>
  </div>
</div>

<!-- BODY -->
<div id="body">

  <!-- SIDEBAR -->
  <div id="sidebar">
    <div id="sidebar-head">
      <label>SEARCH / FILTER</label>
      <input id="search" type="text" placeholder="Search labels…" oninput="renderList()"/>
      <div class="filter-row">
        <button class="filter-btn active" data-f="all"       onclick="setFilter(this)">ALL</button>
        <button class="filter-btn"        data-f="low"       onclick="setFilter(this)">LOW</button>
        <button class="filter-btn"        data-f="pending"   onclick="setFilter(this)">PENDING</button>
        <button class="filter-btn"        data-f="done"      onclick="setFilter(this)">DONE</button>
      </div>
    </div>
    <div id="item-list"></div>
  </div>

  <!-- EDITOR -->
  <div id="editor">
    <div id="editor-header">
      <div id="editor-title">Select an item from the list</div>
      <button class="nav-btn" onclick="navigate(-1)">← Prev</button>
      <button class="nav-btn" onclick="navigate(1)">Next →</button>
      <div id="shortcut-hint">Ctrl+D = Mark Done · Enter = Save · Tab = Save+Next · Esc = Revert</div>
    </div>

    <div id="img-panel">
      <div id="img-frame" onclick="openLightbox()">
        <img id="main-img" src="" alt=""/>
      </div>
      <div id="img-meta">
        <div class="meta-card">
          <h4>Confidence</h4>
          <div id="conf-display" style="font-size:22px;font-family:var(--mono);font-weight:700;color:var(--accent)">—</div>
          <div class="conf-bar-bg"><div class="conf-bar-fill" id="conf-fill" style="width:0%;background:var(--accent)"></div></div>
        </div>
        <div class="meta-card">
          <h4>File Info</h4>
          <div class="meta-row"><span class="meta-key">File</span><span class="meta-val" id="m-file">—</span></div>
          <div class="meta-row"><span class="meta-key">Status</span><span class="meta-val" id="m-status">—</span></div>
          <div class="meta-row"><span class="meta-key">Changed</span><span class="meta-val" id="m-changed">—</span></div>
        </div>
        <div class="meta-card">
          <h4>Actions</h4>
          <button class="btn" style="width:100%;margin-bottom:6px" onclick="markDone()">✓ Mark Done</button>
          <button class="btn danger" style="width:100%" onclick="revertCurrent()">↺ Revert</button>
        </div>
      </div>
    </div>

    <!-- KV TABLE -->
    <div id="kv-panel">
      <div id="kv-header">
        <div class="kv-head-cell">Field</div>
        <div class="kv-head-cell">Value</div>
        <div class="kv-head-cell">Actions</div>
      </div>
      <div id="kv-body">
        <!-- rows injected by JS -->
      </div>
    </div>

    <div id="empty">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>
      </svg>
      No items match the filter
    </div>
  </div>
</div>

<!-- STATUS BAR -->
<div id="statusbar">
  <span>⌨ Ctrl+D = Mark Done · Enter/Tab to save · Esc = revert · ←→ navigate</span>
  <span id="save-status">All changes saved directly to CSV</span>
</div>

<!-- LIGHTBOX -->
<div id="lightbox" onclick="closeLightbox()">
  <img id="lb-img" src="" alt=""/>
</div>

<!-- TOAST -->
<div id="toast"></div>

<script>
// ═══════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════
let data        = [];
let filtered    = [];
let currentIdx  = null;   // index into filtered[]
let activeFilter = 'all';
let pendingEdit  = null;

// ═══════════════════════════════════════════════════════
// BOOT
// ═══════════════════════════════════════════════════════
async function boot(){
  const res = await fetch('/api/data');
  data = await res.json();
  applyFilter();
  updateStats();
  if(filtered.length > 0) selectItem(0);
}

// ═══════════════════════════════════════════════════════
// FILTER & SEARCH
// ═══════════════════════════════════════════════════════
function setFilter(btn){
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  activeFilter = btn.dataset.f;
  applyFilter();
}

function applyFilter(){
  const q = document.getElementById('search').value.toLowerCase();
  filtered = data.filter(r=>{
    const matchQ = !q ||
      (r.auto_label||'').toLowerCase().includes(q) ||
      (r.human_label||'').toLowerCase().includes(q) ||
      (r.image_filename||'').toLowerCase().includes(q);
    if(!matchQ) return false;
    if(activeFilter==='low')     return r.confidence_pct < 75;
    if(activeFilter==='pending') return r.reviewed !== 'Yes';
    if(activeFilter==='done')    return r.reviewed === 'Yes';
    return true;
  });
  renderList();
  if(currentIdx !== null && currentIdx >= filtered.length){
    currentIdx = filtered.length > 0 ? 0 : null;
  }
  if(currentIdx !== null) renderEditor(currentIdx);
}

// ═══════════════════════════════════════════════════════
// SIDEBAR LIST
// ═══════════════════════════════════════════════════════
function renderList(){
  const el = document.getElementById('item-list');
  if(filtered.length === 0){
    el.innerHTML='';
    return;
  }
  el.innerHTML = filtered.map((r,i)=>{
    const conf = r.confidence_pct;
    const badgeCls = r.reviewed==='Yes' ? 'badge-ok' :
                     conf>=75 ? 'badge-hi' : conf>=50 ? 'badge-md' : 'badge-lo';
    const badgeTxt = r.reviewed==='Yes' ? '✓' : Math.round(conf)+'%';
    const doneCls  = r.reviewed==='Yes' ? ' done':'';
    const actCls   = i===currentIdx ? ' active':'';
    const label    = (r.human_label||r.auto_label||'').substring(0,28);
    return `<div class="list-item${doneCls}${actCls}" onclick="selectItem(${i})" id="li-${i}">
      <div class="li-thumb"><img src="/image/${encodeURIComponent(r.image_path)}" loading="lazy"/></div>
      <div class="li-info">
        <div class="li-name">${esc(r.image_filename||'')}</div>
        <div class="li-sub">${esc(label)}</div>
      </div>
      <div class="li-badge ${badgeCls}">${badgeTxt}</div>
    </div>`;
  }).join('');
}

// ═══════════════════════════════════════════════════════
// EDITOR
// ═══════════════════════════════════════════════════════
function selectItem(i){
  currentIdx = i;
  document.querySelectorAll('.list-item').forEach((el,j)=>{
    el.classList.toggle('active', j===i);
  });
  const li = document.getElementById(`li-${i}`);
  if(li) li.scrollIntoView({block:'nearest', behavior:'smooth'});
  renderEditor(i);
  pendingEdit = null;
}

function renderEditor(i){
  if(i===null || i>=filtered.length){
    document.getElementById('img-panel').style.display='none';
    document.getElementById('kv-panel').style.display='none';
    document.getElementById('empty').style.display='flex';
    return;
  }
  document.getElementById('empty').style.display='none';
  document.getElementById('img-panel').style.display='flex';
  document.getElementById('kv-panel').style.display='flex';

  const r = filtered[i];
  document.getElementById('editor-title').innerHTML = `<strong>${esc(r.image_filename||'')}</strong>`;
  document.getElementById('main-img').src = `/image/${encodeURIComponent(r.image_path)}`;
  document.getElementById('conf-display').textContent = Math.round(r.confidence_pct)+'%';
  const confBar = document.getElementById('conf-fill');
  confBar.style.width = r.confidence_pct+'%';
  confBar.style.background = r.confidence_pct<50 ? 'var(--accent2)' : r.confidence_pct<75 ? 'var(--warn)' : 'var(--accent)';
  document.getElementById('m-file').textContent = r.image_filename||'—';
  document.getElementById('m-status').textContent = r.reviewed;
  document.getElementById('m-changed').textContent = (r.human_label && r.human_label!==r.auto_label) ? 'Yes' : 'No';

  const fields = [
    {key:'auto_label',  label:'OCR Text',  editable:false},
    {key:'human_label', label:'Your Label', editable:true}
  ];

  const kvBody = document.getElementById('kv-body');
  kvBody.innerHTML = fields.map(f=>{
    // Show saved human_label if exists, otherwise auto_label
    let val;
    if(f.key === 'human_label'){
      val = r.human_label ? r.human_label : r.auto_label;
    } else {
      val = r[f.key] || '';
    }
    const cls = f.editable ? '' : 'readonly';
    return `<div class="kv-row">
      <div class="kv-key">${f.label}</div>
      <div class="kv-val">
        <textarea class="txt-field ${cls}" data-field="${f.key}" rows="1"
          onfocus="autoResize(this)" oninput="autoResize(this);onFieldEdit(this)"
          onkeydown="onKVKey(event,this,${i})"
          ${f.editable?'':'readonly'}>${esc(val)}</textarea>
      </div>
      <div class="kv-actions">
        ${f.editable ? `<button class="save-mini" onclick="saveField('${f.key}',${i})">Save</button>` : ''}
      </div>
    </div>`;
  }).join('');
  autoResizeAll();
}

function autoResize(el){
  el.style.height = 'auto';
  el.style.height = el.scrollHeight + 'px';
}
function autoResizeAll(){
  document.querySelectorAll('#kv-body textarea').forEach(el=>autoResize(el));
}

function onFieldEdit(el){
  const field = el.dataset.field;
  pendingEdit = { field, value: el.value };
}

function onKVKey(e, el, idx){
  // Ctrl+D / Cmd+D → Mark as Done
  if(e.key === 'd' && (e.ctrlKey || e.metaKey)){
    e.preventDefault();
    markDone();
    return;
  }
  if(e.key==='Enter' && !e.shiftKey){
    e.preventDefault();
    el.blur();
    saveCurrentItem(idx);
  } else if(e.key==='Tab' && !e.shiftKey){
    e.preventDefault();
    el.blur();
    saveCurrentItem(idx).then(()=>navigate(1));
  } else if(e.key==='Escape'){
    e.preventDefault();
    el.blur();
    revertCurrentItemValue(idx);
  }
}

async function saveCurrentItem(idx){
  if(idx===null || idx>=filtered.length) return;
  const r = filtered[idx];
  const humanLabelEl = document.querySelector(`#kv-body textarea[data-field="human_label"]`);
  const newHuman = humanLabelEl ? humanLabelEl.value : '';
  if(newHuman === r.human_label) return;   // no change

  r.human_label = newHuman;
  await fetch('/api/save', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ uid: r.uid, human_label: newHuman, reviewed: r.reviewed })
  });
  updateStats();
  renderList();
  toast('Label saved to CSV');
  pendingEdit = null;
}

async function saveField(field, idx){
  if(field==='human_label'){
    await saveCurrentItem(idx);
  }
}

async function markDone(){
  if(currentIdx===null) return;
  const r = filtered[currentIdx];
  r.reviewed = 'Yes';
  await fetch('/api/save', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ uid: r.uid, human_label: r.human_label, reviewed: 'Yes' })
  });
  updateStats();
  renderList();
  renderEditor(currentIdx);
  toast('Marked as done – saved to CSV');
}

async function revertCurrent(){
  if(currentIdx===null) return;
  const r = filtered[currentIdx];
  await fetch('/api/revert', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ uid: r.uid })
  });
  const res = await fetch('/api/data');
  data = await res.json();
  applyFilter();
  if(currentIdx >= filtered.length) currentIdx = filtered.length-1;
  renderEditor(currentIdx);
  toast('Reverted – CSV updated');
}

function revertCurrentItemValue(idx){
  const r = filtered[idx];
  const textarea = document.querySelector(`#kv-body textarea[data-field="human_label"]`);
  if(textarea) {
    // If no saved human_label, revert to empty (or auto_label – but we want to show the saved value only)
    textarea.value = r.human_label || '';
    autoResize(textarea);
  }
  pendingEdit = null;
}

function navigate(delta){
  if(filtered.length===0) return;
  const newIdx = (currentIdx===null) ? 0 : Math.min(Math.max(currentIdx+delta,0), filtered.length-1);
  if(newIdx!==currentIdx) selectItem(newIdx);
}

function exportCSV(){
  window.open('/api/export_csv', '_blank');
}

function openLightbox(){
  if(!filtered[currentIdx]) return;
  document.getElementById('lb-img').src = document.getElementById('main-img').src;
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox(){
  document.getElementById('lightbox').classList.remove('open');
}

let toastTimer;
function toast(msg){
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>el.classList.remove('show'), 2000);
}

function esc(str){
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function updateStats(){
  const total = data.length;
  const done = data.filter(r=>r.reviewed==='Yes').length;
  const low  = data.filter(r=>r.reviewed!=='Yes' && r.confidence_pct<75).length;
  const pending = total - done;
  document.getElementById('prog-bar').style.width = total? (done/total*100)+'%' : '0%';
  document.getElementById('prog-label').textContent = `${done} / ${total} reviewed`;
  document.getElementById('chip-done').textContent = `✓ ${done}`;
  document.getElementById('chip-pend').textContent = `⏳ ${pending}`;
  document.getElementById('chip-low').textContent = `⚠ ${low}`;
}

// Global shortcuts (outside textarea)
document.addEventListener('keydown', e => {
  // Ctrl+D / Cmd+D anywhere → Mark Done
  if(e.key === 'd' && (e.ctrlKey || e.metaKey)){
    e.preventDefault();
    markDone();
    return;
  }
  // Do not intercept if editing in a text field (except for Ctrl+D which is already handled)
  if(e.target.tagName==='TEXTAREA' || e.target.tagName==='INPUT') return;

  if(e.key==='ArrowLeft')  { e.preventDefault(); navigate(-1); }
  if(e.key==='ArrowRight') { e.preventDefault(); navigate(1); }
});

boot();
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════
@app.route("/")
def index():
    return HTML

@app.route("/api/data")
def api_data():
    """Return all rows as JSON (sorted by confidence)."""
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/save", methods=["POST"])
def api_save():
    """Save human_label / reviewed for a row (identified by uid = image_path)."""
    body = request.get_json()
    uid  = body["uid"]
    mask = df["uid"] == uid
    if mask.any():
        df.loc[mask, "human_label"] = body.get("human_label", "")
        df.loc[mask, "reviewed"]    = body.get("reviewed", "No")
        _save_csv()
    return jsonify({"ok": True})

@app.route("/api/revert", methods=["POST"])
def api_revert():
    """Revert a row to auto_label and mark as not reviewed."""
    body = request.get_json()
    uid  = body["uid"]
    mask = df["uid"] == uid
    if mask.any():
        df.loc[mask, "human_label"] = ""
        df.loc[mask, "reviewed"]    = "No"
        _save_csv()
    return jsonify({"ok": True})

@app.route("/api/export_csv")
def api_export():
    """Download the full CSV (including uid)."""
    out = df.copy()
    out.to_csv("labels_reviewed_export.csv", index=False)
    return send_file("labels_reviewed_export.csv", as_attachment=True)

@app.route("/image/<path:img_path>")
def serve_image(img_path):
    """Serve an image file safely from the current directory."""
    safe_path = os.path.normpath(img_path)
    if not os.path.isfile(safe_path):
        return "Image not found", 404
    return send_file(safe_path)

# ═══════════════════════════════════════════════════════════
def _save_csv():
    """Overwrite the original CSV with the current state."""
    df.to_csv(CSV_PATH, index=False)

# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(port=PORT, debug=True)