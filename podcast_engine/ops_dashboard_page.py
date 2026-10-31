"""Read-only PodcastOps episode status dashboard HTML (TASK-142)."""

OPS_DASHBOARD_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PodcastOps | Episode status</title>
<style>
:root{font:16px/1.5 system-ui,sans-serif;color:#19273a;background:#f6f8fb}
body{margin:0}main{max-width:1240px;margin:auto;padding:2rem 1rem 4rem}
a{color:#244f95}h1{font-size:1.7rem;margin:.6rem 0}
.muted{color:#526176}.toolbar{display:flex;gap:.7rem;flex-wrap:wrap;align-items:center;margin:1.4rem 0}
button,input,select{font:inherit;border-radius:8px;border:1px solid #b8c6d6;padding:.55rem .75rem;background:#fff;color:#19273a}
button{cursor:pointer}.panel{background:#fff;border:1px solid #dae3ed;border-radius:12px;padding:1.2rem}
table{width:100%;border-collapse:collapse;text-align:left}th,td{padding:.75rem .6rem;vertical-align:top;border-bottom:1px solid #e3e8ef}
th{font-size:.82rem;color:#516178;white-space:nowrap}
.title{display:block;font-weight:650}.sub{display:block;color:#516178;font-size:.85rem}
.flag{display:inline-block;background:#e9eef5;border-radius:6px;padding:.1rem .4rem;font-size:.85rem}
.flag.done{background:#e5f4ea;color:#225e35}.flag.warn{background:#fff0d8;color:#7b440f}
.flag.bad{background:#fbe7e7;color:#8b2323}.flag.pending{background:#f0f1f6}
#message{margin:.8rem 0;min-height:1.5rem}.scroll{overflow-x:auto}
@media(max-width:780px){table{min-width:930px}.toolbar input{flex:1;min-width:160px}}
</style>
</head>
<body><main>
<a href="/">← Human Review</a>
<h1>PodcastOps — episode status</h1>
<p class="muted">Every canonical episode, including completed ones. The final delivery indicator is green only when the Mac has reported a byte-matched copy of the current cloud summary. This proves the local Vault copy at the recorded time, not that iCloud uploaded it or that the local file is still present. <strong>Worker audit "started" does not prove that Cloud Run is still running.</strong> The page cannot start jobs, settle cards, or run AI.</p>
<div class="toolbar">
<label>Filter <select id="filter"><option value="all">All episodes</option><option value="review">Awaiting review</option><option value="inflight">Recompile audit open</option><option value="sync">Awaiting Mac Vault Sync</option><option value="unavailable">Unavailable evidence</option></select></label>
<label>Find <input id="search" type="search" placeholder="Podcast or episode"></label>
<button id="refresh" type="button">Refresh</button>
<span class="muted" id="updated"></span>
</div>
<div id="message" role="status" aria-live="polite">Loading episodes…</div>
<section class="panel scroll">
<table><thead><tr>
<th>Episode</th><th>Apple</th><th>Whisper</th><th>Compiler</th>
<th>Human Review</th><th>Pending groups</th><th>Recompile audit</th><th>Summary</th><th>Mac Vault Sync</th><th>Delivered</th>
</tr></thead><tbody id="rows"></tbody></table>
</section>
</main>
<script>
(function(){
"use strict";
let entries=[],loading=false;
const q=id=>document.getElementById(id);
const statusLabel=value=>{
  if(value==="not_recorded")return "No audit recorded";
  if(value==="no_cards")return "No review cards";
  if(value==="reviewed")return "All cards decided";
  if(value==="unavailable"||value==="unverified")return "Evidence unavailable";
  if(value==="confirmed")return "Mac copy confirmed";
  if(value==="unconfirmed")return "No Mac confirmation";
  if(value==="not_ready")return "Waiting for summary";
  if(value==="outdated")return "Mac copy needs update";
  if(value==="synced_to_mac")return "Delivered to Mac Vault";
  if(value==="awaiting_vault_sync")return "Awaiting Mac sync";
  if(value==="awaiting_summary")return "Awaiting summary";
  if(value==="processing")return "Processing";
  if(value==="duplicate")return "Skipped: duplicate";
  if(value==="not_applicable")return "Not needed";
  if(value==="starting"||value==="started")return value+" (audit only)";
  return String(value||"unknown").replaceAll("_"," ");
};
const flagType=value=>{
  if(["completed","ready","reviewed","confirmed","synced_to_mac"].includes(value))return "done";
  if(["unavailable","unverified","unknown","failed","execution_failed"].includes(value))return "bad";
  if(["review_required","pending","starting","started","unconfirmed","outdated","awaiting_vault_sync","awaiting_summary"].includes(value))return "warn";
  return "pending";
};
function cell(tr, value,flag){
  const td=document.createElement("td");
  if(flag){const badge=document.createElement("span");badge.className="flag "+flagType(value);badge.textContent=statusLabel(value);td.append(badge);}
  else td.textContent=value==null?"Not available":String(value);
  tr.append(td);return td;
}
function matchesFilter(entry){
  const filter=q("filter").value,query=q("search").value.trim().toLowerCase();
  if(query&&!(entry.podcast+" "+entry.title).toLowerCase().includes(query))return false;
  if(filter==="review")return entry.compiler==="review_required"||entry.review.state==="pending";
  if(filter==="inflight")return ["starting","started","unknown"].includes(entry.recompile.status);
  if(filter==="sync")return entry.delivery==="awaiting_vault_sync";
  if(filter==="unavailable")return ["unavailable","unknown"].includes(entry.review.state)||entry.recompile.status==="unavailable"||entry.vault_sync.status==="unavailable";
  return true;
}
function render(){
  const body=q("rows");body.replaceChildren();let shown=0;
  for(const row of entries){
    if(!matchesFilter(row))continue;shown++;
    const tr=document.createElement("tr");
    const first=cell(tr,"",false);first.textContent="";
    const title=document.createElement("span");title.className="title";title.textContent=row.title;first.append(title);
    const podcast=document.createElement("span");podcast.className="sub";podcast.textContent=row.podcast||"Unknown podcast";first.append(podcast);
    const key=document.createElement("span");key.className="sub";key.textContent=row.episode_key;first.append(key);
    if(row.duplicate_of){const dup=document.createElement("span");dup.className="sub";dup.textContent="Same conversation as "+(row.duplicate_of.podcast||"another show")+": "+(row.duplicate_of.title||row.duplicate_of.episode_key||"an earlier episode")+" — not processed again";first.append(dup);}
    cell(tr,row.apple,true);cell(tr,row.whisper,true);cell(tr,row.compiler,true);
    const review=cell(tr,row.review.state,true);
    const progress=document.createElement("span");progress.className="sub";
    progress.textContent=row.review.decided===null||row.review.pending===null?"Counts unavailable":row.review.decided+" decided · "+row.review.pending+" pending";
    review.append(progress);
    const groups=row.review.groups;
    cell(tr,groups?"Settled "+groups.settled+" · Sample "+groups.sample+" · Quick "+(groups.click_one+groups.proposal)+" · Full "+groups.full:"Not available",false);
    const worker=cell(tr,row.recompile.status,true);
    const workerTime=row.recompile.completed_at||row.recompile.updated_at;
    if(workerTime){const when=document.createElement("span");when.className="sub";when.textContent=(row.recompile.completed_at?"Completed ":"Updated ")+workerTime;worker.append(when);}
    const summary=cell(tr,row.summary,true);
    if(row.summary_ready_at){const when=document.createElement("span");when.className="sub";when.textContent="Ready "+row.summary_ready_at;summary.append(when);}
    const mac=cell(tr,row.vault_sync.status,true);
    if(row.vault_sync.confirmed_at){const when=document.createElement("span");when.className="sub";when.textContent="Confirmed "+row.vault_sync.confirmed_at;mac.append(when);}
    const done=cell(tr,row.delivery,true);
    if(row.finished_at){const when=document.createElement("span");when.className="sub";when.textContent=row.finished_at;done.append(when);}
    body.append(tr);
  }
  if(!shown){const tr=document.createElement("tr"),td=document.createElement("td");td.colSpan=10;td.textContent="No episodes match this view.";tr.append(td);body.append(tr);}
  q("message").textContent=shown+" of "+entries.length+" canonical episodes";
}
async function refresh(){
  if(loading)return;loading=true;q("refresh").disabled=true;
  try{
    const response=await fetch("/api/ops/episodes",{credentials:"same-origin",cache:"no-store"});
    if(!response.ok)throw Error("HTTP "+response.status);
    const data=await response.json();
    if(!Array.isArray(data.episodes))throw Error("Invalid response");
    entries=data.episodes;render();
    q("updated").textContent="Refreshed "+new Date().toLocaleTimeString();
  }catch(error){q("message").textContent="Status unavailable ("+error.message+"). Existing entries are retained.";}
  finally{loading=false;q("refresh").disabled=false;}
}
q("refresh").addEventListener("click",refresh);
q("search").addEventListener("input",render);
q("filter").addEventListener("change",render);
setInterval(()=>{if(!document.hidden)refresh();},60000);
refresh();
})();
</script>
</body></html>"""
