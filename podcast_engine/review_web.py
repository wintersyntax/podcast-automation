"""Local-only Flask UI for conservative human transcript review."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
import os
from typing import cast

import requests
from werkzeug.exceptions import HTTPException

from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_file,
    session,
    url_for,
)

from compiler.assisted_review import derive_assisted_state
from compiler.third_asr_window import anchored_third_asr_window

from .ai_budget import (
    BudgetAdmissionError,
    BudgetConcurrencyError,
    BudgetIntegrityError,
    DownstreamReserveError,
    budget_summary,
    release_budget_attempt_pre_send,
)
from .human_review import (
    PrepareSessionConflict,
    PrepareSessionError,
    ThirdAsrInFlight,
    advance_prepare_session_item,
    begin_assisted_preparation,
    ensure_audio_clip,
    ensure_third_asr,
    load_prepare_session,
    load_review_record,
    pending_review_items,
    preview_custom_edit_details,
    recompile_status_for_record,
    record_assisted_human_decision_batch,
    record_human_decision,
    record_human_decision_batch,
    request_worker_recompile,
)
from .review_audio import clip_window
from .fingerprint_types import ReviewFingerprintFields, SourceFingerprint
from .storage import get_bucket, load_episodes
from .knowledge.summary_review_storage import (
    load_summary_review_stalled_diagnostics,
)
from .knowledge.tags import KnowledgeAgentStatus, TagRegistry


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover"><title>Podcast Human Review</title>
<style>
:root{color:#172033;background:#f6f8fb;font:16px/1.5 system-ui,sans-serif}body{margin:0}.shell{max-width:1160px;margin:auto;padding:2rem 1rem 3rem}h1,h2,h3,p{margin-top:0}.muted,.secondary{color:#536176}.card,.panel{background:#fff;border:1px solid #d6dde8;border-radius:12px;padding:1.25rem;margin-top:1rem;box-shadow:0 1px 2px #1720330a}.panel{box-shadow:none}.meta{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center}.tag{background:#edf1f7;border-radius:999px;padding:.15rem .55rem;font-size:.875rem}.sources,.choices{display:grid;gap:.75rem;grid-template-columns:repeat(2,minmax(0,1fr))}.source{border-left:4px solid #395b9a}.source.whisper{border-left-color:#0d7b65}.context,.exact{white-space:pre-wrap;background:#f6f8fb;border-radius:7px;padding:.75rem}.exact{font-family:ui-monospace,SFMono-Regular,monospace}.hidden{display:none}.error{color:#a01625;margin:.75rem 0}.audio-error{color:#a01625;font-size:.9rem}button,input,select,textarea{font:inherit}button{cursor:pointer;border:1px solid #aeb9c9;border-radius:7px;padding:.5rem .8rem;background:#fff}button.primary{background:#244f95;color:#fff;border-color:#244f95}button.external{background:#eef7f5;border-color:#5c9e90}button.subtle{color:#536176}button.selected{outline:3px solid #9db7e5}button:disabled{cursor:wait;opacity:.6}textarea,input{box-sizing:border-box;width:100%;padding:.5rem;border:1px solid #aeb9c9;border-radius:7px}textarea{min-height:7rem}audio{width:100%}.section-title{margin:.25rem 0 .5rem;font-size:1rem}@media(max-width:650px){.shell{padding:1rem .75rem}.sources,.choices{grid-template-columns:1fr}}



/* Human Review Mobile Progress Polish */
@media(max-width:520px){
  .meta > b:first-child{
    width:100%;
    margin-bottom:.15rem;
  }

  .meta > .tag:first-of-type{
    display:none;
  }

  .meta{
    gap:.35rem;
  }
}

/* Human Review Source Blocks V1 */
.sources{
  gap:1rem;
}

.source{
  border-left:4px solid #395b9a;
  padding-left:.85rem;
  min-width:0;
}

.source.whisper{
  border-left-color:#0d7b65;
}

.source > b{
  display:block;
  margin:0 0 .4rem;
  line-height:1.2;
}

.source .exact,
.source .context{
  margin:0;
  border-radius:8px;
}

.panel > .sources{
  margin-top:.75rem;
}

@media(max-width:820px){
  .sources{
    gap:1rem;
  }

  .source{
    padding-left:.75rem;
  }

  .source > b{
    margin-bottom:.35rem;
  }
}

@media(max-width:520px){
  .source{
    padding-left:.65rem;
  }

  .source .exact,
  .source .context{
    padding:.75rem;
  }
}

/* Human Review Mobile V1 */
html{box-sizing:border-box;-webkit-text-size-adjust:100%}
*,*:before,*:after{box-sizing:inherit}
body{overflow-x:hidden}
#episode-nav{margin:.9rem 0 1rem}
.episode-queue-head{display:flex;justify-content:space-between;gap:1rem;align-items:baseline;margin-bottom:.65rem}
.episode-queue-head h2{margin:0}
.episode-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.75rem}
.episode-card{display:block;width:100%;min-height:118px;text-align:left;background:#fff;border:1px solid #d6dde8;border-radius:12px;padding:1rem;box-shadow:0 1px 2px #1720330a}
.episode-card:hover{border-color:#9db0cc}
.episode-card.active{border-color:#244f95;box-shadow:0 0 0 2px #244f9522}
.episode-podcast{display:block;color:#536176;font-size:.875rem;font-weight:650;margin-bottom:.25rem}
.episode-title{display:block;color:#172033;line-height:1.35}
.episode-footer{display:flex;justify-content:space-between;gap:.75rem;align-items:center;margin-top:.8rem;color:#536176;font-size:.875rem}
.episode-current{display:block;background:#fff;border:1px solid #d6dde8;border-radius:12px;padding:1.15rem 1.25rem;box-shadow:0 1px 2px #1720330a}
.episode-current .episode-title{margin-top:.2rem}
.episode-reason{display:block;color:#536176;margin-top:.35rem;font-size:.95rem}
.exact,.context{overflow-wrap:anywhere;word-break:break-word}
button{min-height:44px;touch-action:manipulation}

@media(max-width:820px){
  .shell{
    max-width:none;
    padding:1rem .8rem calc(2rem + env(safe-area-inset-bottom));
  }
  h1{
    font-size:1.65rem;
    line-height:1.2;
    margin-bottom:.6rem;
  }
  h2{font-size:1.3rem}
  h3{font-size:1.05rem}
  .card,.panel{
    padding:1rem;
    border-radius:10px;
  }
  .sources{
    grid-template-columns:1fr;
  }
  .episode-grid{
    grid-template-columns:1fr;
  }
  .episode-current{
    align-items:flex-start;
  }
  .choices{
    grid-template-columns:repeat(2,minmax(0,1fr));
  }
  .choices button{
    width:100%;
    min-height:48px;
  }
  audio{
    width:100%;
    min-height:44px;
  }
  .exact,.context{
    padding:.7rem;
  }
}

@media(max-width:520px){
  .shell{
    padding:.8rem .65rem calc(1.5rem + env(safe-area-inset-bottom));
  }
  body.has-selection .shell{
    padding-bottom:calc(6rem + env(safe-area-inset-bottom));
  }
  h1{
    font-size:1.45rem;
  }
  .card,.panel{
    margin-top:.7rem;
    padding:.85rem;
  }
  .meta{
    gap:.35rem;
  }
  .tag{
    font-size:.8rem;
  }
  .choices{
    grid-template-columns:1fr;
    gap:.55rem;
  }
  button{
    width:100%;
    min-height:48px;
  }
  textarea,input,select{
    font-size:16px;
  }
  #confirm-selection{
    width:100%;
    font-weight:700;
  }
  body.has-selection #confirm-selection{
    position:fixed;
    z-index:20;
    left:.65rem;
    right:.65rem;
    bottom:calc(.65rem + env(safe-area-inset-bottom));
    width:auto;
    min-height:52px;
    box-shadow:0 5px 20px #17203333;
  }
}

/* TASK-076 Assisted Review V1 */
.assisted-row{display:flex;flex-wrap:wrap;gap:.6rem;align-items:flex-start;padding:.65rem 0;border-top:1px solid #edf1f7}
.assisted-row:first-child{border-top:none}
.assisted-row .assisted-text{flex:1 1 260px;min-width:0;overflow-wrap:anywhere}
.assisted-actions{display:flex;flex-wrap:wrap;gap:.4rem}
label.assisted-row{cursor:pointer}
.assisted-lane-conflict{border-color:#a01625}
@media(max-width:520px){
  .assisted-row{
    flex-direction:column;
    align-items:stretch;
  }
}

</style>
<main class=shell><h1>Podcast Human Review</h1><p class=muted>Every decision is explicit and stored in the audit trail. Third ASR is evidence, never an automatic resolution.</p><p><a href="/tags">Manage controlled tag vocabulary</a></p><div id=episode-nav></div><section id=batch-actions class="panel hidden"><h3>Batch approval</h3><p class=secondary><span id=batch-count></span> Python-qualified low-risk recommendations can be accepted together. Every accepted item is stored as an individual human decision.</p><button class=primary id=accept-recommended-batch>Accept recommended batch</button></section><section id=triage-degraded class="panel hidden"><p class=error>Advisory triage is unavailable for <b id=triage-degraded-count>0</b> pending cards, so batch approval cannot recommend them. Review them individually.</p></section><section id=assisted-availability class="panel hidden"><p class=secondary>Assisted evidence not yet prepared for eligible pending cards: <b id=assisted-count>0</b>. Optional: continue Detailed Review or use Assisted review to prepare evidence.</p></section><section id=assisted-panel class="panel hidden"><h3>Assisted review</h3><p class=secondary>Third-ASR evidence is advisory only. Preparing evidence never chooses a source; every confirmed decision is stored as an individual human decision, separate from Batch approval.</p><div id=assisted-pending></div><div id=assisted-lanes></div><p id=assisted-budget class=secondary></p><p id=assisted-budget-help class=secondary>These figures cover metered AI attempts for this episode's Apple/Whisper source generation, not just this review session. Settled is recorded actual cost; reserved is an outstanding upper-bound hold, including queued preparation, not confirmed spend. Uncertain retains the reserved amount for unverified cost, ambiguous post-send outcomes, over-reservation integrity failure or reconciled legacy Third-ASR; proven pre-send releases count zero. Remaining is the $1.60 total cap less settled, reserved and uncertain; Third-ASR remaining is the shared $0.10 sub-cap less those Third-ASR amounts, not extra budget. These are ledger headroom, not approval for another paid call: downstream reserves can restrict optional Third-ASR, identity reconciliation gates Third-ASR, and integrity failure blocks new reservations. Selected max cost adds this session's quoted per-item maximums, even after a hold is released; session expires marks the preparation lease, not a ledger reset.</p><p><button class=primary id=assisted-confirm disabled>Confirm 0 human decisions</button></p><p class=error id=assisted-error></p></section><div id=app>Loading…</div></main>
<script>
let record, cards=[], progress={total:0,reviewed:0,remaining:0}, index=0, busy=false, selection=null, deferredIds=new Set(), episodeList=[], activeEpisodeKey=null, activeEpisodeReason=null, customPreviewReady=false, assistedSelection={}, assistedDecisions={}, assistedTouched={}, assistedSession=null, assistedBudget=null, assistedPreparing=false;
const q=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function messageFor(r, text){const fallback=`Request failed (${r.status}): ${r.statusText||'Unexpected response'}`;if(!text)return fallback;try{const body=JSON.parse(text);return body&&typeof body.error==='string'?body.error:fallback}catch(_){return fallback}}
async function api(url, options={}){let r;try{r=await fetch(url,options)}catch(_){throw Error('Network request failed. Please try again.')}const text=await r.text();if(!r.ok)throw Error(messageFor(r,text));if(!text)return {};try{return JSON.parse(text)}catch(_){throw Error(`Request failed (${r.status}): Invalid JSON response`)}}
function setBusy(value){busy=value;document.querySelectorAll('button').forEach(button=>button.disabled=value||button.dataset.requiresThird==='true'&&!currentCard()?.third_available);const confirm=q('#confirm-selection');if(confirm)confirm.disabled=value||!selection||(selection.source==='custom'&&!customPreviewReady)}
function currentCard(){return cards[index]}
function deferredCards(){return cards.filter(c=>deferredIds.has(String(c.id)))}
function reviewableIndices(){return cards.map((c,i)=>deferredIds.has(String(c.id))?null:i).filter(i=>i!==null)}
function nextReviewableIndex(after=index){const available=reviewableIndices();if(!available.length)return -1;return available.find(i=>i>after)??available[0]}
function deferredListHtml(){const items=deferredCards();if(!items.length)return '';return `<section class=panel id=deferred-review><h3>Deferred this pass (${items.length})</h3><p class=secondary>Deferred cards remain unresolved and still block Recompile &amp; continue until you return and confirm a decision.</p>${items.map(c=>`<p><button data-resume-deferred=${c.id}>Review #${c.id} now</button> <span class=secondary>${esc(c.display?.category||'Review item')}</span></p>`).join('')}</section>`}
function wireDeferredControls(){document.querySelectorAll('[data-resume-deferred]').forEach(button=>button.onclick=()=>resumeDeferred(button.dataset.resumeDeferred))}
function deferCurrent(){const c=currentCard();if(!c)return;selection=null;customPreviewReady=false;document.body.classList.remove('has-selection');deferredIds.add(String(c.id));const next=nextReviewableIndex(index);if(next>=0)index=next;render();updateBatchActions();renderAssisted()}
function resumeDeferred(id){const target=cards.findIndex(c=>String(c.id)===String(id));if(target<0)return;deferredIds.delete(String(id));index=target;selection=null;customPreviewReady=false;render();updateBatchActions();renderAssisted()}
function batchRecommendations(){return cards.filter(c=>!deferredIds.has(String(c.id))).map(card=>card.batch_recommendation).filter(item=>item&&Number.isInteger(item.id)&&(item.source==='apple'||item.source==='whisper'))}
function updateBatchActions(){const panel=q('#batch-actions'),button=q('#accept-recommended-batch'),count=q('#batch-count'),decisions=batchRecommendations();if(!panel||!button||!count)return;panel.classList.toggle('hidden',!decisions.length);count.textContent=decisions.length;button.onclick=acceptRecommendedBatch}
// TASK-076 Task 12: Assisted review lanes -- select pending audio_pending
// cards, prepare Third-ASR evidence with a bounded 4-concurrent browser
// scheduler, then confirm Apple/Whisper/Defer per row. Separate end to end
// from the strict low-risk Batch panel above: no shared state, no global
// all-Apple/all-Whisper action, and Conflicting/Insufficient lanes are
// never preselected.
function assistedEligibleCards(){return cards.filter(c=>!deferredIds.has(String(c.id))&&c.assisted_review&&c.assisted_review.routing&&c.assisted_review.routing.eligible)}
function assistedLaneFor(state){if(state==='machine_supported_apple'||state==='machine_supported_whisper')return 'machine';if(state==='evidence_conflict')return 'conflict';if(state==='ambiguous_audio'||state==='neither_source_supported'||state==='audio_unavailable')return 'insufficient';return null}
const ASSISTED_LANES=[['machine','Machine-supported'],['conflict','Conflicting evidence'],['insufficient','Insufficient / unavailable evidence']];
function assistedStateLabel(state){return ({audio_pending:'Evidence not prepared',audio_unavailable:'Evidence unavailable',machine_supported_apple:'Machine supports Apple',machine_supported_whisper:'Machine supports Whisper',evidence_conflict:'Evidence conflict',ambiguous_audio:'Ambiguous audio',neither_source_supported:'Neither source supported'})[state]||state}
function assistedPendingHtml(items){if(!items.length)return '';const rows=items.map(c=>`<label class=assisted-row><input type=checkbox data-assisted-select=${c.id} ${assistedSelection[c.id]?'checked':''}> <span class=assisted-text><b>#${c.id}</b> ${esc(c.display.category)}</span></label>`).join('');return `<section class=panel><h4>Needs evidence (${items.length} of ${progress.total} total review items)</h4>${rows}<p><button id=assisted-prepare ${assistedPreparing?'disabled':''}>${assistedPreparing?'Preparing…':'Prepare selected evidence'}</button></p></section>`}
function assistedRowHtml(c){const a=c.assisted_review,chosen=assistedDecisions[c.id],deferred=!(String(c.id) in assistedDecisions);const third=a.third_asr,evidenceText=third&&third.text?esc(third.text):'Not available';const matchText=a.matches?`Apple ${a.matches.apple.score.toFixed(2)} · Whisper ${a.matches.whisper.score.toFixed(2)}`:'No deterministic match';const marginText=a.margin!=null?` · margin ${a.margin.toFixed(2)}`:'';const triage=a.triage&&a.triage.recommendation?`${esc(a.triage.recommendation)}${a.triage.confidence?` (${esc(a.triage.confidence)})`:''}`:'none';const suggestion=a.compiler_suggestion&&a.compiler_suggestion.source?esc(a.compiler_suggestion.source):'none';const routing=(a.reason_codes||[]).join(', ')||'none';const btn=(source,label)=>`<button data-assisted-id=${c.id} data-assisted-source=${source} class="${(source==='defer'?deferred:chosen===source)?'selected':''}">${label}</button>`;return `<div class=assisted-row><div class=assisted-text><b>#${c.id}</b> ${esc(c.display.category)} · <span class=tag>${esc(assistedStateLabel(a.state))}</span>${a.recommendation?` · recommends <b>${esc(a.recommendation)}</b>`:''}<div class=exact>Apple: ${esc(c.apple_text)}</div><div class=exact>Whisper: ${esc(c.whisper_text)}</div><div class=exact>Third ASR: ${evidenceText}</div><p class=secondary>Match: ${matchText}${marginText} · Triage: ${triage} · Compiler suggestion: ${suggestion} · Routing: ${routing}</p></div><div class=assisted-actions>${btn('apple','Apple')}${btn('whisper','Whisper')}${btn('defer','Defer')}</div></div>`}
function laneHtml(lane,label,items){if(!items.length)return '';return `<section class="panel assisted-lane assisted-lane-${lane}"><h4>${esc(label)} (${items.length})</h4>${items.map(assistedRowHtml).join('')}</section>`}
function assistedBudgetSummaryText(){if(!assistedBudget)return '';const parts=[`Settled $${assistedBudget.settled_spend.toFixed(2)}`,`Reserved $${assistedBudget.live_reservations.toFixed(2)}`,`Uncertain $${assistedBudget.uncertain_spend.toFixed(2)}`,`Remaining $${assistedBudget.headroom_usd.toFixed(2)}`,`Third-ASR remaining $${assistedBudget.third_asr_headroom_usd.toFixed(2)}`];if(assistedSession){const selectedCost=Object.values(assistedSession.items).reduce((sum,item)=>sum+(item.reserved_usd?Number(item.reserved_usd):0),0);parts.push(`Selected max cost $${selectedCost.toFixed(2)}`);parts.push(`Session expires ${new Date(assistedSession.expires_at).toLocaleTimeString()}`)}return parts.join(' · ')}
function updateAssistedConfirmButton(){const n=Object.keys(assistedDecisions).length;const button=q('#assisted-confirm');if(!button)return;button.textContent=`Confirm ${n} human decision${n===1?'':'s'}`;button.disabled=assistedPreparing||!n}
function wireAssistedRowControls(){document.querySelectorAll('[data-assisted-select]').forEach(box=>box.onchange=()=>{assistedSelection[box.dataset.assistedSelect]=box.checked});const prepareButton=q('#assisted-prepare');if(prepareButton)prepareButton.onclick=startAssistedPrepare;document.querySelectorAll('[data-assisted-id]').forEach(button=>button.onclick=()=>{const id=button.dataset.assistedId,source=button.dataset.assistedSource;assistedTouched[id]=true;if(source==='defer')delete assistedDecisions[id];else assistedDecisions[id]=source;renderAssisted()});const confirmButton=q('#assisted-confirm');if(confirmButton)confirmButton.onclick=confirmAssistedDecisions}
function renderAssisted(){const panel=q('#assisted-panel');if(!panel)return;const eligible=assistedEligibleCards();const eligibleIds=new Set(eligible.map(c=>String(c.id)));Object.keys(assistedDecisions).forEach(id=>{if(!eligibleIds.has(id))delete assistedDecisions[id]});Object.keys(assistedTouched).forEach(id=>{if(!eligibleIds.has(id))delete assistedTouched[id]});Object.keys(assistedSelection).forEach(id=>{if(!eligibleIds.has(id))delete assistedSelection[id]});panel.classList.toggle('hidden',!eligible.length);if(!eligible.length){q('#assisted-pending').innerHTML='';q('#assisted-lanes').innerHTML='';updateAssistedConfirmButton();return}const pending=[],byLane={machine:[],conflict:[],insufficient:[]};eligible.forEach(c=>{const lane=assistedLaneFor(c.assisted_review.state);if(lane)byLane[lane].push(c);else pending.push(c)});byLane.machine.forEach(c=>{if(!assistedTouched[c.id]){assistedDecisions[c.id]=c.assisted_review.recommendation;assistedTouched[c.id]=true}});q('#assisted-pending').innerHTML=assistedPendingHtml(pending);q('#assisted-lanes').innerHTML=ASSISTED_LANES.map(([lane,label])=>laneHtml(lane,label,byLane[lane])).join('');const budget=q('#assisted-budget');if(budget)budget.textContent=assistedBudgetSummaryText();wireAssistedRowControls();updateAssistedConfirmButton()}
function assistedQueuedIds(){return assistedSession?Object.entries(assistedSession.items).filter(([_,item])=>item.state==='queued').map(([id])=>id):[]}
async function prepareAssistedItem(id){const item=assistedSession.items[id];if(!item)return;try{const result=await api(`/api/review/episodes/${encodeURIComponent(record.episode_key)}/assisted/prepare/${encodeURIComponent(assistedSession.session_id)}/items/${id}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({attempt_id:item.attempt_id})});if(result.status==='in_flight'){item.state='retrying';await new Promise(resolve=>setTimeout(resolve,Math.min(1000*(result.retry_after_seconds||1),8000)));return prepareAssistedItem(id)}item.state=result.status||'prepared'}catch(e){item.state='failed';q('#assisted-error').textContent=e.message}}
async function runAssistedScheduler(){const concurrency=4;let queue=assistedQueuedIds();while(queue.length){const batch=queue.splice(0,concurrency);await Promise.all(batch.map(id=>prepareAssistedItem(id)));queue=assistedQueuedIds()}}
async function startAssistedPrepare(){if(assistedPreparing)return;const selectedIds=Object.entries(assistedSelection).filter(([_,checked])=>checked).map(([id])=>Number(id));if(!selectedIds.length){q('#assisted-error').textContent='Select at least one item to prepare evidence for.';return}const generation=record?.human_review_generation_fingerprint;if(!generation){q('#assisted-error').textContent='Review generation is unavailable. Reload before preparing evidence.';return}assistedPreparing=true;renderAssisted();q('#assisted-error').textContent='';try{const result=await api(`/api/review/episodes/${encodeURIComponent(record.episode_key)}/assisted/prepare`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({selected_ids:selectedIds,expected_generation_fingerprint:generation})});assistedSession=result.session;assistedBudget=result.budget;await runAssistedScheduler()}catch(e){q('#assisted-error').textContent=e.message}assistedPreparing=false;await loadRecord(record.episode_key)}
async function confirmAssistedDecisions(){if(assistedPreparing)return;const decisions=Object.entries(assistedDecisions).map(([id,source])=>({id:Number(id),source}));if(!decisions.length)return;const generation=record?.human_review_generation_fingerprint;const policyVersion=assistedEligibleCards()[0]?.assisted_review.policy_version;if(!generation||!policyVersion){q('#assisted-error').textContent='Review generation or policy version is unavailable. Reload before confirming.';return}if(!window.confirm(`Confirm ${decisions.length} assisted human decision${decisions.length===1?'':'s'}?`))return;setBusy(true);try{await api(`/api/review/episodes/${encodeURIComponent(record.episode_key)}/assisted/decisions`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({decisions,expected_generation_fingerprint:generation,assisted_policy_version:policyVersion})});assistedDecisions={};assistedTouched={};assistedSelection={};assistedSession=null;await loadRecord(record.episode_key);busy=false;render()}catch(e){setBusy(false);q('#assisted-error').textContent=e.message}}

function renderEpisodeNav(){const nav=q('#episode-nav');if(!episodeList.length){nav.innerHTML='';return}if(episodeList.length===1){const e=episodeList[0];nav.innerHTML=`<section class=episode-current><span class=episode-podcast>${esc(e.podcast||'Podcast')}</span><strong class=episode-title>${esc(e.title)}</strong>${activeEpisodeReason?`<span class=episode-reason>${esc(activeEpisodeReason)}</span>`:''}</section>`;return}nav.innerHTML=`<section class=episode-queue><div class=episode-queue-head><h2>Review queue</h2><span class=secondary>${episodeList.length} episodes</span></div><div class=episode-grid>${episodeList.map(e=>`<button class="episode-card ${e.episode_key===activeEpisodeKey?'active':''}" data-episode-key="${esc(e.episode_key)}"><span class=episode-podcast>${esc(e.podcast||'Podcast')}</span><strong class=episode-title>${esc(e.title)}</strong><span class=episode-footer><span>${e.pending_count} ${e.pending_count===1?'difference':'differences'}</span>${e.episode_key===activeEpisodeKey?'<span class=tag>Reviewing</span>':''}</span></button>`).join('')}</div></section>`;document.querySelectorAll('[data-episode-key]').forEach(button=>button.onclick=()=>loadRecord(button.dataset.episodeKey))}
async function loadEpisodes(){const data=await api('/api/review/episodes');episodeList=data.episodes;if(!episodeList.length){activeEpisodeKey=null;renderEpisodeNav();q('#app').innerHTML='<section class=card><h2>No episodes in the review queue</h2><p class=muted>There are currently no transcript differences that require human review.</p></section>';return}if(!activeEpisodeKey||!episodeList.some(e=>e.episode_key===activeEpisodeKey))activeEpisodeKey=episodeList[0].episode_key;renderEpisodeNav();await loadRecord(activeEpisodeKey)}
async function loadRecord(episodeKey=activeEpisodeKey){const switchingEpisode=activeEpisodeKey!==episodeKey||!record;activeEpisodeKey=episodeKey;renderEpisodeNav();index=0;selection=null;if(switchingEpisode){deferredIds=new Set();assistedDecisions={};assistedTouched={};assistedSelection={};assistedSession=null;assistedBudget=null}const data=await api(`/api/review/episodes/${activeEpisodeKey}`);record={...data.record,recompile_status:data.recompile_status};cards=data.cards;const pendingIds=new Set(cards.map(c=>String(c.id)));deferredIds=new Set([...deferredIds].filter(id=>pendingIds.has(id)));progress=data.progress;const next=nextReviewableIndex(-1);index=next>=0?next:0;render();updateBatchActions();renderAssisted()}
function replacement(c,source){return Object.hasOwn(c.source_choices||{},source)?c.source_choices[source]:undefined}
function select(source,text){selection={source,text,representationText:null};customPreviewReady=source!=='custom';document.body.classList.add('has-selection');q('#preview').textContent=text;q('#error').textContent='';q('#confirm-selection').disabled=!customPreviewReady;const representation=q('#representation-text');if(representation){const eligible=source==='apple'||source==='whisper';representation.disabled=!eligible;representation.value=eligible?text:'';q('#representation-help').textContent=eligible?'Optional case-only edit. Words, spacing, and punctuation must stay exactly the same.':'Choose Apple or Whisper first to adjust representation/casing.'}document.querySelectorAll('[data-source],#custom-open').forEach(button=>button.classList.toggle('selected',button.dataset.source===source||(source==='custom'&&button.id==='custom-open')))}
function recompileStatus(){return record?.recompile_status}
function recompileActive(){return Boolean(recompileStatus())}
function render(){document.body.classList.toggle('has-selection',Boolean(selection));const app=q('#app');const availabilityPanel=q('#assisted-availability');if(availabilityPanel){const count=progress.assisted_unprepared||0;availabilityPanel.classList.toggle('hidden',count===0);const countEl=q('#assisted-count');if(countEl)countEl.textContent=count}const triageDegraded=q('#triage-degraded');if(triageDegraded){const count=progress.triage_unavailable||0;triageDegraded.classList.toggle('hidden',count===0);const countEl=q('#triage-degraded-count');if(countEl)countEl.textContent=count}if(!cards.length){activeEpisodeReason=null;renderEpisodeNav();const status=recompileStatus(),active=Boolean(status),label=status==='unknown'?'Recompile status unknown…':status==='completed'?'Recompile completed':active?'Recompile started…':'Recompile &amp; continue';app.innerHTML=`<section class=card><h2>Human review complete</h2><p>All decisions are stored in the audit trail. <b>Recompile &amp; continue</b> starts the normal worker, which rebuilds the compiled transcript using those audited decisions. If it succeeds, the normal knowledge pipeline continues automatically.</p><button class=primary id=recompile ${active?'disabled':''}>${label}</button><p class=error id=error></p></section>`;if(!active)q('#recompile').onclick=recompile;return}const reviewable=reviewableIndices();if(!reviewable.length){activeEpisodeReason='Deferred items remain';renderEpisodeNav();app.innerHTML=`<section class=card><h2>All remaining cards are deferred</h2><p>These ${cards.length} unresolved card${cards.length===1?' remains':'s remain'} pending. Return to a deferred card and confirm a real decision before Recompile &amp; continue can become available.</p></section>${deferredListHtml()}`;wireDeferredControls();return}if(deferredIds.has(String(currentCard()?.id)))index=nextReviewableIndex(index-1);activeEpisodeReason=currentCard()?.display?.reason||null;renderEpisodeNav();const c=currentCard(),f=c.focus||{},partial=f.scope==='partial',third=c.third_asr;const conflict=partial?`<section class=panel><h3>Focused conflict</h3><div class=sources><div class="source"><b>Apple</b><div class=exact>${esc(f.apple_text)}</div></div><div class="source whisper"><b>Whisper</b><div class=exact>${esc(f.whisper_text)}</div></div></div></section>`:`<section class=panel><h3>Transcript difference</h3><div class=sources><div class="source"><b>Apple</b><div class=exact>${esc(c.apple_text)}</div></div><div class="source whisper"><b>Whisper</b><div class=exact>${esc(c.whisper_text)}</div></div></div></section>`;const suggestion=c.suggestion?.source?`${esc(c.suggestion.source)} — ${esc(c.suggestion.reason||'source-backed compiler preference')}`:'No automatic resolution';const triage=c.triage;const triagePanel=triage?`<section class=panel><h3>Advisory triage</h3><p><b>Recommendation:</b> ${esc(triage.recommendation)}${triage.confidence?` · <b>Confidence:</b> ${esc(triage.confidence)}`:''}</p><p class=secondary>${esc(triage.reason)}</p></section>`:'';const anomaly=c.anomaly;const anomalyPanel=anomaly?`<section class=panel><h3>Review anomaly</h3><p><b>${esc(anomaly.kind)}</b></p><p class=secondary>${esc(anomaly.reason)}</p></section>`:'';const audio=c.audio_window?`${c.audio_window.start.toFixed(1)}–${c.audio_window.end.toFixed(1)} seconds`:'Clip around the Whisper timestamp';const thirdAction=third?'<p class=secondary>Evidence cached.</p>':'<button class=external id=third-run>Run third ASR</button>';app.innerHTML=`<section class=card><div class=meta><b>Reviewed ${progress.reviewed} of ${progress.total}</b><span class=tag>${progress.remaining} remaining</span>${deferredCards().length?`<span class=tag>${deferredCards().length} deferred this pass</span>`:''}<span class=tag>${esc(c.display.category)}</span><span class=tag>${esc(c.display.severity)}</span></div>${conflict}<section class=panel><h3>Suggestion / evidence</h3><p>${suggestion}</p></section>${triagePanel}${anomalyPanel}<section class=panel><h3>Audio evidence</h3><p class=secondary>${esc(audio)}</p><audio id=audio controls src="/api/review/episodes/${encodeURIComponent(record.episode_key)}/items/${c.id}/audio"></audio><p id=audio-error class=audio-error></p></section><section class=panel><h3>Context</h3><div class=sources><div class="source"><b>Apple context</b><div class=context>${esc(c.apple_context)}</div></div><div class="source whisper"><b>Whisper context</b><div class=context>${esc(c.whisper_context)}</div></div></div></section><section class=panel><h3>Third ASR evidence</h3>${third?`<div class=exact>${esc(third.text)}</div><p class=secondary>${esc([third.model,third.duration&&`${third.duration}s`].filter(Boolean).join(' · '))}</p>${c.third_available?`<p><b>Use Third inserts:</b> <span class=exact>${esc(c.third_window)||'(nothing — Third ASR hears no words here)'}</span></p>`:'<p class=secondary>Third ASR could not be aligned to this card; use Custom/Edit if it helps.</p>'}`:'<p class=secondary>Not requested yet.</p>'}${thirdAction}<p id=third-error class=error></p></section><section class=panel><h3>Representation/casing</h3><p id=representation-help class=secondary>Choose Apple or Whisper first to adjust representation/casing.</p><input id=representation-text placeholder="Optional case-only representation" disabled></section><section class=panel><h3>Selected replacement</h3><div id=preview class=exact>Select a choice to preview the exact replacement for this review item.</div><p><input id=note placeholder="Optional audit note"></p><button class=primary id=confirm-selection disabled>Confirm selection</button></section><div class=choices><button class=primary data-source=apple>Use Apple</button><button class=primary data-source=whisper>Use Whisper</button><button class=external data-source=third data-requires-third=true ${c.third_available?'':'disabled'}>Use Third</button><button id=custom-open>Custom/Edit</button><button class=subtle id=defer>Defer for this pass</button></div>${deferredListHtml()}<section id=custom-panel class="panel hidden"><h3>Custom/Edit replacement</h3><textarea id=text placeholder="Human-authored replacement text"></textarea><h3>Expand edit range</h3><p class=secondary>Optional. Include up to 3 adjacent canonical-source words on either side when the spoken correction crosses the detected conflict boundary.</p><div class=sources><label>Previous words<select id=expand-left><option value=0>0</option><option value=1>1</option><option value=2>2</option><option value=3>3</option></select></label><label>Next words<select id=expand-right><option value=0>0</option><option value=1>1</option><option value=2>2</option><option value=3>3</option></select></label></div><h3>Replacing exactly</h3><div id=custom-replaced-text class=exact>The exact canonical-source slice will appear after preview.</div><h3>Final merged context</h3><div id=custom-final-preview class=exact>Enter a custom replacement to calculate the exact final context.</div><p id=custom-preview-error class=error></p></section><p class=error id=error></p></section>`;q('#audio').onerror=()=>q('#audio-error').textContent='Audio clip could not be loaded. You can still review this card.';const thirdRun=q('#third-run');if(thirdRun)thirdRun.onclick=runThird;document.querySelectorAll('[data-source]').forEach(button=>button.onclick=()=>{const text=button.dataset.source==='third'?(c.third_available?c.third_window:undefined):replacement(c,button.dataset.source);if(text===undefined){q('#error').textContent=`No replacement is available for ${button.dataset.source}.`;return}select(button.dataset.source,text)});q('#custom-open').onclick=()=>{q('#custom-panel').classList.remove('hidden');q('#text').focus();previewCustom()};q('#text').oninput=previewCustom;q('#expand-left').onchange=previewCustom;q('#expand-right').onchange=previewCustom;q('#representation-text').oninput=()=>{if(!selection||(selection.source!=='apple'&&selection.source!=='whisper'))return;selection.representationText=q('#representation-text').value;q('#preview').textContent=selection.representationText};q('#confirm-selection').onclick=confirmSelection;q('#defer').onclick=deferCurrent;wireDeferredControls()}

async function previewCustom(){
    const text=q('#text').value;
    const expandLeftWords=Number(q('#expand-left')?.value||0);
    const expandRightWords=Number(q('#expand-right')?.value||0);
    select('custom',text);
    customPreviewReady=false;
    const replacedPreview=q('#custom-replaced-text');
    const finalPreview=q('#custom-final-preview');
    const previewError=q('#custom-preview-error');
    previewError.textContent='';
    q('#confirm-selection').disabled=true;
    if(!text.trim()){
        replacedPreview.textContent='The exact canonical-source slice will appear after preview.';
        finalPreview.textContent='Enter a custom replacement to calculate the exact final context.';
        return;
    }
    replacedPreview.textContent='Calculating exact source range…';
    finalPreview.textContent='Calculating exact final context…';
    const requestedText=text;
    const requestedLeft=expandLeftWords;
    const requestedRight=expandRightWords;
    try{
        const result=await api(
            `/api/review/episodes/${record.episode_key}/items/${currentCard().id}/preview`,
            {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body:JSON.stringify({
                    source:'custom',
                    text:requestedText,
                    expand_left_words:requestedLeft,
                    expand_right_words:requestedRight
                })
            }
        );
        if(
            !selection
            || selection.source!=='custom'
            || q('#text').value!==requestedText
            || Number(q('#expand-left')?.value||0)!==requestedLeft
            || Number(q('#expand-right')?.value||0)!==requestedRight
        )return;
        customPreviewReady=true;
        replacedPreview.textContent=result.replaced_text;
        finalPreview.textContent=result.merged_context;
        q('#confirm-selection').disabled=false;
    }catch(e){
        if(
            q('#text').value!==requestedText
            || Number(q('#expand-left')?.value||0)!==requestedLeft
            || Number(q('#expand-right')?.value||0)!==requestedRight
        )return;
        customPreviewReady=false;
        replacedPreview.textContent='Exact source range is unavailable.';
        finalPreview.textContent='Exact final context is unavailable.';
        previewError.textContent=e.message;
        q('#confirm-selection').disabled=true;
    }
}

async function runThird(){const c=currentCard(),button=q('#third-run');setBusy(true);button.textContent='Running third ASR…';try{const r=await api(`/api/review/episodes/${record.episode_key}/items/${c.id}/third-asr`,{method:'POST'});const cardIdBeforeReload=c.id;const data=await api(`/api/review/episodes/${record.episode_key}`);record={...data.record,recompile_status:data.recompile_status};cards=data.cards;const pendingIds=new Set(cards.map(c=>String(c.id)));deferredIds=new Set([...deferredIds].filter(id=>pendingIds.has(id)));progress=data.progress;const cardIndex=cards.findIndex(card=>card.id===cardIdBeforeReload);if(cardIndex>=0){index=cardIndex}else{const next=nextReviewableIndex(-1);index=next>=0?next:0}busy=false;render();updateBatchActions();renderAssisted()}catch(e){setBusy(false);button.textContent='Run third ASR';q('#third-error').textContent=e.message}}
async function acceptRecommendedBatch(){if(busy)return;const decisions=batchRecommendations().map(({id,source})=>({id,source}));if(!decisions.length)return;const generation=record?.human_review_generation_fingerprint;if(!generation){q('#error').textContent='Review generation is unavailable. Reload before batch approval.';return}if(!window.confirm(`Accept ${decisions.length} low-risk recommendations? Each item will be recorded as an individual human decision.`))return;const episodeKey=record.episode_key;setBusy(true);try{await api(`/api/review/episodes/${encodeURIComponent(episodeKey)}/batch-decision`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({expected_generation_fingerprint:generation,decisions})});await loadRecord(episodeKey);const activeEpisode=episodeList.find(e=>e.episode_key===episodeKey);if(activeEpisode){activeEpisode.pending_count=cards.length;renderEpisodeNav()}busy=false;render();updateBatchActions()}catch(e){busy=false;render();updateBatchActions();const error=q('#error');if(error)error.textContent=e.message}}
async function confirmSelection(){if(busy||!selection)return;const c=currentCard();if(selection.source==='custom'&&!selection.text.trim()){q('#error').textContent='Custom/Edit requires non-empty replacement text.';return}if(selection.source==='custom'&&!customPreviewReady){q('#error').textContent='Custom/Edit requires an exact final merged-context preview before confirmation.';return}const body={source:selection.source,note:q('#note').value||null};if(selection.source==='custom'){body.text=selection.text;body.expand_left_words=Number(q('#expand-left')?.value||0);body.expand_right_words=Number(q('#expand-right')?.value||0)}if((selection.source==='apple'||selection.source==='whisper')&&selection.representationText!==null&&selection.representationText!==selection.text)body.representation_text=selection.representationText;setBusy(true);try{await api(`/api/review/episodes/${record.episode_key}/items/${c.id}/decision`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});deferredIds.delete(String(c.id));const wasEligibleAudioPending=c.assisted_review?.routing?.eligible&&c.assisted_review?.state==='audio_pending';cards.splice(index,1);progress={...progress,reviewed:progress.reviewed+1,remaining:cards.length};if(wasEligibleAudioPending&&progress.assisted_unprepared>0){progress.assisted_unprepared--}const activeEpisode=episodeList.find(e=>e.episode_key===record.episode_key);if(activeEpisode){activeEpisode.pending_count=cards.length;renderEpisodeNav()}const next=nextReviewableIndex(index-1);index=next>=0?next:0;selection=null;busy=false;render();updateBatchActions();renderAssisted()}catch(e){setBusy(false);q('#error').textContent=e.message}}
async function recompile(){const button=q('#recompile');button.disabled=true;try{await api(`/api/review/episodes/${record.episode_key}/recompile`,{method:'POST'});button.textContent='Worker started'}catch(e){button.disabled=false;q('#error').textContent=e.message}}
loadEpisodes().catch(e=>q('#app').textContent=e.message);
</script>"""


TAG_PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover"><title>Personal Knowledge Tags · Podcast Human Review</title>
<style>:root{color:#172033;background:#f6f8fb;font:16px/1.5 system-ui,sans-serif}body{margin:0}.shell{max-width:1160px;margin:auto;padding:2rem 1rem}.card{background:#fff;border:1px solid #d6dde8;border-radius:12px;padding:1rem;margin:1rem 0}.tag{background:#edf1f7;border-radius:999px;padding:.15rem .55rem;margin:.15rem;display:inline-block}button,select,input{font:inherit;padding:.45rem;margin:.2rem}.error{color:#a01625}.muted{color:#536176}.episodes{max-height:10rem;overflow:auto}</style>
<main class=shell><p><a href="/">← Transcript review</a></p><h1>Personal Knowledge Tag Registry</h1><p class=muted>Review decisions only schedule backfill. They never change artifacts; local Markdown writes wait for the macOS knowledge agent.</p><p><button data-status="pending">Pending</button><button data-status="promoted">Canonical</button><button data-status="mapped">Mapped</button><button data-status="rejected">Rejected</button><button id=history>History</button></p><div id=app>Loading…</div></main>
<script>
const app=document.querySelector('#app'), esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function api(url,opts={}){const r=await fetch(url,opts),t=await r.text();if(!r.ok){try{throw Error(JSON.parse(t).error)}catch(e){throw e instanceof Error?e:Error('Request failed')}}return t?JSON.parse(t):{}}
async function list(status='pending'){try{const [data,agent,candidates]=await Promise.all([api(`/api/tag-vocabulary?status=${encodeURIComponent(status)}`),api('/api/tag-vocabulary/agent-status'),api('/api/tag-vocabulary/artifacts/candidates')]);app.innerHTML=agentCard(agent.agent)+candidateCards(candidates.candidates)+(data.candidates.length?data.candidates.map(card).join(''):'<section class=card>No entries in this view.</section>');bind()}catch(e){app.innerHTML=`<p class=error>${esc(e.message)}</p>`}}
function agentCard(a){const r=a.last_run||{};return `<section class=card><h2>Mac Knowledge Agent <span class=tag>${esc(a.status||'Stale')}</span></h2><p>Last sync: ${esc(a.last_successful_sync_at||'Never')} · Last run: ${esc(a.last_finished_at||'Never')} · ${esc(a.duration_seconds??'—')} s</p><p>Notes indexed: ${a.notes_indexed||0} · Waiting for adoption: ${a.waiting_for_adoption||0} · Unresolved tags: ${a.unresolved_tags||0} · Pending backfills: ${a.pending_backfills||0}</p><p>Last run: Scanned ${r.scanned||0} · Changed ${r.changed||0} · Backfilled ${r.backfilled||0} · No-op ${r.no_op||0} · Errors ${r.errors||0}</p></section>`}
function candidateCards(items){if(!items.length)return '';return `<section class=card><h2>Knowledge candidates</h2>${items.map(c=>`<p><b>${esc(c.title||c.relative_path)}</b> · ${esc(c.source_type)} · ${esc(c.relative_path)} <button data-adopt="${esc(c.candidate_id)}">Adopt into Knowledge System</button></p>`).join('')}</section>`}
function card(e){const b=e.backfill||{}, actions=e.status==='pending'||e.status==='rejected'?`<button data-action="promote" data-slug="${esc(e.slug)}">Promote</button><button data-action="map" data-slug="${esc(e.slug)}">Map to existing</button><button data-action="reject" data-slug="${esc(e.slug)}">Reject</button>`:`<button data-action="promote" data-slug="${esc(e.slug)}">Promote</button><button data-action="map" data-slug="${esc(e.slug)}">Change map</button><button data-action="reject" data-slug="${esc(e.slug)}">Reject</button><button data-action="reopen" data-slug="${esc(e.slug)}">Reopen</button>`;return `<section class=card><h2>${esc(e.slug)} <span class=tag>${esc(e.status)}</span></h2><p>${esc(e.category)} · ${e.occurrences} affected artifacts · ${esc(Object.entries(e.source_distribution||{}).map(([k,v])=>`${v} ${k}`).join(', ')||'no sources')}</p><p>Backfill: <b>${esc(b.status||'not_needed')}</b> (${b.completed_artifacts||0}/${b.total_artifacts||0})</p>${actions}<button data-preview="${esc(e.slug)}">Preview affected artifacts</button>${b.status==='pending'||b.status==='failed'||b.status==='completed'?`<button data-run="${esc(e.slug)}">Run backfill</button>`:''}<div id="detail-${esc(e.slug)}"></div></section>`}
function bind(){document.querySelectorAll('[data-action]').forEach(b=>b.onclick=async()=>{let body={action:b.dataset.action};if(body.action==='map'){body.mapped_to=prompt('Existing canonical tag slug:')||''}try{await api(`/api/tag-vocabulary/candidates/${encodeURIComponent(b.dataset.slug)}/decision`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});list()}catch(e){alert(e.message)}});document.querySelectorAll('[data-adopt]').forEach(b=>b.onclick=async()=>{try{const items=(await api('/api/tag-vocabulary/artifacts/candidates')).candidates;const candidate=items.find(x=>x.candidate_id===b.dataset.adopt);if(candidate)await api('/api/tag-vocabulary/artifacts/adopt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(candidate)});list()}catch(e){alert(e.message)}});document.querySelectorAll('[data-preview]').forEach(b=>b.onclick=()=>preview(b.dataset.preview));document.querySelectorAll('[data-run]').forEach(b=>b.onclick=()=>run(b.dataset.run))}
async function preview(slug){try{const d=await api(`/api/tag-vocabulary/candidates/${encodeURIComponent(slug)}/backfill-preview`);document.querySelector(`#detail-${CSS.escape(slug)}`).innerHTML=`<p class=episodes>${d.artifacts.map(e=>`${esc(e.podcast||e.source_type)} — ${esc(e.title)}`).join('<br>')||'No artifacts'}</p>`}catch(e){alert(e.message)}}
async function run(slug){if(!confirm(`Run idempotent backfill for ${slug}?`))return;try{await api(`/api/tag-vocabulary/candidates/${encodeURIComponent(slug)}/backfill`,{method:'POST'});list()}catch(e){alert(e.message)}}
document.querySelectorAll('[data-status]').forEach(b=>b.onclick=()=>list(b.dataset.status));document.querySelector('#history').onclick=async()=>{const d=await api('/api/tag-vocabulary/history');app.innerHTML=`<section class=card><h2>History</h2>${d.history.map(h=>`<p>${esc(h.timestamp)} · <b>${esc(h.candidate)}</b> · ${esc(h.action)} (${esc(h.previous_status)} → ${esc(h.new_status)})</p>`).join('')||'No history yet.'}</section>`};list();
</script>"""


LOGIN_PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover"><title>Podcast Human Review</title>
<style>body{font:16px system-ui;margin:12vh auto;max-width:36rem;padding:1.5rem}.card{border:1px solid #bbb;border-radius:.6rem;padding:1.5rem}.error{color:#a00}</style>
<main class=card><h1>Podcast Human Review</h1><p>Prijavi se s dopuštenim Google računom.</p><div id=google></div><p class=error id=error></p></main>
<script src="https://accounts.google.com/gsi/client" async></script>
<script>
const clientId={{ client_id|tojson }};
async function credential(response){let result;try{result=await fetch('/auth/google',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({credential:response.credential})})}catch(_){document.querySelector('#error').textContent='Mrežni zahtjev nije uspio.';return}if(result.ok){location.assign('/');return}const text=await result.text();try{const body=JSON.parse(text);document.querySelector('#error').textContent=body.error||'Prijava nije uspjela.'}catch(_){document.querySelector('#error').textContent=`Zahtjev nije uspio (${result.status}): ${result.statusText||'Neočekivan odgovor'}`}}
window.onload=()=>{google.accounts.id.initialize({client_id:clientId,callback:credential,auto_select:false});google.accounts.id.renderButton(document.querySelector('#google'),{theme:'outline',size:'large',text:'signin_with'});};
</script>"""


SUMMARY_REVIEW_PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover"><title>Summary Review · Podcast Ops</title>
<style>
:root{color:#172033;background:#f6f8fb;font:16px/1.5 system-ui,sans-serif}body{margin:0}.shell{max-width:760px;margin:auto;padding:2rem 1rem 3rem}.card,.panel{background:#fff;border:1px solid #d6dde8;border-radius:12px;padding:1.25rem;margin-top:1rem}.panel{background:#f9fafb}.muted,.label{color:#536176}.eyebrow{font-size:.75rem;letter-spacing:.12em;font-weight:700;color:#536176}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.75rem}.metric{border:1px solid #e1e6ee;border-radius:10px;padding:.8rem}.metric .label{display:block;font-size:.8rem}.metric strong{display:block;margin-top:.2rem}code{overflow-wrap:anywhere}@media(max-width:620px){.shell{padding:1rem .75rem}.grid{grid-template-columns:1fr}}
</style>
<main class=shell>
<p><a href="/">← PodcastOps</a></p>
<section class=card>
<p class=eyebrow>PODCAST OPS</p>
<h1>{% if diagnostics.stalled %}Summary review stalled{% else %}Summary review{% endif %}</h1>
<p><strong>{{ episode.podcast or "Podcast" }}</strong><br>{{ episode.title or episode.episode_key }}</p>
<div class=grid>
<div class=metric><span class=label>Failure count</span><strong>{{ diagnostics.failure_count }}</strong></div>
<div class=metric><span class=label>Latest review ID</span><strong><code>{{ diagnostics.latest_review_id }}</code></strong></div>
<div class=metric><span class=label>First failure</span><strong>{{ diagnostics.first_failure_at }}</strong></div>
<div class=metric><span class=label>Last failure</span><strong>{{ diagnostics.last_failure_at }}</strong></div>
<div class=metric><span class=label>Retry state</span><strong>{{ retry_state }}</strong></div>
<div class=metric><span class=label>Summary completion</span><strong>{{ summary_completion }}</strong></div>
</div>
<section class=panel>
<h2>Latest failure</h2>
<p><code>{{ latest_failure.stage }}</code> · <code>{{ latest_failure.code }}</code></p>
<p>{{ latest_failure.explanation }}</p>
</section>
{% if diagnostics.canonical_note_preserved %}
<p><strong>Canonical note preserved</strong></p>
{% endif %}
{% if technical %}
<details class=panel>
<summary><strong>Technical details</strong></summary>
<pre><code>{{ technical | tojson(indent=2) }}</code></pre>
</details>
{% endif %}
</section>
</main>"""


def _source_fingerprint(record: dict) -> SourceFingerprint:
    """Select the typed source-generation field from a loaded review record."""

    return cast(ReviewFingerprintFields, record)["source_fingerprint"]


def _episode(episode_key: str) -> dict:
    for episode in load_episodes():
        if episode.get("episode_key") == episode_key:
            return episode
    raise LookupError(f"Episode {episode_key} was not found")


def _readable(value: object, labels: dict[str, str]) -> str:
    """Turn persisted enum values into presentation-only text."""

    if not isinstance(value, str) or not value:
        return "Not specified"
    return labels.get(value, value.replace("_", " ").replace("-", " ").capitalize())


def _review_progress(record: dict, cards: list[dict]) -> dict[str, int]:
    pending_ids = {item.get("id") for item in cards if item.get("id") is not None}
    decided_ids = {
        item.get("id") for item in record.get("human_decisions", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    return {
        "total": len(pending_ids | decided_ids),
        "reviewed": len(decided_ids - pending_ids),
        "remaining": len(pending_ids),
    }


def _present_review_item(item: dict) -> dict:
    """Add UI-only review data without changing the persisted audit item."""

    presented = dict(item)
    focus = item.get("focus") if isinstance(item.get("focus"), dict) else {}
    partial = focus.get("scope") == "partial"
    source_choices = {
        source: (focus.get(f"{source}_text") if partial else item.get(f"{source}_text"))
        for source in ("apple", "whisper")
    }
    presented["source_choices"] = {
        source: text for source, text in source_choices.items() if isinstance(text, str)
    }
    evidence = item.get("third_asr")
    # TASK-123: "Use Third" inserts only the anchored clip words for this
    # card; without a safe anchor the choice is unavailable (Custom/Edit).
    window = anchored_third_asr_window(item)
    presented["third_window"] = window["text"] if window is not None else None
    presented["third_available"] = window is not None
    # TASK-076 Task 12: every presented card also carries its deterministic
    # assisted-review routing/analysis projection, recomputed fresh here
    # (never cached), so the Assisted review UI lane never needs a second
    # request to know eligibility or machine state.
    presented["assisted_review"] = derive_assisted_state(item)
    try:
        presented["audio_window"] = clip_window(
            item.get("whisper_start_timestamp"), item.get("whisper_end_timestamp")
        )
    except (TypeError, ValueError):
        presented["audio_window"] = None
    presented["display"] = {
        "category": _readable(item.get("category"), {
            "other": "General transcript difference",
            "protocol_number": "Protocol or dosage number",
            "negation": "Negation difference",
            "citation": "Citation difference",
        }),
        "severity": _readable(item.get("severity"), {
            "high": "High-risk difference", "medium": "Needs review", "low": "Lower-risk difference",
        }),
        "reason": _readable(item.get("reason"), {
            "compiler_requires_human_review": "Compiler requires human review",
            "needs_human_review": "Needs human review",
        }),
    }
    return presented


def _present_budget_summary(summary: dict) -> dict:
    """Bounded display-only view of one episode budget summary.

    Decimal totals become floats for JSON display only; the ledger itself
    (podcast_engine.ai_budget) remains the sole Decimal-exact spend
    authority and is never recomputed or re-derived from this view.
    """

    return {
        key: (float(value) if isinstance(value, Decimal) else value)
        for key, value in summary.items()
    }


def _safe_runtime_error(error: Exception, *, subject: str) -> tuple[str, int]:
    """Return bounded operational errors without provider bodies or secrets."""

    if isinstance(error, requests.RequestException):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        return f"{subject} request failed" + (f": HTTP {status}" if status else ""), 502
    if isinstance(error, (FileNotFoundError, StopIteration)):
        return f"{subject} is unavailable.", 422
    if isinstance(error, ValueError):
        return f"{subject} is unavailable: {str(error)[:180]}", 422
    if isinstance(error, RuntimeError):
        if "Missing PODCAST_REVIEW_ASR_API_KEY" in str(error):
            return "Third ASR is not configured.", 503
        if "FFmpeg" in str(error):
            return "Review audio is unavailable: FFmpeg review clip extraction failed.", 422
        return f"{subject} is temporarily unavailable.", 503
    return f"{subject} is unavailable.", 500


def create_review_app() -> Flask:
    """Create the UI app. The command entry point binds it to localhost only."""

    app = Flask(__name__)
    require_auth = os.environ.get("REVIEW_REQUIRE_AUTH", "false").casefold() == "true"
    oauth_client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
    session_secret = os.environ.get("REVIEW_SESSION_SECRET")
    allowed_email = os.environ.get("REVIEW_ALLOWED_EMAIL", "reviewer@example.com").casefold()

    if require_auth and (not oauth_client_id or not session_secret):
        raise RuntimeError("Google review login requires GOOGLE_OAUTH_CLIENT_ID and REVIEW_SESSION_SECRET")

    app.config.update(
        REVIEW_AUTH_ENABLED=require_auth,
        REVIEW_ALLOWED_EMAIL=allowed_email,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=require_auth,
    )
    app.secret_key = session_secret or "local-human-review-development-only"
    app.permanent_session_lifetime = timedelta(hours=8)

    @app.errorhandler(HTTPException)
    def api_http_error(error: HTTPException):
        if request.path.startswith("/api/"):
            return jsonify({"error": f"Request failed ({error.code}): {error.name}"}), error.code
        return error

    @app.errorhandler(Exception)
    def api_unexpected_error(error: Exception):
        if request.path.startswith("/api/"):
            app.logger.exception("Human-review API request failed")
            return jsonify({"error": "Request failed (500): Internal Server Error"}), 500
        app.logger.exception("Human-review page request failed")
        return "Internal Server Error", 500

    @app.before_request
    def require_google_login():
        if not app.config["REVIEW_AUTH_ENABLED"]:
            return None
        if request.path in {"/login", "/auth/google", "/health"}:
            return None
        if session.get("reviewer_email") == app.config["REVIEW_ALLOWED_EMAIL"]:
            return None
        if request.path.startswith("/api/"):
            return jsonify({"error": "Prijavi se s dopuštenim Google računom."}), 401
        return redirect(url_for("login"))

    @app.get("/health")
    def health():
        return jsonify({"status": "healthy", "service": "podcast-human-review"})

    @app.get("/login")
    def login():
        if not app.config["REVIEW_AUTH_ENABLED"]:
            return redirect(url_for("home"))
        if session.get("reviewer_email") == app.config["REVIEW_ALLOWED_EMAIL"]:
            return redirect(url_for("home"))
        return render_template_string(LOGIN_PAGE, client_id=oauth_client_id)

    @app.post("/auth/google")
    def google_login():
        if not app.config["REVIEW_AUTH_ENABLED"]:
            return jsonify({"error": "Google prijava nije uključena."}), 404
        credential = (request.get_json(silent=True) or {}).get("credential")
        if not isinstance(credential, str) or not credential:
            return jsonify({"error": "Nedostaje Google credential."}), 400
        try:
            claims = id_token.verify_oauth2_token(
                credential,
                GoogleRequest(),
                oauth_client_id,
            )
        except ValueError:
            return jsonify({"error": "Google credential nije valjan."}), 401
        email = claims.get("email", "").casefold()
        if not claims.get("email_verified") or email != app.config["REVIEW_ALLOWED_EMAIL"]:
            return jsonify({"error": "Ovaj Google račun nema pristup."}), 403
        session.clear()
        session.permanent = True
        session["reviewer_email"] = email
        return jsonify({"email": email})

    @app.get("/")
    def home():
        return Response(PAGE, mimetype="text/html")

    @app.get("/episodes/<episode_key>/summary-review")
    def summary_review_panel(episode_key: str):
        episode = _episode(episode_key)
        diagnostics = load_summary_review_stalled_diagnostics(
            bucket=get_bucket(),
            episode_key=episode_key,
        )
        latest_failure = (
            diagnostics.get("latest_failure")
            if isinstance(diagnostics.get("latest_failure"), dict)
            else {}
        )
        return render_template_string(
            SUMMARY_REVIEW_PAGE,
            episode=episode,
            diagnostics=diagnostics,
            latest_failure=latest_failure,
            retry_state=_readable(diagnostics.get("retry_state"), {}),
            summary_completion=_readable(
                diagnostics.get("summary_completion"),
                {},
            ),
            technical=(
                diagnostics.get("technical")
                if isinstance(diagnostics.get("technical"), dict)
                else {}
            ),
        )

    @app.get("/tags")
    def tag_vocabulary_page():
        return Response(TAG_PAGE, mimetype="text/html")

    @app.get("/api/tag-vocabulary")
    def tag_vocabulary():
        status = request.args.get("status")
        if status and status not in {"pending", "promoted", "mapped", "rejected"}:
            return jsonify({"error": "Invalid tag vocabulary status"}), 400
        try:
            return jsonify({"candidates": TagRegistry().list_candidates(status)})
        except (RuntimeError, ValueError) as error:
            return jsonify({"error": str(error)}), 409

    @app.get("/api/tag-vocabulary/history")
    def tag_vocabulary_history():
        try:
            return jsonify({"history": TagRegistry().history()})
        except (RuntimeError, ValueError) as error:
            return jsonify({"error": str(error)}), 409

    @app.get("/api/tag-vocabulary/agent-status")
    def knowledge_agent_status():
        try:
            return jsonify({"agent": KnowledgeAgentStatus().presented()})
        except (RuntimeError, ValueError) as error:
            return jsonify({"error": str(error)}), 409

    @app.get("/api/tag-vocabulary/artifacts/candidates")
    def knowledge_artifact_candidates():
        try:
            return jsonify({"candidates": TagRegistry().artifact_index.local_candidates()})
        except (RuntimeError, ValueError) as error:
            return jsonify({"error": str(error)}), 409

    @app.post("/api/tag-vocabulary/artifacts/adopt")
    def adopt_knowledge_artifact():
        """Queue an explicit local adoption; Cloud never opens a Mac path."""
        try:
            return jsonify(TagRegistry().adopt_local_artifact(request.get_json(silent=True) or {})), 202
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except RuntimeError:
            return jsonify({"error": "Knowledge registry is temporarily unavailable."}), 503

    @app.post("/api/tag-vocabulary/candidates/<candidate>/decision")
    def tag_vocabulary_decision(candidate: str):
        body = request.get_json(silent=True) or {}
        try:
            decision = TagRegistry().decide(
                candidate,
                body.get("action"),
                mapped_to=body.get("mapped_to"),
                reason=body.get("reason"),
                reviewed_by=session.get("reviewer_email", "local-human-review"),
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except RuntimeError:
            return jsonify({"error": "Tag registry is temporarily unavailable."}), 503
        return jsonify({"decision": decision})

    @app.get("/api/tag-vocabulary/candidates/<candidate>/backfill-preview")
    def tag_vocabulary_preview(candidate: str):
        try:
            return jsonify(TagRegistry().preview_backfill(candidate))
        except ValueError as error:
            return jsonify({"error": str(error)}), 404
        except RuntimeError:
            return jsonify({"error": "Tag registry is temporarily unavailable."}), 503

    @app.post("/api/tag-vocabulary/candidates/<candidate>/backfill")
    def tag_vocabulary_backfill(candidate: str):
        try:
            return jsonify(TagRegistry().run_backfill(candidate))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except RuntimeError:
            return jsonify({"error": "Tag backfill is temporarily unavailable."}), 503

    @app.get("/api/review/episodes")
    def episodes():
        values = []
        for episode in load_episodes():
            if episode.get("status", {}).get("compiler", {}).get("state") != "review_required":
                continue
            try:
                record = load_review_record(episode["episode_key"])
            except (FileNotFoundError, ValueError):
                continue
            values.append({
                "episode_key": episode["episode_key"],
                "podcast": episode.get("podcast"),
                "title": episode.get("title", episode["episode_key"]),
                "pending_count": len(pending_review_items(record)),
            })
        return jsonify({"episodes": values})

    @app.get("/api/review/episodes/<episode_key>")
    def review(episode_key: str):
        record = load_review_record(episode_key)
        cards = pending_review_items(record)
        presented_cards = [_present_review_item(item) for item in cards]
        progress = _review_progress(record, cards)
        # Count eligible pending cards with audio_pending state (awaiting evidence preparation)
        assisted_unprepared = sum(
            1 for card in presented_cards
            if card.get("assisted_review", {}).get("routing", {}).get("eligible")
            and card.get("assisted_review", {}).get("state") == "audio_pending"
        )
        progress["assisted_unprepared"] = assisted_unprepared
        progress["triage_unavailable"] = sum(
            1 for card in presented_cards
            if isinstance(card.get("triage"), dict)
            and card["triage"].get("status") == "unavailable"
        )
        return jsonify({
            "record": record,
            "cards": presented_cards,
            "progress": progress,
            "recompile_status": recompile_status_for_record(record),
        })

    @app.get("/api/review/episodes/<episode_key>/items/<int:difference_id>/audio")
    def audio(episode_key: str, difference_id: int):
        try:
            episode = _episode(episode_key)
            record = load_review_record(episode_key)
            item = next(item for item in pending_review_items(record) if item.get("id") == difference_id)
            clip, _ = ensure_audio_clip(episode, item, fingerprint=_source_fingerprint(record))
        except (FileNotFoundError, RuntimeError, StopIteration, ValueError) as error:
            message, status = _safe_runtime_error(error, subject="Review audio")
            return jsonify({"error": message}), status
        response = send_file(clip, mimetype="audio/wav", conditional=True)
        response.call_on_close(lambda: clip.unlink(missing_ok=True))
        return response

    @app.post("/api/review/episodes/<episode_key>/items/<int:difference_id>/third-asr")
    def third_asr(episode_key: str, difference_id: int):
        try:
            evidence = ensure_third_asr(_episode(episode_key), difference_id)
        except ThirdAsrInFlight as in_flight:
            # TASK-076 Task 9: a concurrent request already owns this
            # item's claim. This is a non-blocking outcome, not an error:
            # the browser orchestrator (4-way concurrent Third-ASR
            # requests, Task 10) is expected to retry shortly rather than
            # have this endpoint block until the other request finishes.
            return jsonify({
                "status": "in_flight",
                "retry_after_seconds": in_flight.retry_after_seconds,
                "cache_key": in_flight.cache_key,
            }), 202
        except (FileNotFoundError, RuntimeError, ValueError, requests.RequestException) as error:
            message, status = _safe_runtime_error(error, subject="Third ASR")
            return jsonify({"error": message}), status
        return jsonify({"evidence": evidence})

    # TASK-076 Task 12: Assisted review -- whole-selection prepare, per-item
    # execution, canonical progress reload, and atomic Assisted decisions.
    # The strict low-risk Batch approval endpoints above are unrelated and
    # untouched; this is a separate lane end to end.

    @app.post("/api/review/episodes/<episode_key>/assisted/prepare")
    def assisted_prepare(episode_key: str):
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Assisted prepare body is required"}), 400

        selected_ids = body.get("selected_ids")
        generation = body.get("expected_generation_fingerprint")
        session_id = body.get("session_id")
        if (
            not isinstance(selected_ids, list)
            or not selected_ids
            or not isinstance(generation, str)
            or not generation
            or (session_id is not None and (not isinstance(session_id, str) or not session_id))
        ):
            return jsonify({"error": "selected_ids and review generation are required"}), 400

        try:
            record = load_review_record(episode_key)
            source_fingerprint = _source_fingerprint(record)
            prepared_session = begin_assisted_preparation(
                episode_key,
                source_fingerprint,
                selected_ids,
                expected_review_generation_fingerprint=generation,
                session_id=session_id,
            )
            budget = budget_summary(episode_key, source_fingerprint)
        except (BudgetAdmissionError, BudgetIntegrityError, DownstreamReserveError) as error:
            return jsonify({"error": f"Assisted preparation is unavailable: {str(error)[:180]}"}), 402
        except (PrepareSessionError, PrepareSessionConflict, BudgetConcurrencyError) as error:
            return jsonify({"error": str(error)}), 409
        except ValueError as error:
            return jsonify({"error": str(error)}), 409
        except (FileNotFoundError, RuntimeError) as error:
            message, status = _safe_runtime_error(error, subject="Assisted preparation")
            return jsonify({"error": message}), status

        return jsonify({
            "session": prepared_session,
            "budget": _present_budget_summary(budget),
        }), 201

    @app.get("/api/review/episodes/<episode_key>/assisted/prepare/<session_id>")
    def assisted_prepare_status(episode_key: str, session_id: str):
        try:
            record = load_review_record(episode_key)
            source_fingerprint = _source_fingerprint(record)
            loaded_session = load_prepare_session(episode_key, source_fingerprint, session_id)
            budget = budget_summary(episode_key, source_fingerprint)
        except (FileNotFoundError, RuntimeError) as error:
            message, status = _safe_runtime_error(error, subject="Assisted preparation")
            return jsonify({"error": message}), status

        if loaded_session is None:
            return jsonify({"error": f"No prepare-session {session_id!r} found"}), 404

        return jsonify({
            "session": loaded_session,
            "budget": _present_budget_summary(budget),
        })

    @app.post("/api/review/episodes/<episode_key>/assisted/prepare/<session_id>/items/<int:difference_id>")
    def assisted_prepare_item(episode_key: str, session_id: str, difference_id: int):
        body = request.get_json(silent=True) or {}
        attempt_id = body.get("attempt_id")
        attempt_token = attempt_id if isinstance(attempt_id, str) else ""

        try:
            record = load_review_record(episode_key)
            source_fingerprint = _source_fingerprint(record)
            loaded_session = load_prepare_session(episode_key, source_fingerprint, session_id)
            if loaded_session is None:
                return jsonify({"error": f"No prepare-session {session_id!r} found"}), 404
            session_item = loaded_session.get("items", {}).get(str(difference_id))
            if session_item is None:
                return jsonify({
                    "error": f"Item {difference_id} is not part of session {session_id!r}"
                }), 404
            if session_item.get("state") == "prepared":
                # Idempotent replay: a cache hit at admission, or an
                # already-completed execution, needs no provider call.
                return jsonify({"status": "prepared", "item": session_item})

            advance_prepare_session_item(
                episode_key,
                source_fingerprint,
                session_id,
                difference_id,
                state="in_flight",
                attempt_id=attempt_token,
            )
        except PrepareSessionError as error:
            return jsonify({"error": str(error)}), 409
        except ValueError as error:
            return jsonify({"error": str(error)}), 409
        except (FileNotFoundError, RuntimeError) as error:
            message, status = _safe_runtime_error(error, subject="Assisted preparation")
            return jsonify({"error": message}), status

        # This request now exclusively owns execution of this item's one
        # pre-authorized attempt -- the in_flight CAS transition above
        # admits only one concurrent winner, so nothing past this point can
        # double-execute. Release the prepare-session's upfront admission
        # hold before delegating: ensure_third_asr makes its own fresh,
        # fully budget-safe reservation for the real attempt (an initial
        # send plus, if needed, its own one bounded paid retry), so
        # holding both reservations at once would double-count the same
        # spend against the shared cap.
        try:
            release_budget_attempt_pre_send(episode_key, source_fingerprint, attempt_token)
        except ValueError:
            pass  # Already released by an earlier reconcile/replay; harmless.

        try:
            evidence = ensure_third_asr(
                _episode(episode_key), difference_id, prepare_session_id=session_id
            )
        except ThirdAsrInFlight as in_flight:
            advance_prepare_session_item(
                episode_key, source_fingerprint, session_id, difference_id,
                state="retrying", attempt_id=attempt_token,
            )
            return jsonify({
                "status": "in_flight",
                "retry_after_seconds": in_flight.retry_after_seconds,
                "cache_key": in_flight.cache_key,
            }), 202
        except (FileNotFoundError, RuntimeError, ValueError, requests.RequestException) as error:
            advance_prepare_session_item(
                episode_key, source_fingerprint, session_id, difference_id,
                state="failed", attempt_id=attempt_token,
            )
            message, status = _safe_runtime_error(error, subject="Assisted Third ASR")
            return jsonify({"error": message, "status": "failed"}), status

        advance_prepare_session_item(
            episode_key, source_fingerprint, session_id, difference_id,
            state="prepared", attempt_id=attempt_token,
        )
        return jsonify({"status": "prepared", "evidence": evidence})

    @app.post("/api/review/episodes/<episode_key>/assisted/decisions")
    def assisted_decisions(episode_key: str):
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Assisted decision body is required"}), 400

        generation = body.get("expected_generation_fingerprint")
        policy_version = body.get("assisted_policy_version")
        decisions = body.get("decisions")
        if (
            not isinstance(generation, str)
            or not generation
            or not isinstance(policy_version, str)
            or not policy_version
            or not isinstance(decisions, list)
            or not decisions
        ):
            return jsonify({
                "error": "Assisted decisions, policy version, and review generation are required"
            }), 400

        try:
            updated = record_assisted_human_decision_batch(
                episode_key,
                decisions,
                expected_generation_fingerprint=generation,
                assisted_policy_version=policy_version,
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 409

        return jsonify({
            "accepted_count": len(decisions),
            "ready_to_recompile": not pending_review_items(updated),
        })

    @app.post("/api/review/episodes/<episode_key>/items/<int:difference_id>/preview")
    def preview_decision(episode_key: str, difference_id: int):
        body = request.get_json(silent=True) or {}
        if body.get("source") != "custom":
            return jsonify({"error": "Only Custom/Edit requires merge preview"}), 400
        try:
            record = load_review_record(episode_key)
            item = next(
                item
                for item in pending_review_items(record)
                if item.get("id") == difference_id
            )
            preview = preview_custom_edit_details(
                item,
                body.get("text"),
                expand_left_words=body.get("expand_left_words", 0),
                expand_right_words=body.get("expand_right_words", 0),
            )
        except StopIteration:
            return jsonify({"error": "Review item is not pending"}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(preview)


    @app.post("/api/review/episodes/<episode_key>/batch-decision")
    def batch_decision(episode_key: str):
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Batch decision body is required"}), 400

        generation = body.get("expected_generation_fingerprint")
        decisions = body.get("decisions")
        if (
            not isinstance(generation, str)
            or not generation
            or not isinstance(decisions, list)
            or not decisions
        ):
            return jsonify({"error": "Batch decisions and review generation are required"}), 400

        try:
            updated = record_human_decision_batch(
                episode_key,
                decisions,
                expected_generation_fingerprint=generation,
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 409

        return jsonify({
            "accepted_count": len(decisions),
            "ready_to_recompile": not pending_review_items(updated),
        })


    @app.post("/api/review/episodes/<episode_key>/items/<int:difference_id>/decision")
    def decision(episode_key: str, difference_id: int):
        body = request.get_json(silent=True) or {}
        try:
            source = body.get("source")
            decision_kwargs = {
                "source": source,
                "text": body.get("text"),
                "representation_text": body.get("representation_text"),
                "note": body.get("note"),
            }
            if source == "custom":
                decision_kwargs.update(
                    expand_left_words=body.get("expand_left_words", 0),
                    expand_right_words=body.get("expand_right_words", 0),
                )
            elif (
                body.get("expand_left_words", 0) != 0
                or body.get("expand_right_words", 0) != 0
            ):
                raise ValueError(
                    "Edit-range expansion is available only for Custom/Edit"
                )

            result = record_human_decision(
                episode_key,
                difference_id,
                **decision_kwargs,
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        return jsonify({"decision": result, "ready_to_recompile": not pending_review_items(load_review_record(episode_key))})

    @app.post("/api/review/episodes/<episode_key>/recompile")
    def recompile(episode_key: str):
        try:
            _episode(episode_key)
            request_record = request_worker_recompile(
                episode_key,
                requested_by=session.get("reviewer_email", "local-human-review"),
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            return jsonify({"error": str(error)}), 409
        return jsonify({"recompile": request_record}), 202

    return app
