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
from compiler.review_tiers import derive_review_tier

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
    record_materiality_decision_batch,
    record_tier_a_decision_batch,
    request_worker_recompile,
)
from compiler.materiality import materiality_inputs

from .materiality_queue import materiality_groups
from .review_audio import review_clip_window
from .third_asr_prefetch import third_asr_refresh_needed
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
<style>/* TASK-133 review flow */.flow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.6rem;margin:1rem 0 .25rem}.step{background:#fff;border:1px solid #d6dde8;border-radius:12px;padding:.7rem .9rem}.step .n{display:block;font-size:1.5rem;font-weight:650;line-height:1.2}.step .l{display:block;color:#536176;font-size:.9rem}.step.active{border-color:#244f95;box-shadow:0 0 0 2px #244f9533}.step.done{background:#f6f8fb;color:#8a96a8}.quick{border:2px solid #244f95}.quick .exact{font-size:1.1rem}.quick .reading{border:1px solid #d6dde8;border-radius:10px;padding:.75rem}.quick .reading.pick{border:2px solid #244f95;background:#eef3fb}.quick .reading b{display:block;margin-bottom:.35rem}.quick .context{margin:.75rem 0}.actions{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:.85rem 0 0}kbd{display:inline-block;min-width:1.3em;text-align:center;border:1px solid #aeb9c9;border-bottom-width:2px;border-radius:5px;padding:0 .3rem;font:600 .8em ui-monospace,monospace;background:#f6f8fb;color:#172033}details.more,details.advanced{margin-top:1rem}details.more>summary,details.advanced>summary{cursor:pointer;color:#536176;font-weight:600;padding:.6rem 0}details.advanced{border-top:1px solid #d6dde8;padding-top:.5rem;margin-top:2rem}.step-hint{color:#536176;margin:.25rem 0 0}.card>.choices{margin:1rem 0}.quick-audio{display:block;width:100%;margin:.9rem 0 0}.quick-actions{display:flex;flex-wrap:wrap;align-items:center;gap:.6rem;margin:1rem 0 0}.quick-actions button{min-height:40px;padding:.45rem 1.1rem;border-radius:8px}.quick-actions .primary{font-weight:600;padding:.45rem 1.6rem}.choice-group{display:inline-flex;border:1px solid #aeb9c9;border-radius:8px;overflow:hidden}.quick-actions .choice-group button{border:0;border-radius:0;background:#fff;min-width:6.5rem}.quick-actions .choice-group button+button{border-left:1px solid #aeb9c9}.quick-actions .choice-group button.on{background:#244f95;color:#fff}.quick-actions .spacer{flex:1}.quick-actions button.link{border:0;background:none;color:#244f95;padding:.45rem .4rem;text-decoration:underline;text-underline-offset:3px}.quick-actions button.link:hover{color:#172033}.keys{color:#536176;font-size:.85rem;margin:.65rem 0 0}.context-label{color:#536176;font-size:.85rem;margin:.75rem 0 .25rem}.quick .context-label+.context{margin-top:0}mark.span{background:#fff1b8;color:#172033;padding:.05rem .35rem;border-radius:4px;box-shadow:0 0 0 1px #e0c25c;font-weight:600}mark.span.apple{background:#dde7f7;box-shadow:0 0 0 1px #395b9a}mark.span.whisper{background:#d7efe8;box-shadow:0 0 0 1px #0d7b65}mark.span.none{background:#edf1f7;box-shadow:0 0 0 1px #aeb9c9;font-weight:500}.reading.apple{border-left:4px solid #395b9a}.reading.whisper{border-left:4px solid #0d7b65}.quick .reading.apple.pick{border-color:#395b9a;background:#eef3fb}.quick .reading.whisper.pick{border-color:#0d7b65;background:#e8f5f1}.reading.apple>b,.src-apple{color:#244f95}.reading.whisper>b,.src-whisper{color:#0b6654}.quick-actions .choice-group button.on[data-m-source=whisper]{background:#0d7b65}button.primary[data-source=whisper]{background:#0d7b65;border-color:#0d7b65}.source>b{color:#244f95}.source.whisper>b{color:#0b6654}.pick-list{margin:.75rem 0 0;border-top:1px solid #edf1f7}.pick-row{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;padding:.45rem 0;border-bottom:1px solid #edf1f7}.pick-row .spacer{flex:1}button.link{border:0;background:none;color:#244f95;min-height:36px;padding:.3rem .4rem;text-decoration:underline;text-underline-offset:3px;width:auto}button.link:disabled{color:#aeb9c9;text-decoration:none;cursor:default}.reading .tag{font-weight:500;margin-left:.35rem}@media(max-width:520px){.quick-actions .choice-group{display:flex;width:100%}.quick-actions .choice-group button{flex:1;width:auto;min-width:0}.quick-actions .primary{width:100%}.quick-actions .spacer{display:none}.quick-actions button.link{width:auto}.keys{display:none}}@media(max-width:820px){.flow{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><main class=shell><h1>Podcast Human Review</h1><p class=muted>Every decision is explicit and stored in the audit trail. Third ASR is evidence, never an automatic resolution.</p><p><a href="/tags">Manage controlled tag vocabulary</a></p><div id=episode-nav></div><div id=flow class="flow hidden"></div><section id=triage-degraded class="panel hidden"><p class=error>Advisory triage is unavailable for <b id=triage-degraded-count>0</b> pending cards, so batch approval cannot recommend them. Review them individually.</p></section><section id=assisted-availability class="panel hidden"><p class=secondary>Assisted evidence not yet prepared for eligible pending cards: <b id=assisted-count>0</b>. Optional: continue Detailed Review or use Assisted review to prepare evidence.</p></section><section id=materiality class="panel hidden"></section><section id=materiality-settled class="panel hidden"></section><details class=more id=feedback><summary>Notes about this review (saved with your next save)</summary><textarea id=feedback-text placeholder="What was wrong, unclear or annoying? Which card should the filter have handled differently?"></textarea></details><div id=app>Loading…</div><details class=advanced id=advanced><summary>Advanced: third-ASR status, group confirmation (tier A, batch), Assisted review and costs</summary><section id=third-prefetch class="panel hidden"><p class=secondary id=third-prefetch-text></p></section><section id=tier-a class="panel hidden"><h3>Tier A — confirm together</h3><p class=secondary id=tier-summary></p><p class=secondary>Each card below has one proposal backed by evidence: the third voice agrees with one source, Whisper heard nothing where Apple has words, or advisory triage is highly confident. Untick any card you want to review yourself; ticked cards are stored as individual human decisions.</p><div id=tier-a-list></div><button class=primary id=confirm-tier-a>Confirm selected</button></section><section id=batch-actions class="panel hidden"><h3>Batch approval</h3><p class=secondary><span id=batch-count></span> Python-qualified low-risk recommendations can be accepted together. Every accepted item is stored as an individual human decision.</p><button class=primary id=accept-recommended-batch>Accept recommended batch</button></section><section id=assisted-panel class="panel hidden"><h3>Assisted review</h3><p class=secondary>Third-ASR evidence is advisory only. Preparing evidence never chooses a source; every confirmed decision is stored as an individual human decision, separate from Batch approval.</p><div id=assisted-pending></div><div id=assisted-lanes></div><p id=assisted-budget class=secondary></p><p id=assisted-budget-help class=secondary>These figures cover metered AI attempts for this episode's Apple/Whisper source generation, not just this review session. Settled is recorded actual cost; reserved is an outstanding upper-bound hold, including queued preparation, not confirmed spend. Uncertain retains the reserved amount for unverified cost, ambiguous post-send outcomes, over-reservation integrity failure or reconciled legacy Third-ASR; proven pre-send releases count zero. Remaining is the $1.60 total cap less settled, reserved and uncertain; Third-ASR remaining is the shared $0.10 sub-cap less those Third-ASR amounts, not extra budget. These are ledger headroom, not approval for another paid call: downstream reserves can restrict optional Third-ASR, identity reconciliation gates Third-ASR, and integrity failure blocks new reservations. Selected max cost adds this session's quoted per-item maximums, even after a hold is released; session expires marks the preparation lease, not a ledger reset.</p><p><button class=primary id=assisted-confirm disabled>Confirm 0 human decisions</button></p><p class=error id=assisted-error></p></section></details></main>
<script>
let record, cards=[], progress={total:0,reviewed:0,remaining:0}, index=0, busy=false, selection=null, deferredIds=new Set(), episodeList=[], activeEpisodeKey=null, activeEpisodeReason=null, customPreviewReady=false, assistedSelection={}, assistedDecisions={}, assistedTouched={}, assistedSession=null, assistedBudget=null, assistedPreparing=false;
const q=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const TIER_REASONS={two_of_three:'third voice agrees',two_of_three_disputed:'third voice confirms every disputed word',spoken_colloquial_form:'only the spoken form differs (gonna / going to)',colloquial_mixed:'spoken form differs, plus other words',whisper_gap:'Whisper heard nothing here',triage_high:'triage: high confidence',third_new_reading:'third voice heard something else',needs_listening:'needs listening',protected_without_confirmation:'protected, needs confirmation',conflicting_evidence:'evidence conflicts',no_safe_proposal:'no safe proposal',malformed_item:'needs review'};const TIER_RANK={B:0,C:1,A:2};let tierAUnchecked=new Set();
let prefetchState={key:null,running:false,done:0,total:0,failed:0,stopped:null},prefetchFailedIds=new Set();
let allCards=[],detailIds=new Set(),quickPicks={},quickTimes={},quickIndex=0,quickShownId=null,quickShownAt=0,quickSelection=null,fullOpen=false,settledListOpen=false,settledListOpened=false;const pageStartedAt=new Date().toISOString();
function prefetchCandidates(){return cards.filter(c=>{const t=c.review_tier?.tier;return (t==='B'||t==='C')&&(!c.third_asr||c.third_refresh)&&c.audio_window&&!prefetchFailedIds.has(`${record.episode_key}:${c.id}`)})}
function renderPrefetch(){const panel=q('#third-prefetch'),text=q('#third-prefetch-text');if(!panel||!text)return;const s=prefetchState;const show=record&&s.key===record.episode_key&&(s.running||s.stopped||s.failed);panel.classList.toggle('hidden',!show);if(!show)return;text.textContent=s.running?`Fetching the third voice for tier B and C cards: ${s.done} of ${s.total} done${s.failed?`, ${s.failed} failed`:''}. Tier A updates as evidence arrives.`:s.stopped?`Third voice stopped after ${s.done} of ${s.total}: ${s.stopped}`:`Third voice fetched for ${s.done} of ${s.total} cards; ${s.failed} could not be fetched.`}
function reviewerIsActive(){const custom=q('#custom-panel'),audio=q('#audio');return Boolean(selection)||(custom&&!custom.classList.contains('hidden'))||(audio&&!audio.paused)}
async function softRefresh(key){if(busy||reviewerIsActive()||!record||record.episode_key!==key)return;const currentId=currentCard()?.id;const data=await api(`/api/review/episodes/${encodeURIComponent(key)}`);if(busy||reviewerIsActive()||!record||record.episode_key!==key)return;record={...data.record,recompile_status:data.recompile_status};cards=applyCards(sortByTier(data.cards));progress=data.progress;const at=cards.findIndex(c=>c.id===currentId);index=at>=0?at:Math.max(0,nextReviewableIndex(-1));render();updateBatchActions()}
async function prefetchThird(){if(!record)return;const key=record.episode_key;if(prefetchState.running&&prefetchState.key===key)return;const ids=prefetchCandidates().map(c=>c.id);if(!ids.length)return;prefetchState={key,running:true,done:0,total:ids.length,failed:0,stopped:null};renderPrefetch();const queue=ids.map(id=>({id,tries:0}));let sinceRefresh=0;const worker=async()=>{while(queue.length&&prefetchState.key===key&&!prefetchState.stopped){const job=queue.shift();try{const r=await fetch(`/api/review/episodes/${encodeURIComponent(key)}/items/${job.id}/third-asr`,{method:'POST'});if(r.status===202&&job.tries<3){job.tries++;await new Promise(res=>setTimeout(res,4000));queue.push(job);continue}if(r.ok&&r.status!==202){prefetchState.done++;sinceRefresh++}else{let message='';try{message=JSON.parse(await r.text()).error||''}catch(_){}prefetchState.failed++;prefetchFailedIds.add(`${key}:${job.id}`);if([402,403,409,429,503].includes(r.status)||/budget|cap|reconcil|PODCAST_REVIEW_ASR/i.test(message))prefetchState.stopped=message||`HTTP ${r.status}`}}catch(_){prefetchState.failed++;prefetchFailedIds.add(`${key}:${job.id}`)}renderPrefetch();if(sinceRefresh>=6){sinceRefresh=0;try{await softRefresh(key)}catch(_){}}}};await Promise.all([worker(),worker(),worker()]);if(prefetchState.key===key)prefetchState.running=false;renderPrefetch();try{await softRefresh(key)}catch(_){}}
function sortByTier(list){return list.map((c,i)=>[c,i]).sort((a,b)=>((TIER_RANK[a[0].review_tier?.tier]??1)-(TIER_RANK[b[0].review_tier?.tier]??1))||a[1]-b[1]).map(([c])=>c)}
function tierACards(){return cards.filter(c=>c.review_tier?.tier==='A'&&(c.review_tier.source==='apple'||c.review_tier.source==='whisper')&&!deferredIds.has(String(c.id)))}
function tierASelectedIds(){return tierACards().filter(c=>!tierAUnchecked.has(String(c.id))).map(c=>c.id)}
function renderTierA(){const panel=q('#tier-a');if(!panel)return;const list=tierACards();const t=progress.tiers||{};const summary=q('#tier-summary');if(summary)summary.textContent=`A: ${t.A||0} to confirm · B: ${t.B||0} need listening · C: ${t.C||0} protected`;panel.classList.toggle('hidden',!list.length);if(!list.length){q('#tier-a-list').innerHTML='';return}q('#tier-a-list').innerHTML=list.map(c=>{const r=c.review_tier,checked=!tierAUnchecked.has(String(c.id));return `<div class=panel><label><input type=checkbox data-tier-a-id="${Number(c.id)}" ${checked?'checked':''} style="width:auto"> <b>Use ${r.source==='apple'?'Apple':'Whisper'}</b> · <span class=secondary>${esc(TIER_REASONS[r.reason]||r.reason)} · ${esc(c.display?.category||'')}</span></label><div class=sources><div class="source"><b>Apple</b><div class=exact>${esc(c.source_choices?.apple??c.apple_text)}</div></div><div class="source whisper"><b>Whisper</b><div class=exact>${esc(c.source_choices?.whisper??c.whisper_text)}</div></div></div></div>`}).join('');document.querySelectorAll('[data-tier-a-id]').forEach(box=>box.onchange=()=>{const id=String(box.dataset.tierAId);if(box.checked)tierAUnchecked.delete(id);else tierAUnchecked.add(id);updateTierAButton()});updateTierAButton()}
function updateTierAButton(){const button=q('#confirm-tier-a');if(!button)return;const n=tierASelectedIds().length;button.textContent=`Confirm ${n} selected`;button.disabled=busy||!n;button.onclick=confirmTierA}
async function confirmTierA(){if(busy)return;const ids=tierASelectedIds();if(!ids.length)return;const generation=record?.human_review_generation_fingerprint;if(!generation){q('#error').textContent='Review generation is unavailable. Reload before confirming tier A.';return}if(!window.confirm(`Confirm ${ids.length} tier-A proposals? Each card is stored as an individual human decision.`))return;const episodeKey=record.episode_key;setBusy(true);try{await api(`/api/review/episodes/${encodeURIComponent(episodeKey)}/tier-a-decision`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({expected_generation_fingerprint:generation,ids})});tierAUnchecked=new Set();await loadRecord(episodeKey);const activeEpisode=episodeList.find(e=>e.episode_key===episodeKey);if(activeEpisode){activeEpisode.pending_count=cards.length;renderEpisodeNav()}busy=false;render();updateBatchActions()}catch(e){busy=false;render();updateBatchActions();const err=q('#error');if(err)err.textContent=e.message}}
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
function updateBatchActions(){renderTierA();const panel=q('#batch-actions'),button=q('#accept-recommended-batch'),count=q('#batch-count'),decisions=batchRecommendations();if(!panel||!button||!count)return;panel.classList.toggle('hidden',!decisions.length);count.textContent=decisions.length;button.onclick=acceptRecommendedBatch}
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
async function loadRecord(episodeKey=activeEpisodeKey){const switchingEpisode=activeEpisodeKey!==episodeKey||!record;activeEpisodeKey=episodeKey;renderEpisodeNav();index=0;selection=null;if(switchingEpisode){deferredIds=new Set();detailIds=new Set();quickPicks={};quickTimes={};quickIndex=0;quickShownId=null;quickShownAt=0;quickSelection=null;fullOpen=false;settledListOpen=false;settledListOpened=false;tierAUnchecked=new Set();assistedDecisions={};assistedTouched={};assistedSelection={};assistedSession=null;assistedBudget=null}const data=await api(`/api/review/episodes/${activeEpisodeKey}`);record={...data.record,recompile_status:data.recompile_status};cards=applyCards(sortByTier(data.cards));const pendingIds=new Set(cards.map(c=>String(c.id)));deferredIds=new Set([...deferredIds].filter(id=>pendingIds.has(id)));progress=data.progress;const next=nextReviewableIndex(-1);index=next>=0?next:0;render();updateBatchActions();renderAssisted();prefetchThird()}
function replacement(c,source){return Object.hasOwn(c.source_choices||{},source)?c.source_choices[source]:undefined}
function select(source,text){selection={source,text,representationText:null};customPreviewReady=source!=='custom';document.body.classList.add('has-selection');q('#preview').textContent=text;q('#error').textContent='';q('#confirm-selection').disabled=!customPreviewReady;const representation=q('#representation-text');if(representation){const eligible=source==='apple'||source==='whisper';representation.disabled=!eligible;representation.value=eligible?text:'';q('#representation-help').textContent=eligible?'Optional case-only edit. Words, spacing, and punctuation must stay exactly the same.':'Choose Apple or Whisper first to adjust representation/casing.'}document.querySelectorAll('[data-source],#custom-open').forEach(button=>button.classList.toggle('selected',button.dataset.source===source||(source==='custom'&&button.id==='custom-open')))}
function recompileStatus(){return record?.recompile_status}
function recompileActive(){return Boolean(recompileStatus())}
function render(){document.body.classList.toggle('has-selection',Boolean(selection));renderMateriality();const app=q('#app');const availabilityPanel=q('#assisted-availability');if(availabilityPanel){const count=progress.assisted_unprepared||0;availabilityPanel.classList.toggle('hidden',count===0);const countEl=q('#assisted-count');if(countEl)countEl.textContent=count}const triageDegraded=q('#triage-degraded');if(triageDegraded){const count=progress.triage_unavailable||0;triageDegraded.classList.toggle('hidden',count===0);const countEl=q('#triage-degraded-count');if(countEl)countEl.textContent=count}if(!cards.length&&(quickCards().length||settledCards().length||pickCount())){activeEpisodeReason=null;renderEpisodeNav();app.innerHTML='<section class=card><p>Full review is done. The remaining cards are in the panels above.</p></section>';return}if(!cards.length){activeEpisodeReason=null;renderEpisodeNav();const status=recompileStatus(),active=Boolean(status),label=status==='unknown'?'Recompile status unknown…':status==='completed'?'Recompile completed':active?'Recompile started…':'Recompile &amp; continue';app.innerHTML=`<section class=card><h2>Human review complete</h2><p>All decisions are stored in the audit trail. <b>Recompile &amp; continue</b> starts the normal worker, which rebuilds the compiled transcript using those audited decisions. If it succeeds, the normal knowledge pipeline continues automatically.</p><button class=primary id=recompile ${active?'disabled':''}>${label}</button><p class=error id=error></p></section>`;if(!active)q('#recompile').onclick=recompile;return}if(!fullOpen&&(quickCards().length||pickCount()||settledCards().length)){activeEpisodeReason=null;renderEpisodeNav();app.innerHTML=`<section class=card><h3>3. Full review: ${cards.length} ${cards.length===1?'card':'cards'}</h3><p class=secondary>Cards to listen to and decide: numbers that differ and material cards without a proposal. They come after the quick cards and the settled cards.</p><div class=quick-actions><button id=open-full>Open full review now</button></div></section>`;q('#open-full').onclick=()=>{fullOpen=true;render()};return}const reviewable=reviewableIndices();if(!reviewable.length){activeEpisodeReason='Deferred items remain';renderEpisodeNav();app.innerHTML=`<section class=card><h2>All remaining cards are deferred</h2><p>These ${cards.length} unresolved card${cards.length===1?' remains':'s remain'} pending. Return to a deferred card and confirm a real decision before Recompile &amp; continue can become available.</p></section>${deferredListHtml()}`;wireDeferredControls();return}if(deferredIds.has(String(currentCard()?.id)))index=nextReviewableIndex(index-1);activeEpisodeReason=currentCard()?.display?.reason||null;renderEpisodeNav();const c=currentCard(),f=c.focus||{},partial=f.scope==='partial',third=c.third_asr;const conflict=partial?`<section class=panel><h3>Focused conflict</h3><div class=sources><div class="source"><b>Apple</b><div class=exact>${esc(f.apple_text)}</div></div><div class="source whisper"><b>Whisper</b><div class=exact>${esc(f.whisper_text)}</div></div></div></section>`:`<section class=panel><h3>Transcript difference</h3><div class=sources><div class="source"><b>Apple</b><div class=exact>${esc(c.apple_text)}</div></div><div class="source whisper"><b>Whisper</b><div class=exact>${esc(c.whisper_text)}</div></div></div></section>`;const suggestion=c.suggestion?.source?`${esc(c.suggestion.source)} — ${esc(c.suggestion.reason||'source-backed compiler preference')}`:'No automatic resolution';const triage=c.triage;const triagePanel=triage?`<section class=panel><h3>Advisory triage</h3><p><b>Recommendation:</b> ${esc(triage.recommendation)}${triage.confidence?` · <b>Confidence:</b> ${esc(triage.confidence)}`:''}</p><p class=secondary>${esc(triage.reason)}</p></section>`:'';const anomaly=c.anomaly;const anomalyPanel=anomaly?`<section class=panel><h3>Review anomaly</h3><p><b>${esc(anomaly.kind)}</b></p><p class=secondary>${esc(anomaly.reason)}</p></section>`:'';const audio=c.audio_window?`${c.audio_window.start.toFixed(1)}–${c.audio_window.end.toFixed(1)} seconds`:'Clip around the Whisper timestamp';const thirdAction=third?'<p class=secondary>Evidence cached.</p>':'<button class=external id=third-run>Run third ASR</button>';app.innerHTML=`<section class=card><div class=meta>${detailIds.has(String(c.id))&&c.materiality&&c.materiality.group!=='full'?'<button class=link id=back-quick>← Back to quick cards</button>':''}<b>Reviewed ${progress.reviewed} of ${progress.total}</b><span class=tag>${progress.remaining} remaining</span>${c.review_tier?`<span class=tag>Tier ${esc(c.review_tier.tier)} · ${esc(TIER_REASONS[c.review_tier.reason]||c.review_tier.reason)}</span>`:''}${deferredCards().length?`<span class=tag>${deferredCards().length} deferred this pass</span>`:''}<span class=tag>${esc(c.display.category)}</span><span class=tag>${esc(c.display.severity)}</span></div>${conflict}<section class=panel><h3>Audio evidence</h3><p class=secondary>${esc(audio)}</p><audio id=audio controls src="/api/review/episodes/${encodeURIComponent(record.episode_key)}/items/${c.id}/audio"></audio><p id=audio-error class=audio-error></p></section><section class=panel><h3>Context</h3><div class=sources><div class="source"><b>Apple context</b><div class=context>${esc(c.apple_context)}</div></div><div class="source whisper"><b>Whisper context</b><div class=context>${esc(c.whisper_context)}</div></div></div></section><div class=choices><button class=primary data-source=apple>Use Apple</button><button class=primary data-source=whisper>Use Whisper</button><button class=external data-source=third data-requires-third=true ${c.third_available?'':'disabled'}>Use Third</button><button id=custom-open>Custom/Edit</button><button class=subtle id=defer>Defer for this pass</button></div><section class=panel><h3>Selected replacement</h3><div id=preview class=exact>Select a choice to preview the exact replacement for this review item.</div><p><input id=note placeholder="Optional audit note"></p><button class=primary id=confirm-selection disabled>Confirm selection</button></section>${anomalyPanel}<section class=panel><h3>Third ASR evidence</h3>${third?`<div class=exact>${esc(third.text)}</div><p class=secondary>${esc([third.model,third.duration&&`${third.duration}s`].filter(Boolean).join(' · '))}</p>${c.third_available?`<p><b>Use Third inserts:</b> <span class=exact>${esc(c.third_window)||'(nothing — Third ASR hears no words here)'}</span></p>`:'<p class=secondary>Third ASR could not be aligned to this card; use Custom/Edit if it helps.</p>'}`:'<p class=secondary>Not requested yet.</p>'}${thirdAction}<p id=third-error class=error></p></section><details class=more><summary>More: compiler suggestion, advisory triage, casing</summary><section class=panel><h3>Suggestion / evidence</h3><p>${suggestion}</p></section>${triagePanel}<section class=panel><h3>Representation/casing</h3><p id=representation-help class=secondary>Choose Apple or Whisper first to adjust representation/casing.</p><input id=representation-text placeholder="Optional case-only representation" disabled></section></details>${deferredListHtml()}<section id=custom-panel class="panel hidden"><h3>Custom/Edit replacement</h3><textarea id=text placeholder="Human-authored replacement text"></textarea><h3>Expand edit range</h3><p class=secondary>Optional. Include up to 3 adjacent canonical-source words on either side when the spoken correction crosses the detected conflict boundary.</p><div class=sources><label>Previous words<select id=expand-left><option value=0>0</option><option value=1>1</option><option value=2>2</option><option value=3>3</option></select></label><label>Next words<select id=expand-right><option value=0>0</option><option value=1>1</option><option value=2>2</option><option value=3>3</option></select></label></div><h3>Replacing exactly</h3><div id=custom-replaced-text class=exact>The exact canonical-source slice will appear after preview.</div><h3>Final merged context</h3><div id=custom-final-preview class=exact>Enter a custom replacement to calculate the exact final context.</div><p id=custom-preview-error class=error></p></section><p class=error id=error></p></section>`;q('#audio').onerror=()=>q('#audio-error').textContent='Audio clip could not be loaded. You can still review this card.';const thirdRun=q('#third-run');if(thirdRun)thirdRun.onclick=runThird;document.querySelectorAll('[data-source]').forEach(button=>button.onclick=()=>{const text=button.dataset.source==='third'?(c.third_available?c.third_window:undefined):replacement(c,button.dataset.source);if(text===undefined){q('#error').textContent=`No replacement is available for ${button.dataset.source}.`;return}select(button.dataset.source,text)});q('#custom-open').onclick=()=>{q('#custom-panel').classList.remove('hidden');q('#text').focus();previewCustom()};q('#text').oninput=previewCustom;q('#expand-left').onchange=previewCustom;q('#expand-right').onchange=previewCustom;q('#representation-text').oninput=()=>{if(!selection||(selection.source!=='apple'&&selection.source!=='whisper'))return;selection.representationText=q('#representation-text').value;q('#preview').textContent=selection.representationText};q('#confirm-selection').onclick=confirmSelection;q('#defer').onclick=deferCurrent;const bq=q('#back-quick');if(bq)bq.onclick=()=>backToQuick(c.id);wireDeferredControls()}
// TASK-126: keyboard shortcuts for the card stream. The third reading is not
// pre-selected: on historical decisions it was right on only about 8 of 26
// cards where it differed from both sources.
function typingTarget(target){const tag=(target?.tagName||'').toLowerCase();return tag==='input'||tag==='textarea'||tag==='select'||target?.isContentEditable}
// TASK-133: materiality groups -- control sample, "click one" and proposals one card at a time, settled cards accepted with one explicit click.
const M_ORDER=['sample','click_one','proposal'];
const M_LABEL={sample:'Control sample',click_one:'Click one',proposal:'Confirm the proposal'};
const M_HELP={sample:'The filter settled this card on its own. Check its choice.',click_one:'Does not change the note, but no evidence decided which reading stays. Pick one and move on.',proposal:'Changes the note. The filter’s proposal is preselected: confirm it, switch, or open the detailed review to edit.'};
const M_STEP={third_asr:'third ASR agrees',registry_term:'registry term',judge:'judge 3/3',fuller:'fuller reading',advertisement:'without the ad',compiler_suggestion:'same text, written differently',number_gap:'one source has nothing here'};
function mGroup(c){return c?.materiality?.group||'full'}
function applyCards(list){allCards=list;return list.filter(c=>mGroup(c)==='full'||detailIds.has(String(c.id)))}
function pickCount(){return Object.keys(quickPicks).length}
function quickCards(){return M_ORDER.flatMap(g=>allCards.filter(c=>mGroup(c)===g&&!detailIds.has(String(c.id))))}
function quickCurrent(){const list=quickCards();return quickIndex<list.length?list[Math.max(0,quickIndex)]:null}
function settledCards(){return allCards.filter(c=>mGroup(c)==='settled'&&!detailIds.has(String(c.id)))}
function sampleLeftCount(){return allCards.filter(x=>mGroup(x)==='sample'&&!detailIds.has(String(x.id))).length}
function mName(s){return s==='apple'?'Apple':'Whisper'}
function mTag(s){return `<b class=src-${s}>${mName(s)}</b>`}
function plural(n,one,many){return `${n} ${n===1?one:many}`}
function pickFor(c){return quickPicks[c.id]||(mGroup(c)==='click_one'?null:c.materiality.source)}
// Active time only: counted while the page is visible and focused; a gap without any input or audio counts at most 30 s.
const QUICK_IDLE_MS=30000;
function pageActive(){return document.visibilityState==='visible'&&document.hasFocus()}
function tickCard(){const c=quickCurrent(),now=Date.now();if(c&&quickShownId===c.id&&quickShownAt){quickTimes[c.id]=(quickTimes[c.id]||0)+Math.min(now-quickShownAt,QUICK_IDLE_MS)/1000}quickShownAt=pageActive()?now:0}
function leaveCard(){tickCard();quickShownId=null;quickShownAt=0}
['keydown','mousemove','mousedown','wheel','scroll','touchstart','blur'].forEach(name=>window.addEventListener(name,tickCard,{passive:true}));document.addEventListener('visibilitychange',tickCard);window.addEventListener('focus',()=>{if(quickShownId!==null)quickShownAt=Date.now()});
function quickContext(c,pick){const mc=c.materiality_context;if(!mc)return esc(c.whisper_context||c.apple_context||'');const text=pick?((c.source_choices||{})[pick]||''):null;const mark=pick===null||pick===undefined?'<i>Apple or Whisper?</i>':(text?esc(text):'<i>(nothing)</i>');return `${mc.left?'… '+esc(mc.left.trim())+' ':''}<mark class="span ${pick||'none'}">${mark}</mark>${mc.right?' '+esc(mc.right.trim())+' …':''}`}
function feedbackText(){const el=q('#feedback-text');return el?el.value.trim().slice(0,2000):''}
function renderFlow(){const flow=q('#flow');if(!flow)return;if(!record||!allCards.length){flow.classList.add('hidden');flow.innerHTML='';return}const quick=quickCards().length,settled=settledCards().length,full=cards.length;const steps=[['Quick cards',quick,quick?`${plural(pickCount(),'card','cards')} ready to save`:'sample, click one, proposals'],['Settled by the filter',settled,sampleLeftCount()?'after the control sample':'accept with one click'],['Full review',full,'listen and decide'],['Recompile',0,'when everything is done']];const active=steps.findIndex(([,n])=>n>0);flow.classList.remove('hidden');flow.innerHTML=steps.map(([label,n,hint],i)=>{const isLast=i===3,state=(active===-1&&isLast)||i===active?'active':(active===-1||i<active)?'done':'';return `<div class="step ${state}"><span class=l>${i+1}. ${label}</span><span class=n>${isLast?(active===-1?'ready':'—'):(n?n:'✓')}</span><span class=l>${hint}</span></div>`}).join('')}
function renderMateriality(){renderFlow();const panel=q('#materiality'),settledPanel=q('#materiality-settled');if(!panel||!settledPanel)return;const list=quickCards();if(quickIndex>list.length)quickIndex=list.length;const c=quickCurrent();panel.classList.toggle('quick',list.length>0);if(!list.length){panel.classList.add('hidden');panel.innerHTML=''}else if(!c){panel.classList.remove('hidden');const rows=list.map((x,i)=>{const s=quickPicks[x.id],changed=s&&x.materiality.source&&s!==x.materiality.source;return `<div class=pick-row><span class=tag>${M_LABEL[mGroup(x)]}</span><span>#${x.id}</span>${s?`<span>${mTag(s)} “${esc((x.source_choices||{})[s])}”</span>${changed?'<span class=tag>changed</span>':''}`:'<span class=secondary>no choice yet</span>'}<span class=spacer></span><button class=link data-m-goto=${i}>Open</button></div>`}).join('');panel.innerHTML=`<div class=meta><h3 style="margin:0">Quick cards: check and save</h3><span class=tag>${pickCount()} of ${list.length} ready</span></div><p class=step-hint>Nothing is saved yet. Open any card to change it, then save. Cards without a choice stay in the queue.</p><div class=pick-list>${rows}</div><div class=quick-actions><button class=primary id=m-save ${pickCount()?'':'disabled'}>Save ${plural(pickCount(),'decision','decisions')}</button><button class=link id=m-back>Back to the last card</button></div><p class=keys><kbd>Enter</kbd> save · <kbd>←</kbd> back</p><p class=error id=m-error></p>`}else{panel.classList.remove('hidden');if(quickShownId!==c.id){quickShownId=c.id;quickShownAt=pageActive()?Date.now():0}const g=mGroup(c),m=c.materiality,src=c.source_choices||{},pick=pickFor(c);const box=s=>`<div class="reading ${s} ${pick===s?'pick':''}"><b>${mName(s)}${m.source===s?' <span class=tag>proposal</span>':''}</b><div class=exact>${esc(src[s])||'(nothing)'}</div></div>`;const seg=s=>`<button data-m-source=${s} class="${pick===s?'on':''}">${mName(s)}</button>`;const keys=g==='click_one'?'<kbd>1</kbd> Apple · <kbd>2</kbd> Whisper · <kbd>←</kbd> back · <kbd>→</kbd> skip · <kbd>E</kbd> details · <kbd>Space</kbd> audio':'<kbd>1</kbd>/<kbd>2</kbd> choose · <kbd>Enter</kbd> next · <kbd>←</kbd> back · <kbd>→</kbd> skip · <kbd>E</kbd> details · <kbd>Space</kbd> audio';panel.innerHTML=`<div class=meta><h3 style="margin:0">${M_LABEL[g]}</h3><span class=tag>card ${quickIndex+1} of ${list.length}</span><span class=tag>${pickCount()} ready</span>${quickPicks[c.id]?'<span class=tag>chosen</span>':''}</div><p class=step-hint>${M_HELP[g]}${m.source?` Filter: ${mTag(m.source)} (${esc(M_STEP[m.step]||m.step||'')}).`:''}</p><p class=context-label>In the transcript${pick?` (with ${mName(pick)})`:''}:</p><div class=context>${quickContext(c,pick)}</div><div class=sources>${box('apple')}${box('whisper')}</div><audio id=m-audio class=quick-audio controls preload=none src="/api/review/episodes/${encodeURIComponent(record.episode_key)}/items/${c.id}/audio"></audio><div class=quick-actions><div class=choice-group>${seg('apple')}${seg('whisper')}</div><button class=primary id=m-next ${pick?'':'disabled'}>Next</button><span class=spacer></span><button class=link id=m-back ${quickIndex?'':'disabled'}>Back</button><button class=link id=m-skip>Skip</button><button class=link id=m-review>Check &amp; save (${pickCount()})</button><button class=link id=m-detail>Open detailed review</button></div><p class=keys>${keys}</p><p class=error id=m-error></p>`}const audio=q('#m-audio');if(audio)audio.ontimeupdate=tickCard;panel.querySelectorAll('[data-m-source]').forEach(b=>b.onclick=()=>quickPick(b.dataset.mSource));panel.querySelectorAll('[data-m-goto]').forEach(b=>b.onclick=()=>quickGo(Number(b.dataset.mGoto)));const bind=(sel,fn)=>{const el=q(sel);if(el)el.onclick=fn};bind('#m-next',quickNext);bind('#m-back',quickBack);bind('#m-skip',quickSkip);bind('#m-review',()=>quickGo(quickCards().length));bind('#m-detail',quickDetail);bind('#m-save',saveStaged);const settled=settledCards();settledPanel.classList.toggle('hidden',!settled.length);if(!settled.length){settledPanel.innerHTML='';return}const sampleLeft=sampleLeftCount();const rows=settled.map(x=>{const s=x.materiality.source,o=s==='apple'?'whisper':'apple',sc=x.source_choices||{};return `<p class=secondary>#${x.id} · ${esc(M_STEP[x.materiality.step]||x.materiality.step||'')}: ${mTag(s)} “${esc(sc[s])}” instead of “${esc(sc[o])}”</p>`}).join('');settledPanel.innerHTML=`<h3>The filter settled ${plural(settled.length,'card','cards')} without you</h3><p class=secondary>They do not change the note. Each reading was chosen from evidence (third ASR, judge, registry terms, fuller reading), never by source. One click stores them as your decisions with the filter’s reason.</p><details id=settled-list ${settledListOpen?'open':''}><summary>Show the list</summary>${rows}</details><div class=quick-actions><button class=primary id=m-accept-settled ${sampleLeft?'disabled':''}>Accept ${settled.length}</button>${sampleLeft?`<span class=secondary>Save the control sample first (${sampleLeft} left).</span>`:''}</div><p class=error id=m-settled-error></p>`;q('#settled-list').ontoggle=e=>{settledListOpen=e.target.open;if(e.target.open)settledListOpened=true};q('#m-accept-settled').onclick=acceptSettled}
function quickGo(i){if(busy)return;leaveCard();quickIndex=Math.max(0,Math.min(i,quickCards().length));quickSelection=null;renderMateriality()}
function quickPick(source){const c=quickCurrent();if(!c||busy)return;quickPicks[c.id]=source;if(mGroup(c)==='click_one')quickGo(quickIndex+1);else renderMateriality()}
function quickNext(){if(busy)return;const c=quickCurrent();if(!c){if(pickCount())saveStaged();return}const s=pickFor(c);if(!s)return;quickPicks[c.id]=s;quickGo(quickIndex+1)}
function quickBack(){quickGo(quickIndex-1)}
function quickSkip(){if(quickCurrent())quickGo(quickIndex+1)}
function quickDetail(){const c=quickCurrent();if(!c||busy)return;leaveCard();detailIds.add(String(c.id));delete quickPicks[c.id];fullOpen=true;cards=applyCards(allCards);index=Math.max(0,cards.findIndex(x=>x.id===c.id));quickSelection=null;selection=null;render();updateBatchActions()}
function backToQuick(id){detailIds.delete(String(id));cards=applyCards(allCards);const at=quickCards().findIndex(x=>String(x.id)===String(id));quickIndex=at>=0?at:0;index=0;selection=null;render();updateBatchActions()}
async function saveStaged(){if(!pickCount()||busy)return;leaveCard();const decisions=quickCards().filter(c=>quickPicks[c.id]).map(c=>({id:c.id,source:quickPicks[c.id],seconds:Math.round(quickTimes[c.id]||0)}));await sendMateriality(decisions,'#m-error',()=>{quickPicks={};quickTimes={};quickIndex=0})}
async function acceptSettled(){const list=settledCards();if(!list.length||busy||sampleLeftCount())return;await sendMateriality(list.map(c=>({id:c.id,source:c.materiality.source})),'#m-settled-error')}
async function sendMateriality(decisions,errorSel,onOk){const generation=record?.human_review_generation_fingerprint;if(!generation){const el=q(errorSel);if(el)el.textContent='Review generation is unavailable. Reload the page.';return}const episodeKey=record.episode_key;const session={started_at:pageStartedAt,settled_list_opened:settledListOpened,note:feedbackText()};setBusy(true);try{await api(`/api/review/episodes/${encodeURIComponent(episodeKey)}/materiality-decision`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({expected_generation_fingerprint:generation,decisions,session})});if(onOk)onOk();const fb=q('#feedback-text');if(fb)fb.value='';quickSelection=null;busy=false;await loadRecord(episodeKey);const e=episodeList.find(x=>x.episode_key===episodeKey);if(e){e.pending_count=allCards.length;renderEpisodeNav()}}catch(e){busy=false;setBusy(false);render();const el=q(errorSel);if(el)el.textContent=e.message}}
window.addEventListener('beforeunload',event=>{if(pickCount()){event.preventDefault();event.returnValue=''}});
document.addEventListener('keydown',event=>{if(event.metaKey||event.ctrlKey||event.altKey||typingTarget(event.target)||busy)return;if(!quickCards().length)return;const c=quickCurrent(),k=event.key;let handled=true;if(k==='1'&&c)quickPick('apple');else if(k==='2'&&c)quickPick('whisper');else if(k==='Enter')quickNext();else if(k==='ArrowLeft'||k==='u'||k==='U')quickBack();else if(k==='ArrowRight'&&c)quickSkip();else if((k==='e'||k==='E')&&c)quickDetail();else if(k===' '&&q('#m-audio')){const a=q('#m-audio');if(a.paused)a.play().catch(()=>{});else a.pause()}else handled=false;if(handled){event.preventDefault();event.stopImmediatePropagation()}});
document.addEventListener('keydown',event=>{if(event.metaKey||event.ctrlKey||event.altKey||typingTarget(event.target))return;const c=typeof currentCard==='function'?currentCard():null;if(!c||busy)return;const key=event.key;if(key==='1'||key==='2'||key==='3'){const source=key==='1'?'apple':key==='2'?'whisper':'third';const button=document.querySelector(`[data-source=${source}]`);if(button&&!button.disabled){event.preventDefault();button.click()}}else if(key==='Enter'){const confirm=q('#confirm-selection');if(confirm&&!confirm.disabled&&selection){event.preventDefault();confirmSelection()}}else if(key===' '){const audio=q('#audio');if(audio){event.preventDefault();if(audio.paused)audio.play().catch(()=>{});else audio.pause()}}else if(key==='d'||key==='D'){const defer=q('#defer');if(defer){event.preventDefault();defer.click()}}});

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

async function runThird(){const c=currentCard(),button=q('#third-run');setBusy(true);button.textContent='Running third ASR…';try{const r=await api(`/api/review/episodes/${record.episode_key}/items/${c.id}/third-asr`,{method:'POST'});const cardIdBeforeReload=c.id;const data=await api(`/api/review/episodes/${record.episode_key}`);record={...data.record,recompile_status:data.recompile_status};cards=applyCards(sortByTier(data.cards));const pendingIds=new Set(cards.map(c=>String(c.id)));deferredIds=new Set([...deferredIds].filter(id=>pendingIds.has(id)));progress=data.progress;const cardIndex=cards.findIndex(card=>card.id===cardIdBeforeReload);if(cardIndex>=0){index=cardIndex}else{const next=nextReviewableIndex(-1);index=next>=0?next:0}busy=false;render();updateBatchActions();renderAssisted()}catch(e){setBusy(false);button.textContent='Run third ASR';q('#third-error').textContent=e.message}}
async function acceptRecommendedBatch(){if(busy)return;const decisions=batchRecommendations().map(({id,source})=>({id,source}));if(!decisions.length)return;const generation=record?.human_review_generation_fingerprint;if(!generation){q('#error').textContent='Review generation is unavailable. Reload before batch approval.';return}if(!window.confirm(`Accept ${decisions.length} low-risk recommendations? Each item will be recorded as an individual human decision.`))return;const episodeKey=record.episode_key;setBusy(true);try{await api(`/api/review/episodes/${encodeURIComponent(episodeKey)}/batch-decision`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({expected_generation_fingerprint:generation,decisions})});await loadRecord(episodeKey);const activeEpisode=episodeList.find(e=>e.episode_key===episodeKey);if(activeEpisode){activeEpisode.pending_count=cards.length;renderEpisodeNav()}busy=false;render();updateBatchActions()}catch(e){busy=false;render();updateBatchActions();const error=q('#error');if(error)error.textContent=e.message}}
async function confirmSelection(){if(busy||!selection)return;const c=currentCard();if(selection.source==='custom'&&!selection.text.trim()){q('#error').textContent='Custom/Edit requires non-empty replacement text.';return}if(selection.source==='custom'&&!customPreviewReady){q('#error').textContent='Custom/Edit requires an exact final merged-context preview before confirmation.';return}const body={source:selection.source,note:q('#note').value||null};if(selection.source==='custom'){body.text=selection.text;body.expand_left_words=Number(q('#expand-left')?.value||0);body.expand_right_words=Number(q('#expand-right')?.value||0)}if((selection.source==='apple'||selection.source==='whisper')&&selection.representationText!==null&&selection.representationText!==selection.text)body.representation_text=selection.representationText;setBusy(true);try{await api(`/api/review/episodes/${record.episode_key}/items/${c.id}/decision`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});deferredIds.delete(String(c.id));const wasEligibleAudioPending=c.assisted_review?.routing?.eligible&&c.assisted_review?.state==='audio_pending';cards.splice(index,1);allCards=allCards.filter(x=>x.id!==c.id);progress={...progress,reviewed:progress.reviewed+1,remaining:cards.length};if(wasEligibleAudioPending&&progress.assisted_unprepared>0){progress.assisted_unprepared--}const activeEpisode=episodeList.find(e=>e.episode_key===record.episode_key);if(activeEpisode){activeEpisode.pending_count=cards.length;renderEpisodeNav()}const next=nextReviewableIndex(index-1);index=next>=0?next:0;selection=null;busy=false;render();updateBatchActions();renderAssisted()}catch(e){setBusy(false);q('#error').textContent=e.message}}
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
<main class=card><h1>Podcast Human Review</h1><p>Sign in with the allowed Google account.</p><div id=google></div><p class=error id=error></p></main>
<script src="https://accounts.google.com/gsi/client" async></script>
<script>
const clientId={{ client_id|tojson }};
async function credential(response){let result;try{result=await fetch('/auth/google',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({credential:response.credential})})}catch(_){document.querySelector('#error').textContent='Network request failed.';return}if(result.ok){location.assign('/');return}const text=await result.text();try{const body=JSON.parse(text);document.querySelector('#error').textContent=body.error||'Prijava nije uspjela.'}catch(_){document.querySelector('#error').textContent=`Zahtjev nije uspio (${result.status}): ${result.statusText||'Unexpected response'}`}}
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
    # TASK-127: the card's review tier and tier-A proposal, recomputed fresh.
    presented["review_tier"] = derive_review_tier(item)
    try:
        presented["audio_window"] = review_clip_window(item)
    except (TypeError, ValueError):
        presented["audio_window"] = None
    # TASK-126: unanchored evidence from a shorter clip is fetched once more.
    presented["third_refresh"] = third_asr_refresh_needed(item)
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
            return jsonify({"error": "Sign in with the allowed Google account."}), 401
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
            return jsonify({"error": "Google sign-in is not enabled."}), 404
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
            return jsonify({"error": "This Google account has no access."}), 403
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
        # TASK-133: the materiality filter's group and reading for each card.
        groups = materiality_groups(record, episode_key, cards)
        for card, item in zip(presented_cards, cards):
            card["materiality"] = groups.get(card.get("id"))
            # The words around the disputed span, so the page can show exactly
            # where the chosen reading goes in the transcript.
            inputs = materiality_inputs(item)
            card["materiality_context"] = (
                {"left": inputs["left"][-220:], "right": inputs["right"][:220]} if inputs else None
            )
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
        progress["tiers"] = {
            tier: sum(1 for card in presented_cards if card["review_tier"]["tier"] == tier)
            for tier in ("A", "B", "C")
        }
        progress["materiality"] = {
            group: sum(1 for value in groups.values() if value["group"] == group)
            for group in ("settled", "sample", "click_one", "proposal", "full")
        }
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


    @app.post("/api/review/episodes/<episode_key>/tier-a-decision")
    def tier_a_decision(episode_key: str):
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Tier-A confirmation body is required"}), 400
        generation = body.get("expected_generation_fingerprint")
        ids = body.get("ids")
        if not isinstance(generation, str) or not generation or not isinstance(ids, list) or not ids:
            return jsonify({"error": "Tier-A card IDs and review generation are required"}), 400
        try:
            updated = record_tier_a_decision_batch(
                episode_key,
                ids,
                expected_generation_fingerprint=generation,
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 409
        return jsonify({
            "accepted_count": len(ids),
            "ready_to_recompile": not pending_review_items(updated),
        })

    @app.post("/api/review/episodes/<episode_key>/materiality-decision")
    def materiality_decision(episode_key: str):
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Materiality decision body is required"}), 400
        generation = body.get("expected_generation_fingerprint")
        decisions = body.get("decisions")
        if not isinstance(generation, str) or not generation or not isinstance(decisions, list) or not decisions:
            return jsonify({"error": "Decisions and review generation are required"}), 400
        try:
            updated = record_materiality_decision_batch(
                episode_key,
                decisions,
                expected_generation_fingerprint=generation,
                session=body.get("session"),
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
