"""Global Personal Knowledge Tag Registry and minimal artifact index.

Review decisions mutate registry state only.  Explicit backfill may rewrite a
podcast artifact or queue a local operation; cloud code never opens local files.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re
import secrets
from typing import Callable

import yaml
from google.api_core.exceptions import PreconditionFailed

from ..episode_contract import now_iso, paths_for
from ..storage import get_bucket, load_episodes
from .frontmatter import _tag_slug, render

REGISTRY_PATH = "knowledge/tags/registry-v1.json"
ARTIFACT_INDEX_PATH = "knowledge/artifacts/index-v1.json"
AGENT_STATUS_PATH = "knowledge/agents/macos-knowledge-sync-v1.json"
REGISTRY_SCHEMA_VERSION = 2
ARTIFACT_INDEX_SCHEMA_VERSION = 1
AGENT_STATUS_SCHEMA_VERSION = 1
MAX_RETRIES = 3
CATEGORIES = {"domain", "training", "nutrition", "supplements", "recovery", "research", "other"}
SOURCE_TYPES = {"podcast", "book", "article", "youtube", "personal-note", "other"}
LOCAL_SOURCE_TYPES = SOURCE_TYPES - {"podcast"}
_INITIAL_TAGS = {"fitness":"domain","nutrition":"domain","hypertrophy":"training","strength":"training","training-volume":"training","training-frequency":"training","training-intensity":"training","exercise-selection":"training","technique":"training","recovery":"recovery","sleep":"recovery","fatigue":"recovery","protein":"nutrition","calories":"nutrition","energy-balance":"nutrition","fat-loss":"nutrition","creatine":"supplements","caffeine":"supplements","beta-alanine":"supplements","research":"research","research-methodology":"research","evidence-quality":"research"}

def podcast_artifact_id(episode_key: str) -> str:
    if not isinstance(episode_key, str) or not episode_key.strip(): raise ValueError("Podcast artifact requires an episode_key")
    return f"podcast:{episode_key.strip()}"

def new_knowledge_id() -> str: return f"note:{secrets.token_hex(16)}"

def _tags(values: object) -> list[str]:
    result=[]
    for value in values if isinstance(values, list) else []:
        slug=_tag_slug(value)
        if slug and slug not in result: result.append(slug)
    return result

def taxonomy_fingerprint(knowledge_id: str, source_type: str, tags: list[str], topics: list[str] | None=None) -> str:
    """Hash taxonomy fields only; never Markdown body."""
    raw=json.dumps({"knowledge_id":knowledge_id,"source_type":source_type,"tags":_tags(tags),"topics":list(topics or [])},ensure_ascii=False,sort_keys=True,separators=(",",":"))
    return "sha256:"+hashlib.sha256(raw.encode()).hexdigest()

def _ref(artifact_id: str, source_type: str) -> dict:
    if not isinstance(artifact_id,str) or not artifact_id.strip(): raise ValueError("artifact_id is required")
    if source_type not in SOURCE_TYPES: raise ValueError("Unsupported artifact source type")
    return {"artifact_id":artifact_id.strip(),"source_type":source_type}

def _refs(values: object) -> list[dict]:
    out=[]; seen=set()
    for value in values if isinstance(values,list) else []:
        if not isinstance(value,dict) or value.get("artifact_id") in seen: continue
        try: out.append(_ref(value.get("artifact_id"),value.get("source_type"))); seen.add(value["artifact_id"])
        except ValueError: pass
    return out

def _artifact_ids(values: object) -> list[str]:
    out=[]; seen=set()
    for value in values if isinstance(values,list) else []:
        artifact_id=value.strip() if isinstance(value,str) else ""
        if not artifact_id or artifact_id in seen: continue
        out.append(artifact_id); seen.add(artifact_id)
    return out

def _backfill(refs: object=(), target_tag: str|None=None, obsolete_tags: object=()) -> dict:
    refs=_refs(refs); episodes=[r["artifact_id"].removeprefix("podcast:") for r in refs if r["source_type"]=="podcast"]
    return {"status":"pending","total_artifacts":len(refs),"completed_artifact_ids":[],"completed_artifacts":0,"total_episodes":len(episodes),"completed_episodes":0,"local_work_items":[],"last_run_at":None,"last_error":None,"target_tag":target_tag,"obsolete_tags":sorted(set(_tags(obsolete_tags)))}

def _not_needed() -> dict: return _backfill()|{"status":"not_needed"}

def initial_registry() -> dict:
    aliases={"creatine":["creatine supplementation","creatine monohydrate"],"training-volume":["volume","weekly volume"]}
    return {"schema_version":REGISTRY_SCHEMA_VERSION,"updated_at":now_iso(),"canonical_tags":{k:{"category":v,"aliases":aliases.get(k,[])} for k,v in _INITIAL_TAGS.items()},"candidates":{},"review_history":[]}

def initial_artifact_index() -> dict: return {"schema_version":ARTIFACT_INDEX_SCHEMA_VERSION,"updated_at":now_iso(),"artifacts":{},"local_actions":[],"local_candidates":[]}

def initial_agent_status() -> dict:
    return {"schema_version":AGENT_STATUS_SCHEMA_VERSION,"updated_at":now_iso(),"agent":"macos-knowledge-sync","status":"Stale","last_started_at":None,"last_finished_at":None,"last_successful_sync_at":None,"duration_seconds":None,"notes_indexed":0,"waiting_for_adoption":0,"unresolved_tags":0,"pending_backfills":0,"last_run":{"scanned":0,"indexed":0,"changed":0,"backfilled":0,"no_op":0,"errors":0}}

def semantic_projection(registry: dict) -> dict:
    return {"schema_version":registry.get("schema_version"),"canonical_tags":{s:{"category":v.get("category"),"aliases":sorted(v.get("aliases",[]),key=str.casefold)} for s,v in sorted(registry.get("canonical_tags",{}).items()) if isinstance(v,dict)}}

def _is_current_registry(registry: object) -> bool:
    return (
        isinstance(registry, dict)
        and registry.get("schema_version") == REGISTRY_SCHEMA_VERSION
        and isinstance(registry.get("canonical_tags"), dict)
        and isinstance(registry.get("candidates"), dict)
        and isinstance(registry.get("review_history"), list)
        and all(
            not isinstance(entry, dict)
            or (
                "episode_keys" not in entry
                and (
                    not isinstance(entry.get("backfill"), dict)
                    or "completed_episode_keys" not in entry["backfill"]
                )
            )
            for entry in registry["candidates"].values()
        )
    )

class _JsonObject:
    def __init__(self,bucket,path,initial,valid,migrate=lambda x:x): self.bucket,self.path,self.initial,self.valid,self.migrate=bucket,path,initial,valid,migrate
    def _read(self):
        blob=self.bucket.blob(self.path)
        if not blob.exists(): return None,0
        try: value=self.migrate(json.loads(blob.download_as_text(encoding="utf-8"))); blob.reload()
        except (json.JSONDecodeError,UnicodeDecodeError) as error: raise ValueError(f"{self.path} is not valid JSON") from error
        if not self.valid(value): raise ValueError(f"{self.path} schema is unsupported")
        return value,int(blob.generation)
    def load(self):
        value,_=self._read()
        if value is not None: return value
        for _ in range(MAX_RETRIES):
            value=self.initial()
            try: self.bucket.blob(self.path).upload_from_string(json.dumps(value,ensure_ascii=False,indent=2)+"\n",content_type="application/json",if_generation_match=0); return value
            except PreconditionFailed: value,_=self._read();
            if value is not None:return value
        raise RuntimeError(f"{self.path} changed repeatedly while initializing")
    def mutate(self,operation):
        for _ in range(MAX_RETRIES):
            value,generation=self._read()
            if value is None: self.load(); continue
            working=deepcopy(value); result=operation(working); working["updated_at"]=now_iso()
            try: self.bucket.blob(self.path).upload_from_string(json.dumps(working,ensure_ascii=False,indent=2)+"\n",content_type="application/json",if_generation_match=generation); return result
            except PreconditionFailed: continue
        raise RuntimeError(f"{self.path} changed repeatedly; please retry")

class ArtifactIndex:
    """Small GCS index for management metadata only; it rejects Markdown body."""
    def __init__(self,*,bucket=None): self.bucket=bucket or get_bucket(); self.store=_JsonObject(self.bucket,ARTIFACT_INDEX_PATH,initial_artifact_index,lambda x:isinstance(x,dict) and x.get("schema_version")==ARTIFACT_INDEX_SCHEMA_VERSION and isinstance(x.get("artifacts"),dict),self._migrate)
    @staticmethod
    def _migrate(value):
        if not isinstance(value,dict): return value
        value=deepcopy(value); value.setdefault("local_actions",[]); value.setdefault("local_candidates",[])
        return value
    def load(self): return self.store.load()
    def get(self,artifact_id):
        value=self.load()["artifacts"].get(artifact_id); return deepcopy(value) if isinstance(value,dict) else None
    def queue_local_action(self, action: dict) -> dict:
        """Persist an agent-owned local write request without file contents."""
        if not isinstance(action, dict) or action.get("kind") != "adopt": raise ValueError("Unsupported local action")
        clean={key:deepcopy(value) for key,value in action.items() if key in {"kind","artifact_id","knowledge_id","source_type","vault","relative_path","status","created_at"}}
        if not clean.get("artifact_id") or not clean.get("knowledge_id"): raise ValueError("Local adoption action requires an artifact_id")
        def queue(index):
            actions=index.setdefault("local_actions",[])
            existing=next((item for item in actions if item.get("kind")=="adopt" and item.get("artifact_id")==clean["artifact_id"]),None)
            if existing is None: actions.append(clean); existing=clean
            return deepcopy(existing)
        return self.store.mutate(queue)
    def pending_local_actions(self) -> list[dict]:
        return [deepcopy(x) for x in self.load().get("local_actions",[]) if isinstance(x,dict) and x.get("status")=="waiting_for_agent"]
    def report_local_action(self,artifact_id,status,*,error=None):
        if status not in {"completed","failed"}: raise ValueError("Local action status must be completed or failed")
        def report(index):
            action=next((x for x in index.setdefault("local_actions",[]) if x.get("artifact_id")==artifact_id and x.get("kind")=="adopt"),None)
            if action is None: raise ValueError("Local action was not found")
            action.update({"status":status,"completed_at":now_iso(),"last_error":(error or "Local agent reported failure")[:180] if status=="failed" else None})
            return deepcopy(action)
        return self.store.mutate(report)
    def record_candidate(self,record:dict):
        """Index an adoption candidate using management metadata only."""
        if not isinstance(record,dict) or "body" in record or "markdown" in record: raise ValueError("Artifact index accepts management metadata, never Markdown body")
        vault,relative_path=record.get("vault"),record.get("relative_path")
        if not isinstance(vault,str) or not vault or not isinstance(relative_path,str) or not relative_path: raise ValueError("Local candidate requires vault and relative_path")
        source_type=record.get("source_type") or record.get("type")
        if source_type not in LOCAL_SOURCE_TYPES: raise ValueError("Local candidate requires a supported source type")
        candidate_id="candidate:"+hashlib.sha256(f"{vault}\0{relative_path}".encode()).hexdigest()[:24]
        clean={"candidate_id":candidate_id,"source_type":source_type,"title":record.get("title"),"vault":vault,"relative_path":relative_path,"tags":_tags(record.get("tags",[])),"last_seen_at":record.get("last_seen_at") or now_iso(),"status":"waiting_for_adoption"}
        def upsert(index):
            candidates=index.setdefault("local_candidates",[]); existing=next((x for x in candidates if x.get("candidate_id")==candidate_id),None)
            if existing is None: candidates.append(clean); existing=clean
            else: existing.update(clean)
            return deepcopy(existing)
        return self.store.mutate(upsert)
    def local_candidates(self) -> list[dict]:
        return [deepcopy(x) for x in self.load().get("local_candidates",[]) if isinstance(x,dict) and x.get("status")=="waiting_for_adoption"]
    def mark_candidate_adopted(self,vault,relative_path,knowledge_id):
        def mark(index):
            for candidate in index.setdefault("local_candidates",[]):
                if candidate.get("vault")==vault and candidate.get("relative_path")==relative_path:
                    candidate.update({"status":"adopted","knowledge_id":knowledge_id,"adopted_at":now_iso()})
        self.store.mutate(mark)
    def upsert(self,record:dict):
        if not isinstance(record,dict) or "body" in record or "markdown" in record: raise ValueError("Artifact index accepts management metadata, never Markdown body")
        ref=_ref(record.get("artifact_id"),record.get("source_type")); allowed={"artifact_id","source_type","title","vault","relative_path","tags","unresolved_tags","taxonomy_fingerprint","last_seen_at","onboarding_status"}; clean={k:deepcopy(v) for k,v in record.items() if k in allowed}; clean|=ref; clean["tags"]=_tags(clean.get("tags",[])); clean["unresolved_tags"]=_tags(clean.get("unresolved_tags",[])); clean["last_seen_at"]=clean.get("last_seen_at") or now_iso()
        return self.store.mutate(lambda index:(index["artifacts"].__setitem__(ref["artifact_id"],clean),deepcopy(clean))[1])

class KnowledgeAgentStatus:
    """GCS heartbeat for the periodic local executor; never stores note data."""
    STALE_AFTER_SECONDS = 18 * 60 * 60
    def __init__(self,*,bucket=None): self.bucket=bucket or get_bucket(); self.store=_JsonObject(self.bucket,AGENT_STATUS_PATH,initial_agent_status,lambda x:isinstance(x,dict) and x.get("schema_version")==AGENT_STATUS_SCHEMA_VERSION and isinstance(x.get("last_run"),dict))
    def load(self): return self.store.load()
    def report(self,record:dict):
        allowed={"agent","status","last_started_at","last_finished_at","last_successful_sync_at","duration_seconds","notes_indexed","waiting_for_adoption","unresolved_tags","pending_backfills","last_run"}
        clean={key:deepcopy(value) for key,value in record.items() if key in allowed}; clean["agent"]="macos-knowledge-sync"
        return self.store.mutate(lambda status:(status.update(clean),deepcopy(status))[1])
    def presented(self,*,now:datetime|None=None):
        status=self.load(); stamp=status.get("last_started_at") if status.get("status")=="Running" else status.get("last_finished_at")
        try: finished=datetime.fromisoformat(stamp.replace("Z","+00:00")) if isinstance(stamp,str) else None
        except ValueError: finished=None
        current=now or datetime.now(timezone.utc)
        if finished is None or (current-finished.astimezone(timezone.utc)).total_seconds()>self.STALE_AFTER_SECONDS: status["status"]="Stale"
        return status

class TagRegistry:
    def __init__(self,*,bucket=None,episodes_loader:Callable[[],list[dict]]|None=None):
        self.bucket=bucket or get_bucket(); self.episodes_loader=episodes_loader or load_episodes; self.store=_JsonObject(self.bucket,REGISTRY_PATH,initial_registry,_is_current_registry); self.artifact_index=ArtifactIndex(bucket=self.bucket)
    def load(self): return self.store.load()
    def _mutate(self,operation): return self.store.mutate(operation)
    @staticmethod
    def _canonical(registry,value):
        slug=_tag_slug(value)
        if not slug:return None
        if slug in registry["canonical_tags"]:return slug
        return next((name for name,row in registry["canonical_tags"].items() if slug in {_tag_slug(x) for x in row.get("aliases",[])}),None)
    @staticmethod
    def _candidate(registry,slug,category):
        entry=registry["candidates"].get(slug)
        if not isinstance(entry,dict):
            stamp=now_iso(); entry={"category":category if category in CATEGORIES else "other","status":"pending","occurrences":0,"artifact_refs":[],"first_seen_at":stamp,"last_seen_at":stamp,"review":None,"backfill":_not_needed()}; registry["candidates"][slug]=entry
        return entry
    @staticmethod
    def _add_ref(entry,ref):
        entry["artifact_refs"]=_refs(entry.get("artifact_refs",[])+[ref]); entry["occurrences"]=len(entry["artifact_refs"]); entry["last_seen_at"]=now_iso()
    def resolve_and_record_artifact(self,artifact_id,source_type,extracted,*,record_unknown_existing=False):
        ref=_ref(artifact_id,source_type)
        def apply(registry):
            resolved=[]; unknown=[]; recorded=[]
            for value in extracted.get("existing_tags",[]):
                target=self._canonical(registry,value)
                if target and target not in resolved: resolved.append(target)
                elif not target:
                    unknown.append(value); slug=_tag_slug(value)
                    if record_unknown_existing and slug: self._add_ref(self._candidate(registry,slug,"other"),ref); recorded.append(slug)
            for proposal in extracted.get("new_tag_candidates",[])[:2]:
                name=proposal.get("tag") if isinstance(proposal,dict) else proposal; category=proposal.get("category") if isinstance(proposal,dict) else "other"; slug=_tag_slug(name)
                if not slug: continue
                target=self._canonical(registry,slug)
                if target:
                    if target not in resolved:resolved.append(target)
                else: self._add_ref(self._candidate(registry,slug,category),ref); recorded.append(slug)
            return {"topics":list(extracted.get("topics",[])),"people":list(extracted.get("people",[])),"tags":resolved,"tag_candidates":list(dict.fromkeys(recorded)),"unknown_existing_tags":unknown}
        return self._mutate(apply)
    def resolve_and_record(self,episode_key,extracted): return self.resolve_and_record_artifact(podcast_artifact_id(episode_key),"podcast",extracted)
    def record_local_artifact(self,record):
        knowledge_id=record.get("knowledge_id") or record.get("artifact_id"); source_type=record.get("source_type") or record.get("type")
        if not isinstance(knowledge_id,str) or not knowledge_id.startswith("note:"): raise ValueError("Local knowledge artifact requires a note: knowledge_id")
        if source_type not in LOCAL_SOURCE_TYPES: raise ValueError("Local knowledge artifact requires a supported source type")
        tags=_tags(record.get("tags",[])); result=self.resolve_and_record_artifact(knowledge_id,source_type,{"existing_tags":tags,"new_tag_candidates":[],"topics":record.get("topics",[])},record_unknown_existing=True)
        indexed=self.artifact_index.upsert({"artifact_id":knowledge_id,"source_type":source_type,"title":record.get("title"),"vault":record.get("vault"),"relative_path":record.get("relative_path"),"tags":tags,"unresolved_tags":_tags(result["unknown_existing_tags"]),"taxonomy_fingerprint":taxonomy_fingerprint(knowledge_id,source_type,tags,record.get("topics",[])),"last_seen_at":record.get("last_seen_at"),"onboarding_status":"adopted"})
        if record.get("vault") and record.get("relative_path"): self.artifact_index.mark_candidate_adopted(record["vault"],record["relative_path"],knowledge_id)
        return {"artifact":indexed,**result}
    def adopt_local_artifact(self,candidate):
        source_type=candidate.get("source_type") or candidate.get("type")
        if source_type not in LOCAL_SOURCE_TYPES: raise ValueError("Adoption requires a supported local source type")
        vault,relative_path=candidate.get("vault") or "default",candidate.get("relative_path")
        if not isinstance(vault,str) or not vault or not isinstance(relative_path,str) or not relative_path: raise ValueError("Adoption requires vault and relative_path")
        for artifact in self.artifact_index.load()["artifacts"].values():
            if isinstance(artifact,dict) and artifact.get("vault")==vault and artifact.get("relative_path")==relative_path:
                return {"artifact":deepcopy(artifact),"local_work_item":None}
        existing=next((x for x in self.artifact_index.pending_local_actions() if x.get("vault")==vault and x.get("relative_path")==relative_path),None)
        if existing:
            artifact=self.artifact_index.get(existing["artifact_id"])
            return {"artifact":artifact,"local_work_item":existing}
        knowledge_id=new_knowledge_id(); tags=_tags(candidate.get("tags",[])); record={"artifact_id":knowledge_id,"source_type":source_type,"title":candidate.get("title"),"vault":vault,"relative_path":relative_path,"tags":tags,"unresolved_tags":tags,"taxonomy_fingerprint":taxonomy_fingerprint(knowledge_id,source_type,tags,candidate.get("topics",[])),"onboarding_status":"waiting_for_agent"}; artifact=self.artifact_index.upsert(record)
        work=self.artifact_index.queue_local_action({"kind":"adopt","artifact_id":knowledge_id,"knowledge_id":knowledge_id,"source_type":source_type,"vault":artifact.get("vault"),"relative_path":artifact.get("relative_path"),"status":"waiting_for_agent","created_at":now_iso()})
        return {"artifact":artifact,"local_work_item":work}
    def decide(self,candidate,action,*,mapped_to=None,reason=None,reviewed_by=None):
        slug=_tag_slug(candidate)
        if not slug or action not in {"promote","map","reject","reopen"}: raise ValueError("Invalid tag review action")
        def apply(registry):
            entry=registry["candidates"].get(slug)
            if not isinstance(entry,dict):raise ValueError("Candidate was not found")
            previous=entry.get("status"); old=entry.get("mapped_to") if previous=="mapped" else (slug if previous=="promoted" else None); target=None
            if action=="promote": target=slug; registry["canonical_tags"].setdefault(slug,{"category":entry.get("category","other"),"aliases":[]}); entry["status"]="promoted"; entry.pop("mapped_to",None)
            elif action=="map":
                target=self._canonical(registry,mapped_to)
                if not target:raise ValueError("Map target must be an existing canonical tag")
                entry["status"]="mapped"; entry["mapped_to"]=target; aliases=registry["canonical_tags"][target].setdefault("aliases",[])
                if slug!=target and slug not in {_tag_slug(x) for x in aliases}:aliases.append(slug)
            elif action=="reject":entry["status"]="rejected";entry.pop("mapped_to",None)
            else:entry["status"]="pending";entry.pop("mapped_to",None)
            corrective=old is not None and old!=target; obsolete=[old] if corrective else ([slug] if action=="map" else []); entry["backfill"]=_backfill(entry.get("artifact_refs",[]),target,obsolete) if action in {"promote","map"} or corrective else _not_needed(); entry["review"]={"action":action,"reason":reason,"reviewed_at":now_iso(),"reviewed_by":reviewed_by}; history={"timestamp":now_iso(),"candidate":slug,"action":action,"previous_status":previous,"new_status":entry["status"],"mapped_to":target,"reason":reason,"reviewed_by":reviewed_by,"backfill_status":entry["backfill"]["status"]};registry["review_history"].append(history);return {"candidate":slug,"entry":deepcopy(entry),"history":history}
        return self._mutate(apply)
    def list_candidates(self,status=None):
        values=[]
        for slug,entry in self.load()["candidates"].items():
            if status and entry.get("status")!=status:continue
            value={"slug":slug,**deepcopy(entry)}; distribution={}
            for ref in value.get("artifact_refs",[]):distribution[ref["source_type"]]=distribution.get(ref["source_type"],0)+1
            value["source_distribution"]=distribution;values.append(value)
        return sorted(values,key=lambda x:(-x.get("occurrences",0),x["slug"]))
    def history(self):return list(self.load()["review_history"])
    def preview_backfill(self,candidate):
        slug=_tag_slug(candidate); entry=self.load()["candidates"].get(slug)
        if not isinstance(entry,dict):raise ValueError("Candidate was not found")
        episodes={x.get("episode_key"):x for x in self.episodes_loader()}; artifacts=[]
        for ref in entry.get("artifact_refs",[]):
            if ref["source_type"]=="podcast":
                key=ref["artifact_id"].removeprefix("podcast:");episode=episodes.get(key,{}) ;artifacts.append({**ref,"episode_key":key,"podcast":episode.get("podcast"),"title":episode.get("title",key)})
            else:
                index=self.artifact_index.get(ref["artifact_id"]) or {};artifacts.append({**ref,"title":index.get("title",ref["artifact_id"]),"vault":index.get("vault"),"relative_path":index.get("relative_path")})
        return {"candidate":slug,"backfill":deepcopy(entry.get("backfill",{})),"artifacts":artifacts,"episodes":[x for x in artifacts if x["source_type"]=="podcast"]}
    def _update_backfill(self,slug,update):
        def apply(registry):
            entry=registry["candidates"].get(slug)
            if not isinstance(entry,dict):raise ValueError("Candidate was not found")
            update(entry["backfill"]);return deepcopy(entry["backfill"])
        return self._mutate(apply)
    def _rewrite_episode(self,episode,b):
        paths=paths_for(episode["episode_key"]); metadata_blob=self.bucket.blob(paths["summary_metadata"]);body_blob=self.bucket.blob(paths["summary_body"]);summary_blob=self.bucket.blob(paths["summary"])
        if not metadata_blob.exists() or not body_blob.exists():raise FileNotFoundError("Knowledge artifacts are unavailable")
        original=metadata_blob.download_as_text(encoding="utf-8");metadata_blob.reload(); generation=int(metadata_blob.generation);manifest=json.loads(original);metadata=manifest.get("metadata");summary=manifest.get("summary")
        if not isinstance(metadata,dict) or not isinstance(summary,dict):raise ValueError("Knowledge metadata manifest is invalid")
        metadata["tags"]=semantic_tag_transition(metadata.get("tags",[]),b.get("target_tag"),b.get("obsolete_tags",[]))["tags"];final=render(episode,metadata,summary["generated_at"],body_blob.download_as_text(encoding="utf-8"));text=json.dumps(manifest,ensure_ascii=False,indent=2)+"\n"
        if original!=text:metadata_blob.upload_from_string(text,content_type="application/json",if_generation_match=generation)
        if summary_blob.exists():before=summary_blob.download_as_text(encoding="utf-8");summary_blob.reload();sg=int(summary_blob.generation)
        else:before,sg=None,0
        if before!=final:summary_blob.upload_from_string(final,content_type="text/markdown; charset=utf-8",if_generation_match=sg)
    def run_backfill(self,candidate):
        slug=_tag_slug(candidate); preview=self.preview_backfill(slug)
        if preview["backfill"].get("status")=="not_needed":raise ValueError("This decision does not require backfill")
        by_key={x.get("episode_key"):x for x in self.episodes_loader()};self._update_backfill(slug,lambda b:b.update({"status":"running","last_error":None,"last_run_at":now_iso()}))
        for item in preview["artifacts"]:
            aid=item["artifact_id"];current=self.preview_backfill(slug)["backfill"]
            if aid in current.get("completed_artifact_ids",[]):continue
            if item["source_type"]!="podcast":
                def queue(b,item=deepcopy(item)):
                    if not any(x.get("artifact_id")==item["artifact_id"] for x in b.setdefault("local_work_items",[])):b["local_work_items"].append({"kind":"tag_backfill","candidate":slug,"artifact_id":item["artifact_id"],"source_type":item["source_type"],"target_tag":b.get("target_tag"),"obsolete_tags":b.get("obsolete_tags",[]),"vault":item.get("vault"),"relative_path":item.get("relative_path"),"status":"waiting_for_agent","created_at":now_iso()})
                self._update_backfill(slug,queue);continue
            try:
                key=aid.removeprefix("podcast:");episode=by_key.get(key)
                if not episode:raise FileNotFoundError("Episode is unavailable")
                self._rewrite_episode(episode,current)
                def done(b,aid=aid):
                    if aid not in b.setdefault("completed_artifact_ids",[]):b["completed_artifact_ids"].append(aid)
                    b["completed_artifacts"]=len(b["completed_artifact_ids"]);b["completed_episodes"]=len([artifact_id for artifact_id in b["completed_artifact_ids"] if artifact_id.startswith("podcast:")]);b["last_run_at"]=now_iso()
                self._update_backfill(slug,done)
            except Exception as error:return {"candidate":slug,"backfill":self._update_backfill(slug,lambda b:b.update({"status":"failed","last_error":str(error)[:180],"last_run_at":now_iso()}))}
        def finish(b):b.update({"status":"waiting_for_agent" if any(x.get("status")=="waiting_for_agent" for x in b.get("local_work_items",[])) else "completed","completed_artifacts":len(b.get("completed_artifact_ids",[])),"completed_episodes":len([artifact_id for artifact_id in b.get("completed_artifact_ids",[]) if artifact_id.startswith("podcast:")]),"last_error":None,"last_run_at":now_iso()})
        return {"candidate":slug,"backfill":self._update_backfill(slug,finish)}
    def pending_local_work(self):return [deepcopy(w) for c in self.list_candidates() for w in c.get("backfill",{}).get("local_work_items",[]) if w.get("status")=="waiting_for_agent"]
    def pending_adoption_work(self): return self.artifact_index.pending_local_actions()
    def report_adoption_work(self,artifact_id,status,*,error=None): return self.artifact_index.report_local_action(artifact_id,status,error=error)
    def report_local_work(self,candidate,artifact_id,status,*,error=None):
        if status not in {"completed","failed"}:raise ValueError("Local work status must be completed or failed")
        def update(b):
            work=next((x for x in b.get("local_work_items",[]) if x.get("artifact_id")==artifact_id),None)
            if not work:raise ValueError("Local work item was not found")
            work.update({"status":status,"completed_at":now_iso()})
            if status=="completed" and artifact_id not in b.setdefault("completed_artifact_ids",[]):b["completed_artifact_ids"].append(artifact_id)
            b["completed_artifacts"]=len(b["completed_artifact_ids"]);b["last_error"]=(error or "Local agent reported failure")[:180] if status=="failed" else None;b["status"]="failed" if status=="failed" else ("completed" if all(x.get("status")=="completed" for x in b["local_work_items"]) else "waiting_for_agent");b["last_run_at"]=now_iso()
        return self._update_backfill(_tag_slug(candidate),update)

def semantic_tag_transition(tags:object,target_tag:str|None,obsolete_tags:object)->dict:
    old=_tags(tags);result=[tag for tag in old if tag not in set(_tags(obsolete_tags))];target=_tag_slug(target_tag)
    if target and target not in result:result.append(target)
    return {"tags":result,"changed":result!=old}

def patch_local_markdown(markdown:str,*,target_tag:str|None,obsolete_tags:list[str],knowledge_id:str|None=None)->tuple[str,bool]:
    """Re-read helper: patch only tag/ID lines while retaining all other bytes."""
    if not markdown.startswith("---\n"):raise ValueError("Knowledge note must have YAML frontmatter")
    end=markdown.find("\n---",4)
    if end<0:raise ValueError("Knowledge note frontmatter is not closed")
    front,body=markdown[4:end+1],markdown[end+4:]; data=yaml.safe_load(front) or {}
    if not isinstance(data,dict):raise ValueError("Knowledge note frontmatter must be a mapping")
    changed=False
    if knowledge_id and not data.get("knowledge_id"):
        front=f"knowledge_id: {knowledge_id}\n"+front; changed=True
    transition=semantic_tag_transition(data.get("tags",[]),target_tag,obsolete_tags)
    if transition["changed"]:
        replacement="tags:\n"+"".join(f"- {tag}\n" for tag in transition["tags"])
        pattern=r"(?m)^tags:[^\n]*(?:\n[ \t]*-[^\n]*)*(?:\n(?=\S)|\n?$)"
        match=re.search(pattern,front)
        front=(front[:match.start()]+replacement+front[match.end():]) if match else front+replacement
        changed=True
    return ("---\n"+front+"---"+body,True) if changed else (markdown,False)
