import json
import os
import re
import time
import threading
import select
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
import psycopg2
import psycopg2.extras
import boto3
from botocore.config import Config

# ──────────────────────────────────────────────────────────────────────────────
# Load .env for local dev (never commit real secrets). In Docker/EC2, inject env.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)      # <— override any stale shell vars
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG (Environment)
# ──────────────────────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")           # optional (prefer IAM role)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")   # optional
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", "")           # optional (for temp creds)

MODEL_ID = os.getenv("MODEL_ID", "meta.llama3-70b-instruct-v1:0")

LISTEN_CHANNEL = os.getenv("LISTEN_CHANNEL", "meeting_transcript_insert")
CHUNK_CHAR_LIMIT = int(os.getenv("CHUNK_CHAR_LIMIT", "14000"))
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "5"))
BACKFILL_LIMIT = int(os.getenv("BACKFILL_LIMIT", "500"))

# Narrative summary constraints
SUMMARY_MIN_SENTENCES = int(os.getenv("SUMMARY_MIN_SENTENCES", "3"))
SUMMARY_MAX_SENTENCES = int(os.getenv("SUMMARY_MAX_SENTENCES", "12"))
PARA_MIN = int(os.getenv("PARA_MIN", "2"))
PARA_MAX = int(os.getenv("PARA_MAX", "4"))

# Treat [] as "missing" so they get reprocessed
TREAT_EMPTY_LIST_AS_MISSING = os.getenv("TREAT_EMPTY_LIST_AS_MISSING", "true").strip().lower() in ("1","true","yes","on")

# NEW: allow inferred next-steps when none are explicit
ALLOW_INFERRED_ITEMS = os.getenv("ALLOW_INFERRED_ITEMS", "true").strip().lower() in ("1","true","yes","on")
INFER_ITEMS_MIN = int(os.getenv("INFER_ITEMS_MIN", "1"))   # ensure at least N
INFER_ITEMS_MAX = int(os.getenv("INFER_ITEMS_MAX", "3"))   # cap

# Logging
LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "DEBUG").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.DEBUG)
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", os.path.join(LOG_DIR, "app.log"))
LOG_PREVIEW_CHARS = int(os.getenv("LOG_PREVIEW_CHARS", "300"))

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(threadName)s | %(funcName)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(fmt)
    logger = logging.getLogger("fireflies")
    logger.setLevel(LOG_LEVEL)

    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("psycopg2").setLevel(logging.INFO)

    logger.debug("Logger initialized. Log file: %s", LOG_FILE)
    return logger

logger = setup_logging()

# ──────────────────────────────────────────────────────────────────────────────
# Flask + Bedrock client
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
logger.info(
    'Booting | region=%s | table=%s | TREAT_EMPTY_LIST_AS_MISSING=%s | ALLOW_INFERRED_ITEMS=%s',
    AWS_REGION, 'public."myApp_meetingtranscript"', TREAT_EMPTY_LIST_AS_MISSING, ALLOW_INFERRED_ITEMS
)
logger.info("DB host=%s db=%s user=%s", DB_HOST, DB_NAME, DB_USER)

def assert_aws_auth():
    try:
        sess = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID or None,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY or None,
            aws_session_token=AWS_SESSION_TOKEN or None,
            region_name=AWS_REGION,
        )
        who = sess.client("sts").get_caller_identity()
        logger.info("AWS auth OK | account=%s | arn=%s", who.get("Account"), who.get("Arn"))
    except Exception as e:
        logger.error("AWS auth FAILED (bad/expired creds or wrong region). %s", e)
        raise

assert_aws_auth()

bedrock_kwargs = {
    "region_name": AWS_REGION,
    "config": Config(retries={"max_attempts": 8, "mode": "adaptive"}),
}
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    bedrock_kwargs.update({
        "aws_access_key_id": AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    })
    if AWS_SESSION_TOKEN:
        bedrock_kwargs["aws_session_token"] = AWS_SESSION_TOKEN

bedrock = boto3.client("bedrock-runtime", **bedrock_kwargs)
logger.debug("Bedrock client ready for model_id=%s", MODEL_ID)

# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────
def preview(text: str, n: int = LOG_PREVIEW_CHARS) -> str:
    if text is None:
        return "<none>"
    t = text.replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")

def count_sentences(md: str) -> int:
    parts = re.split(r"[.!?]+(?:\s|$)", md.strip())
    return len([p for p in parts if p.strip()])

def count_paragraphs(md: str) -> int:
    paras = [p for p in re.split(r"\n\s*\n", md.strip()) if p.strip()]
    return len(paras)

# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────
def db_conn():
    logger.debug("Opening DB connection …")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
    logger.debug("DB connection established.")
    return conn

def ensure_trigger():
    """Create LISTEN/NOTIFY trigger (idempotent)."""
    sql = """
    CREATE OR REPLACE FUNCTION public.notify_new_transcript()
    RETURNS trigger AS $$
    BEGIN
      PERFORM pg_notify('meeting_transcript_insert', NEW.id::text);
      RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    DROP TRIGGER IF EXISTS trg_notify_new_transcript
      ON public."myApp_meetingtranscript";

    CREATE TRIGGER trg_notify_new_transcript
    AFTER INSERT ON public."myApp_meetingtranscript"
    FOR EACH ROW
    EXECUTE FUNCTION public.notify_new_transcript();
    """
    try:
        with db_conn() as conn, conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
        logger.info("Trigger ensured on channel '%s'.", LISTEN_CHANNEL)
    except Exception:
        logger.exception("Failed to ensure trigger.")

def fetch_row_by_id(row_id: int) -> Optional[Dict[str, Any]]:
    q = """
        SELECT id,
               "userId",
               meetingid,
               transcription,
               summary,
               "actionItem",
               created_at
        FROM public."myApp_meetingtranscript"
        WHERE id = %s
    """
    try:
        with db_conn() as conn, conn.cursor() as cur:
            cur.execute(q, (row_id,))
            row = cur.fetchone()
            logger.debug("fetch_row_by_id(%s) -> %s", row_id, "found" if row else "none")
            return row
    except Exception:
        logger.exception("DB error in fetch_row_by_id(%s)", row_id)
        return None

def fetch_missing(limit: int) -> List[Dict[str, Any]]:
    """
    Rows with empty/NULL summary OR missing actionItem.
    Missing includes:
      - SQL NULL
      - JSONB null
      - {}
      - [] (iff TREAT_EMPTY_LIST_AS_MISSING=True)
    """
    q = """
        SELECT id,
               "userId",
               meetingid,
               transcription,
               summary,
               "actionItem",
               created_at
        FROM public."myApp_meetingtranscript"
        WHERE (summary IS NULL OR btrim(summary) = '')
           OR (
                "actionItem" IS NULL
             OR jsonb_typeof("actionItem") = 'null'
             OR "actionItem"::text = '{}'
             OR (%s AND jsonb_typeof("actionItem") = 'array' AND jsonb_array_length("actionItem") = 0)
           )
        ORDER BY created_at DESC, id DESC
        LIMIT %s
    """
    try:
        with db_conn() as conn, conn.cursor() as cur:
            cur.execute(q, (TREAT_EMPTY_LIST_AS_MISSING, limit))
            rows = cur.fetchall()
            logger.debug(
                "fetch_missing(limit=%s) -> %s rows (empty-list-missing=%s)",
                limit, len(rows), TREAT_EMPTY_LIST_AS_MISSING
            )
            if rows:
                logger.debug("missing ids=%s", [r["id"] for r in rows])
            return rows
    except Exception:
        logger.exception("DB error in fetch_missing(limit=%s)", limit)
        return []

def update_row_partial(row_id: int, summary: Optional[str] = None, action_items: Optional[Any] = None):
    sets, params = [], []
    if summary is not None:
        sets.append("summary = %s")
        params.append(summary)
    if action_items is not None:
        sets.append('"actionItem" = %s')
        params.append(json.dumps(action_items))
    if not sets:
        logger.debug("update_row_partial(%s): nothing to update.", row_id)
        return

    params.append(row_id)
    q = f'UPDATE public."myApp_meetingtranscript" SET {", ".join(sets)} WHERE id = %s'
    try:
        with db_conn() as conn, conn.cursor() as cur:
            cur.execute(q, params)
            conn.commit()
        logger.info("Row %s updated. summary=%s action_items=%s",
                    row_id, summary is not None, action_items is not None)
    except Exception:
        logger.exception("DB error updating row_id=%s", row_id)

# ──────────────────────────────────────────────────────────────────────────────
# LLM prompts
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are a meticulous meeting analyst. Use only information present in the transcript. "
    "Write in neutral, professional tone. Do not invent names, dates, times, owners, or tasks. "
    "If a detail is not explicitly mentioned, leave it out."
)

def mk_prompt_chunk(transcript_chunk: str) -> Dict[str, Any]:
    chat = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_MSG}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "From the transcript CHUNK below, return STRICT JSON with exactly this schema:\n"
        "{\n"
        '  "summary_md": string,  // multi-paragraph narrative with '
        f'{PARA_MIN}-{PARA_MAX} paragraphs and {SUMMARY_MIN_SENTENCES}-{SUMMARY_MAX_SENTENCES} total sentences; '
        "NO headings, NO bullet lists; chronological; only facts present.\n"
        '  "action_items": [\n'
        '    {"item": string, "owner": string, "deadline": string}\n'
        "  ]\n"
        "}\n"
        "- Include ONLY clear action items; omit vague ideas. Use \"\" for unknown owner/deadline.\n"
        "Transcript chunk:\n"
        f"{transcript_chunk}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return {"prompt": chat, "max_gen_len": 900, "temperature": 0.2, "top_p": 0.9, "stop": ["<|eot_id|>"]}

def mk_prompt_merge(summaries: List[str], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    summaries_join = "\n\n---\n\n".join(summaries)
    items_join = json.dumps(items, ensure_ascii=False)
    chat = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_MSG}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "Merge the CHUNK outputs below into ONE final result. Return STRICT JSON:\n"
        "{\n"
        '  "summary_md": string,\n'
        '  "action_items": [ { "item":string, "owner":string, "deadline":string } ]\n'
        "}\n"
        "- Deduplicate near-duplicates; keep only precise tasks.\n\n"
        "Chunk summaries to merge:\n"
        f"{summaries_join}\n\n"
        "All action items from chunks:\n"
        f"{items_join}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return {"prompt": chat, "max_gen_len": 1100, "temperature": 0.2, "top_p": 0.9, "stop": ["<|eot_id|>"]}

def mk_prompt_rewrite(summary_md: str) -> Dict[str, Any]:
    chat = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "Rewrite into neutral, professional prose.\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Rewrite the following to be {PARA_MIN}-{PARA_MAX} paragraphs and "
        f"{SUMMARY_MIN_SENTENCES}-{SUMMARY_MAX_SENTENCES} sentences total, chronological, no headings/bullets.\n\n"
        f"{summary_md}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return {"prompt": chat, "max_gen_len": 700, "temperature": 0.2, "top_p": 0.9, "stop": ["<|eot_id|>"]}

# NEW: infer items even when none are explicit
def mk_prompt_infer_actions(transcript_text: str, summary_text: str = "") -> Dict[str, Any]:
    context = summary_text.strip() or transcript_text
    chat = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a practical project assistant. When a transcript lacks explicit tasks, "
        "infer 1–3 reasonable next steps based on context. Do NOT invent names or dates; "
        "use empty strings for unknown owner/deadline.\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "From the CONTEXT below, return STRICT JSON:\n"
        "{\n"
        '  "action_items": [ {"item": string, "owner": "", "deadline": ""} ]\n'
        "}\n"
        f"Context:\n{context}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return {"prompt": chat, "max_gen_len": 500, "temperature": 0.3, "top_p": 0.9, "stop": ["<|eot_id|>"]}

# ──────────────────────────────────────────────────────────────────────────────
# Bedrock call
# ──────────────────────────────────────────────────────────────────────────────
def bedrock_invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        resp = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        body = resp["body"].read().decode("utf-8")
        dt = (time.perf_counter() - t0) * 1000
        logger.debug("Bedrock invoke ok in %.0f ms. Body preview=%s", dt, preview(body))
        return json.loads(body)
    except Exception:
        dt = (time.perf_counter() - t0) * 1000
        logger.exception("Bedrock invoke FAILED after %.0f ms", dt)
        raise

def parse_generation_json(generation_text: str) -> Dict[str, Any]:
    try:
        return json.loads(generation_text)
    except Exception:
        m = re.search(r"\{.*\}", generation_text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                logger.debug("parse_generation_json: JSON block failed. Preview=%s", preview(generation_text))
    logger.warning("parse_generation_json: returning empty structure. Preview=%s", preview(generation_text))
    return {"summary_md": "", "action_items": []}

# ──────────────────────────────────────────────────────────────────────────────
# Transcript → summary/action
# ──────────────────────────────────────────────────────────────────────────────
def flatten_transcription(transcription: Any) -> str:
    """Turns JSONB transcript into sequential text with speakers if present."""
    if transcription is None:
        logger.debug("flatten_transcription: None input.")
        return ""
    if isinstance(transcription, str):
        try:
            transcription = json.loads(transcription)
        except Exception:
            logger.debug("flatten_transcription: raw string used. preview=%s", preview(transcription))
            return transcription

    lines: List[str] = []
    if isinstance(transcription, list):
        for it in transcription:
            text = (it or {}).get("text", "")
            spk = (it or {}).get("speaker") or (it or {}).get("name")
            if not spk and (it or {}).get("socket") is not None:
                spk = f"S{(it or {}).get('socket')}"
            line = f"{spk}: {text}" if spk else text
            line = re.sub(r"\s+", " ", (line or "")).strip()
            if line:
                lines.append(line)
        result = "\n".join(lines)
        logger.debug("flatten_transcription: produced %s chars, %s lines", len(result), len(lines))
        return result
    dumped = json.dumps(transcription, ensure_ascii=False)
    logger.debug("flatten_transcription: non-list json -> %s chars", len(dumped))
    return dumped

def normalize_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return [{"item","owner","deadline"}] with dedupe + blanks normalized."""
    out, seen = [], set()
    for it in items or []:
        if not isinstance(it, dict):
            continue
        item = (it.get("item") or it.get("task") or "").strip()
        if not item:
            continue
        owner = it.get("owner")
        owner = "" if owner in (None, "null") else str(owner).strip()
        deadline = it.get("deadline") or it.get("due_date") or it.get("date")
        if deadline is None:
            t = it.get("time")
            deadline = t if isinstance(t, str) else ""
        else:
            deadline = str(deadline).strip()
        norm = re.sub(r"\s+", " ", item.lower())
        if norm in seen:
            continue
        seen.add(norm)
        out.append({"item": item, "owner": owner, "deadline": deadline})
    return out

def summarize_transcript(text: str) -> Dict[str, Any]:
    """Summarization with chunk/merge + bounds + fallback inference."""
    if not text.strip():
        logger.info("summarize_transcript: empty transcript.")
        return {"summary_md": "", "action_items": []}

    logger.debug("summarize_transcript: total chars=%s", len(text))

    if len(text) <= CHUNK_CHAR_LIMIT:
        data = bedrock_invoke(mk_prompt_chunk(text))
        obj = parse_generation_json(data.get("generation", ""))
    else:
        chunks = [text[i:i+CHUNK_CHAR_LIMIT] for i in range(0, len(text), CHUNK_CHAR_LIMIT)]
        chunk_summaries, chunk_items = [], []
        for ch in chunks:
            data = bedrock_invoke(mk_prompt_chunk(ch))
            co = parse_generation_json(data.get("generation", ""))
            if co.get("summary_md"):
                chunk_summaries.append(co["summary_md"])
            if isinstance(co.get("action_items"), list):
                chunk_items.extend(co["action_items"])
        data = bedrock_invoke(mk_prompt_merge(chunk_summaries, chunk_items))
        obj = parse_generation_json(data.get("generation", ""))

    # Enforce paragraph/sentence bounds
    summary = (obj.get("summary_md") or "").strip()
    n_sent = count_sentences(summary)
    n_para = count_paragraphs(summary)
    if (n_sent < SUMMARY_MIN_SENTENCES or n_sent > SUMMARY_MAX_SENTENCES) or (n_para < PARA_MIN or n_para > PARA_MAX):
        data2 = bedrock_invoke(mk_prompt_rewrite(summary))
        summary2 = (data2.get("generation") or "").strip()
        if summary2:
            summary = summary2

    # Normalize items from main pass
    items = normalize_items(obj.get("action_items", []))

    # Fallback: infer 1–3 items when none are explicit
    if ALLOW_INFERRED_ITEMS and len(items) < INFER_ITEMS_MIN:
        logger.info("No explicit items; invoking inferred-items fallback…")
        d = bedrock_invoke(mk_prompt_infer_actions(text, summary))
        obj2 = parse_generation_json(d.get("generation", ""))
        inferred = obj2.get("action_items") or obj2.get("items") or obj2.get("tasks") or []
        items = normalize_items(inferred)
        if INFER_ITEMS_MAX > 0:
            items = items[:INFER_ITEMS_MAX]
        logger.debug("Inferred %s action items.", len(items))

    return {"summary_md": summary, "action_items": items}

# ──────────────────────────────────────────────────────────────────────────────
# Processing
# ──────────────────────────────────────────────────────────────────────────────
def is_items_missing(js: Any) -> bool:
    """Missing when NULL / json null / {} / [] (if configured)."""
    if js is None:
        return True
    try:
        obj = json.loads(js) if isinstance(js, str) else js
    except Exception:
        return False  # unknown text -> treat as filled
    if obj is None:
        return True
    if isinstance(obj, dict) and len(obj) == 0:
        return True
    if TREAT_EMPTY_LIST_AS_MISSING and isinstance(obj, list) and len(obj) == 0:
        return True
    return False

def needs_processing(row: Dict[str, Any]) -> Tuple[bool, bool]:
    need_summary = (row.get("summary") is None) or (str(row.get("summary")).strip() == "")
    need_items = is_items_missing(row.get("actionItem"))
    return need_summary, need_items

def process_row(row_id: int):
    try:
        row = fetch_row_by_id(row_id)
        if not row:
            logger.warning("process_row: row %s not found.", row_id)
            return

        need_summary, need_items = needs_processing(row)
        logger.info("Row %s | meetingid=%s | need_summary=%s need_items=%s",
                    row_id, row.get("meetingid"), need_summary, need_items)
        if not (need_summary or need_items):
            logger.debug("Row %s already complete; skipping.", row_id)
            return

        transcript_txt = flatten_transcription(row.get("transcription"))
        result = summarize_transcript(transcript_txt)

        summary_to_write = result["summary_md"] if need_summary else None
        items_to_write   = result["action_items"] if need_items else None

        update_row_partial(row_id, summary=summary_to_write, action_items=items_to_write)
    except Exception:
        logger.exception("process_row failed for id=%s", row_id)

# ──────────────────────────────────────────────────────────────────────────────
# Background workers
# ──────────────────────────────────────────────────────────────────────────────
def pg_listener():
    """Instant processing on INSERT via LISTEN/NOTIFY."""
    while True:
        try:
            conn = db_conn()
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
            cur.execute(f"LISTEN {LISTEN_CHANNEL};")
            logger.info("LISTEN on channel: %s", LISTEN_CHANNEL)

            while True:
                if select.select([conn], [], [], 5) == ([], [], []):
                    logger.debug("listener heartbeat…")
                    continue
                conn.poll()
                while conn.notifies:
                    note = conn.notifies.pop(0)
                    logger.info("NOTIFY payload=%s", note.payload)
                    try:
                        row_id = int(note.payload)
                        process_row(row_id)
                    except Exception:
                        logger.exception("Listener parse/process error for payload=%s", note.payload)
        except Exception:
            logger.exception("pg_listener crashed; restarting in 3s")
            time.sleep(3)

def missing_scanner():
    """Continuously ensure ALL rows missing either field are filled."""
    # startup sweep
    rows = fetch_missing(BACKFILL_LIMIT)
    logger.info("Startup backfill: %s rows with missing fields", len(rows))
    for r in rows:
        process_row(r["id"])

    # continuous
    while True:
        try:
            rows = fetch_missing(BACKFILL_LIMIT)
            if rows:
                logger.info("Scanner found %s rows with missing fields", len(rows))
                for r in rows:
                    process_row(r["id"])
            else:
                logger.debug("Scanner: no missing rows.")
        except Exception:
            logger.exception("Scanner loop error")
        time.sleep(SCAN_INTERVAL_SECONDS)

def start_workers():
    ensure_trigger()
    threading.Thread(target=pg_listener, name="pg-listener", daemon=True).start()
    threading.Thread(target=missing_scanner, name="scanner", daemon=True).start()

start_workers()

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

@app.get("/row/<int:row_id>")
def get_row(row_id: int):
    row = fetch_row_by_id(row_id)
    if not row:
        return jsonify({"error": "row not found"}), 404
    return jsonify(row)

@app.post("/reprocess/<int:row_id>")
def reprocess(row_id: int):
    process_row(row_id)
    return jsonify({"ok": True, "row_id": row_id})

@app.post("/backfill")
def backfill():
    rows = fetch_missing(BACKFILL_LIMIT)
    for r in rows:
        process_row(r["id"])
    return jsonify({"processed": len(rows)})

@app.get("/debug/missing")
def debug_missing():
    rows = fetch_missing(200)
    return jsonify({"count": len(rows), "ids": [r["id"] for r in rows]})

if __name__ == "__main__":
    # Disable reloader to avoid double background threads on Windows
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)
