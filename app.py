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

# NEW: load .env (local dev). In containers/EC2, you can inject env vars instead.
try:
    from dotenv import load_dotenv
    load_dotenv()  # loads variables from .env into process env if present
except Exception:
    pass

# =========================
# CONFIG via ENV VARS (no hard-coded secrets)
# =========================
DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
# NOTE: Prefer IAM roles in production. If keys are present in env, boto3 will use them automatically.
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

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

# Logging
LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "DEBUG").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.DEBUG)
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", os.path.join(LOG_DIR, "app.log"))
LOG_PREVIEW_CHARS = int(os.getenv("LOG_PREVIEW_CHARS", "300"))

# =========================
# Logging setup
# =========================
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

# =========================
# Flask + Bedrock client
# =========================
app = Flask(__name__)

logger.info("Booting app | region=%s | table=%s", AWS_REGION, 'public."myApp_meetingtranscript"')
logger.info("DB host=%s db=%s user=%s", DB_HOST, DB_NAME, DB_USER)

# Build Bedrock client. Let boto3 pick creds from env/instance role automatically.
bedrock_kwargs = {
    "region_name": AWS_REGION,
    "config": Config(retries={"max_attempts": 8, "mode": "adaptive"}),
}
# If you *must* use env keys locally, boto3 will read them automatically from env.
# But we also allow explicit pass-through if set (optional).
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    bedrock_kwargs.update({
        "aws_access_key_id": AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    })

bedrock = boto3.client("bedrock-runtime", **bedrock_kwargs)
logger.debug("Bedrock client ready for model_id=%s", MODEL_ID)

# =========================
# Utils
# =========================
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

# =========================
# DB helpers (unchanged except they use env-backed config)
# =========================
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
    """Create the NOTIFY function + trigger (idempotent)."""
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
    Any row with empty/NULL summary OR missing actionItem.
    Missing actionItem includes:
      - SQL NULL
      - JSONB null
      - empty object {}
    NOTE: We intentionally do NOT treat [] as missing (prevents infinite loops when a meeting truly has no tasks).
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
           )
        ORDER BY created_at DESC, id DESC
        LIMIT %s
    """
    try:
        with db_conn() as conn, conn.cursor() as cur:
            cur.execute(q, (limit,))
            rows = cur.fetchall()
            logger.debug(
                "fetch_missing(limit=%s) -> %s rows (treating NULL/json null/{} as missing, [] as filled)",
                limit, len(rows)
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

# =========================
# LLM prompts — Narrative (2–4 paragraphs, 3–12 sentences), precise AIs
# Action items must be [{"item":"...","owner":"...","deadline":"..."}]
# Owner/deadline blank "" if unknown.
# =========================
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
        "NO headings, NO bullet lists. Be chronological; cover context/purpose, key points, decisions, risks, next steps—ONLY if present.\n"
        '  "action_items": [\n'
        "    {\n"
        '      "item": string,          // verb-first, precise, ≤ 25 words\n'
        '      "owner": string,         // if owner not explicit, set ""\n'
        '      "deadline": string       // date e.g. 2/15/2023; if not explicit, set ""\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"- summary_md MUST be plain Markdown prose, {PARA_MIN}-{PARA_MAX} paragraphs, {SUMMARY_MIN_SENTENCES}-{SUMMARY_MAX_SENTENCES} sentences total.\n"
        "- Include ALL clear action items; omit vague ideas. Do NOT add extra fields to items.\n"
        '- Owner and deadline MUST be strings; use "" when unknown.\n'
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
        '  "summary_md": string,  // multi-paragraph narrative with '
        f'{PARA_MIN}-{PARA_MAX} paragraphs and {SUMMARY_MIN_SENTENCES}-{SUMMARY_MAX_SENTENCES} sentences total; '
        "no headings/bullets; chronological; detailed but only facts present.\n"
        '  "action_items": [ { "item":string, "owner":string, "deadline":string } ]\n'
        "}\n"
        "- Deduplicate near-duplicates in action_items.\n"
        '- Ensure each item has all three keys; use "" for unknown owner/deadline.\n\n'
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
        f"{SUMMARY_MIN_SENTENCES}-{SUMMARY_MAX_SENTENCES} sentences total, chronological, and without headings/bullets. "
        "Keep facts unchanged.\n\n"
        f"{summary_md}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return {"prompt": chat, "max_gen_len": 700, "temperature": 0.2, "top_p": 0.9, "stop": ["<|eot_id|>"]}

# =========================
# Bedrock helpers
# =========================
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
        obj = json.loads(generation_text)
        return obj
    except Exception:
        m = re.search(r"\{.*\}", generation_text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                logger.debug("parse_generation_json: JSON block failed. Preview=%s", preview(generation_text))
    logger.warning("parse_generation_json: returning empty structure. Preview=%s", preview(generation_text))
    return {"summary_md": "", "action_items": []}

# =========================
# Transcript -> summary/action
# =========================
def flatten_transcription(transcription: Any) -> str:
    """Turns JSONB transcript into sequential text with speakers if present."""
    if transcription is None:
        logger.debug("flatten_transcription: None input.")
        return ""
    if isinstance(transcription, str):
        try:
            transcription = json.loads(transcription)
        except Exception:
            logger.debug("flatten_transcription: string not JSON; using raw text preview=%s", preview(transcription))
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
    """
    Keep precise, verb-first items; map blanks to ""; de-dup by normalized text.
    Target schema: [{"item": "...", "owner": "...", "deadline": "..."}]
    """
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
    """Main summarization flow with chunking, merge, and paragraph/sentence enforcement."""
    if not text.strip():
        logger.info("summarize_transcript: empty transcript text.")
        return {"summary_md": "", "action_items": []}

    logger.debug("summarize_transcript: total chars=%s", len(text))

    # Single-shot if small
    if len(text) <= CHUNK_CHAR_LIMIT:
        logger.debug("Single-shot summarization path.")
        data = bedrock_invoke(mk_prompt_chunk(text))
        obj = parse_generation_json(data.get("generation", ""))
    else:
        # Chunked + merge
        chunks = [text[i:i+CHUNK_CHAR_LIMIT] for i in range(0, len(text), CHUNK_CHAR_LIMIT)]
        logger.debug("Chunked summarization: %s chunks of up to %s chars.", len(chunks), CHUNK_CHAR_LIMIT)

        chunk_summaries, chunk_items = [], []
        for idx, ch in enumerate(chunks, start=1):
            logger.debug("Processing chunk %s/%s (chars=%s)", idx, len(chunks), len(ch))
            data = bedrock_invoke(mk_prompt_chunk(ch))
            co = parse_generation_json(data.get("generation", ""))
            if co.get("summary_md"):
                logger.debug("Chunk %s summary preview=%s", idx, preview(co["summary_md"]))
                chunk_summaries.append(co["summary_md"])
            if isinstance(co.get("action_items"), list):
                logger.debug("Chunk %s action_items=%s", idx, len(co["action_items"]))
                chunk_items.extend(co["action_items"])

        data = bedrock_invoke(mk_prompt_merge(chunk_summaries, chunk_items))
        obj = parse_generation_json(data.get("generation", ""))

    # Enforce paragraph/sentence bounds
    summary = (obj.get("summary_md") or "").strip()
    n_sent = count_sentences(summary)
    n_para = count_paragraphs(summary)
    if (n_sent < SUMMARY_MIN_SENTENCES or n_sent > SUMMARY_MAX_SENTENCES) or (n_para < PARA_MIN or n_para > PARA_MAX):
        logger.info("Rewriting summary to meet bounds (sent=%s, para=%s).", n_sent, n_para)
        data2 = bedrock_invoke(mk_prompt_rewrite(summary))
        summary2 = (data2.get("generation") or "").strip()
        if summary2:
            summary = summary2
            logger.debug("Rewrite preview=%s", preview(summary))
        else:
            logger.warning("Rewrite returned empty; keeping original.")

    # Normalize precise items & dedupe to target schema
    items = normalize_items(obj.get("action_items", []))
    logger.debug("Final summary len=%s, action_items=%s", len(summary), len(items))
    return {"summary_md": summary, "action_items": items}

# =========================
# Missing checks & row processing
# =========================
def is_items_missing(js: Any) -> bool:
    """
    Treat actionItem as missing when it is:
      - SQL NULL
      - JSONB null
      - empty object {}
    Treat [] as 'filled' (meeting had no actionable tasks).
    """
    if js is None:
        return True
    try:
        obj = json.loads(js) if isinstance(js, str) else js
    except Exception:
        # Unknown string -> treat as filled to avoid loops.
        return False
    if obj is None:
        return True
    if isinstance(obj, dict) and len(obj) == 0:
        return True
    return False  # list [] or non-empty structures are considered filled

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
        logger.debug("Row %s transcript chars=%s preview=%s",
                     row_id, len(transcript_txt), preview(transcript_txt))

        result = summarize_transcript(transcript_txt)

        summary_to_write = result["summary_md"] if need_summary else None
        items_to_write   = result["action_items"] if need_items else None

        if need_items and isinstance(items_to_write, list) and len(items_to_write) == 0:
            logger.info("Row %s: no explicit action items found; saving empty list once and marking complete.", row_id)

        update_row_partial(row_id, summary=summary_to_write, action_items=items_to_write)
    except Exception:
        logger.exception("process_row failed for id=%s", row_id)

# =========================
# Background workers
# =========================
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
                    logger.info("NOTIFY received payload=%s", note.payload)
                    try:
                        row_id = int(note.payload)
                        process_row(row_id)
                    except Exception:
                        logger.exception("Listener could not parse/process payload=%s", note.payload)
        except Exception:
            logger.exception("pg_listener crashed; restarting in 3s")
            time.sleep(3)

def missing_scanner():
    """Continuously ensure ALL rows missing either field are filled."""
    # Initial sweep at startup
    rows = fetch_missing(BACKFILL_LIMIT)
    logger.info("Startup backfill: %s rows with missing fields", len(rows))
    for r in rows:
        process_row(r["id"])

    # Continuous scan
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

# =========================
# Routes
# =========================
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
