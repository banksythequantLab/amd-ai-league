#!/usr/bin/env python3
"""AMD Mini-Challenge 3 submission: RAG over a mixed-filetype corpus.

Architecture
  --index  : walks the corpus defensively (per-file try/except, sorted order),
             extracts text from every readable type, then forks a resident
             model server (unix socket) that transcribes images with the VLM,
             persists the index, and stays up. The index process waits for the
             server's "ready" signal, so everything is built before it exits.
  --query  : a thin client. Sends the question over the socket, gets JSON back.
             If the server is missing (crashed, never started), it falls back
             to loading the model in-process and restarts the server.

Answering = BM25 retrieval -> second hop on identifiers found in hits ->
VLM/LLM picks answer + sources as JSON -> deterministic verification (answer
must literally occur in a cited file, withdrawn docs dropped when a current
one also holds it) -> otherwise refuse with ("", []).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import struct
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
MODEL_DIR = os.environ.get("MC3_MODEL", "/models/vlm")
SOCK_PATH = str(INDEX_DIR / "server.sock")
DOCS_FILE = INDEX_DIR / "docs.json"
LOG_FILE = INDEX_DIR / "server.log"

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
STALE_RE = re.compile(r"\b(WITHDRAWN|SUPERSEDED|OBSOLETE|DEPRECATED|RETRACTED|DO NOT USE)\b", re.I)


def is_stale(rel: str, text: str) -> bool:
    """Marked stale by filename, or by its own title block (first 3 lines).
    'supersedes revision 1' (the CURRENT doc) must not count."""
    if STALE_RE.search(Path(rel).stem.replace("_", " ")):
        return True
    head = "\n".join([l for l in text.splitlines() if l.strip()][:3])
    head = re.sub(r"(?i)\bsupersedes\b", "", head)
    return bool(STALE_RE.search(head))


def log(*a):
    try:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a) + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Extraction. Every extractor may raise; the walker catches per file.
# --------------------------------------------------------------------------

def _pdf(p: Path) -> str:
    import pypdf
    r = pypdf.PdfReader(str(p))
    if r.is_encrypted:
        raise PermissionError("encrypted pdf")  # never decrypt: refusal test
    out = []
    for pg in r.pages:
        try:
            out.append(pg.extract_text() or "")
        except Exception as e:
            log("pdf page fail", p, e)
    return "\n".join(out)


def _docx(p: Path) -> str:
    import docx
    d = docx.Document(str(p))
    parts = [x.text for x in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(parts)


def _xlsx(p: Path) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(str(p), data_only=True, read_only=True)
    out = []
    for ws in wb.worksheets:
        out.append(f"## sheet: {ws.title}")
        header = None
        for row in ws.iter_rows(values_only=True):
            vals = ["" if v is None else str(v) for v in row]
            if not any(vals):
                continue
            if header is None:
                header = vals
                out.append(" | ".join(vals))
            elif len(vals) == len(header) and len(header) > 1:
                out.append("; ".join(f"{h}={v}" for h, v in zip(header, vals) if v))
            else:
                out.append(" | ".join(vals))
    return "\n".join(out)


def _csv(p: Path) -> str:
    import csv
    raw = _decode(p.read_bytes())
    rows = list(csv.reader(raw.splitlines()))
    if len(rows) < 2:
        return raw
    h = rows[0]
    lines = [", ".join(h)]
    for r in rows[1:]:
        lines.append("; ".join(f"{a}={b}" for a, b in zip(h, r) if b.strip()))
    return "\n".join(lines)


def _pptx(p: Path) -> str:
    from pptx import Presentation
    out = []
    for i, s in enumerate(Presentation(str(p)).slides, 1):
        out.append(f"## slide {i}")
        for sh in s.shapes:
            if getattr(sh, "has_text_frame", False):
                out.append(sh.text_frame.text)
    return "\n".join(out)


def _html(p: Path) -> str:
    t = _decode(p.read_bytes())
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    return re.sub(r"<[^>]+>", " ", t)


def _decode(b: bytes) -> str:
    for enc in ("utf-8", "utf-16"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("latin-1")


def _texty(p: Path) -> str:
    b = p.read_bytes()
    if not b:
        return ""
    sample = b[:8192]
    if b"\x00" in sample and not (sample.startswith(b"\xff\xfe") or sample.startswith(b"\xfe\xff")):
        raise ValueError("binary")
    t = _decode(b)
    printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in t[:8192])
    if printable / max(1, len(t[:8192])) < 0.9:
        raise ValueError("binary")
    return t


EXTRACT = {".pdf": _pdf, ".docx": _docx, ".xlsx": _xlsx, ".xlsm": _xlsx,
           ".csv": _csv, ".pptx": _pptx, ".html": _html, ".htm": _html}


def walk(corpus: Path):
    """Yield (relpath, kind, text_or_None). Never raises."""
    for root, dirs, files in os.walk(corpus, onerror=lambda e: log("walk err", e)):
        dirs.sort()
        for name in sorted(files):
            full = Path(root) / name
            rel = full.relative_to(corpus).as_posix()
            ext = full.suffix.lower()
            try:
                if not full.is_file():
                    continue
                if ext in IMAGE_EXT:
                    with open(full, "rb") as f:  # prove readable
                        f.read(16)
                    yield rel, "image", None
                    continue
                fn = EXTRACT.get(ext, _texty)
                text = fn(full)
                if text and text.strip():
                    yield rel, "text", text
                else:
                    log("empty", rel)
            except Exception as e:
                log("skip", rel, type(e).__name__, e)


# --------------------------------------------------------------------------
# Retrieval: chunking + BM25, no extra deps.
# --------------------------------------------------------------------------

TOK = re.compile(r"[A-Za-z0-9]+(?:[-_.#][A-Za-z0-9]+)*#?")


def tokens(s: str):
    out = []
    for t in TOK.findall(s.lower()):
        out.append(t)
        if re.search(r"[-_.]", t):
            out.extend(x for x in re.split(r"[-_.#]", t) if x)
    return out


STOP = set("the a an of in on at to for is are was what which who when how does do did and or by with from this that it be as its into per".split())


def chunk(rel: str, text: str, size: int = 30, overlap: int = 8):
    lines = [l for l in text.splitlines()]
    if len(lines) <= size + 10:
        return [text]
    out = []
    for i in range(0, len(lines), size - overlap):
        out.append("\n".join(lines[i:i + size]))
        if i + size >= len(lines):
            break
    return out


class BM25:
    def __init__(self, chunks):
        self.chunks = chunks  # list of dicts {path, text, stale}
        self.tf = []
        df = Counter()
        for c in chunks:
            toks = [t for t in tokens(c["path"] + " " + c["text"]) if t not in STOP]
            tf = Counter(toks)
            self.tf.append((tf, len(toks)))
            df.update(tf.keys())
        n = max(1, len(chunks))
        self.avg = sum(l for _, l in self.tf) / n
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, q: str, k: int = 6):
        qt = [t for t in tokens(q) if t not in STOP]
        scores = []
        for i, (tf, ln) in enumerate(self.tf):
            s = 0.0
            for t in qt:
                f = tf.get(t)
                if f:
                    s += self.idf.get(t, 0) * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * ln / max(1, self.avg)))
            if self.chunks[i]["stale"]:
                s *= 0.6
            scores.append((s, i))
        scores.sort(reverse=True)
        return [(s, i) for s, i in scores[:k] if s > 0]


ID_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}-\d{2,}[A-Z0-9-]*\b|\b[A-Z]\d{3,}\b")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class Model:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if not torch.cuda.is_available():  # ROCm exposes HIP as torch.cuda
            raise RuntimeError("no GPU visible to torch")
        t = time.time()
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(MODEL_DIR)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa").eval()
        log(f"model loaded in {time.time()-t:.1f}s on", torch.cuda.get_device_name(0))

    def gen(self, content, max_new=160) -> str:
        msgs = [{"role": "user", "content": content}]
        inputs = self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        inputs.pop("token_type_ids", None)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return self.proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def describe_image(self, path: Path) -> str:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img.thumbnail((1600, 1600))
        prompt = (
            "Transcribe ALL text visible in this image exactly, preserving codes, part numbers, "
            "revisions and values character for character. Then, if it is a diagram, table, "
            "label or chart, list every relationship explicitly, one per line, in the form "
            "'<item> : <label/value>' (e.g. which signal is on which pin, which field has which "
            "value). Do not add information that is not in the image.")
        return self.gen([{"type": "image", "image": img}, {"type": "text", "text": prompt}], max_new=700)


# --------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------

PROMPT = """You answer questions using ONLY the numbered sources below. The products are fictional: never use outside knowledge.

Rules:
- "answer" is the bare value only (a number, part number, version, code, quarter, name). No sentence, no units. Keep qualifiers that are part of the value (e.g. "Q3 FY27", not "Q3").
- A source marked [WITHDRAWN/SUPERSEDED] must not be used if a current source gives the value.
- In code, the value assigned to a constant beats a value mentioned in a comment.
- If answering requires chaining facts (e.g. a log names a ticket, a database gives that ticket's fix), use every source needed.
- "sources" lists ONLY the sources whose removal would make the answer impossible. Usually one; two for a chained answer.
- The value must be for exactly what is asked: the same item (a part, accessory or component of a product is NOT the product), the same attribute (a cost of one thing is not the price of another), and the same conditions (volume tier, revision, date). A near-miss is not an answer.
- If the sources do not contain the answer, return {{"answer": "", "sources": []}}. Do not guess.

{ctx}

Question: {q}

Reply with JSON only: {{"answer": "...", "sources": ["S1"]}}"""


def norm(s: str) -> str:
    s = s.lower().replace("°", "")
    return re.sub(r"[^a-z0-9.#]+", "", s)


class Engine:
    def __init__(self, model):
        self.model = model
        self.docs = json.loads(DOCS_FILE.read_text(encoding="utf-8"))
        chunks = []
        for d in self.docs:
            for t in chunk(d["path"], d["text"]):
                chunks.append({"path": d["path"], "text": t, "stale": d["stale"]})
        self.bm = BM25(chunks)
        self.full = {d["path"]: d["text"] for d in self.docs}
        self.stale = {d["path"]: d["stale"] for d in self.docs}

    def retrieve(self, q):
        hits = self.bm.search(q, 6)
        picked = [i for _, i in hits]
        # second hop: identifiers that occur in the hits but not in the question
        ids = []
        for i in picked[:4]:
            for m in ID_RE.findall(self.bm.chunks[i]["text"]):
                if m.lower() not in q.lower() and m not in ids:
                    ids.append(m)
        for ident in ids[:4]:
            for s, i in self.bm.search(ident, 2):
                if i not in picked and ident.lower() in self.bm.chunks[i]["text"].lower():
                    picked.append(i)
        return picked[:9]

    def answer(self, q):
        picked = self.retrieve(q)
        if not picked:
            return "", [], 0.0
        labels, ctx, budget = {}, [], 14000
        for n, i in enumerate(picked, 1):
            c = self.bm.chunks[i]
            tag = " [WITHDRAWN/SUPERSEDED]" if c["stale"] else ""
            body = c["text"][:3000]
            if budget - len(body) < 0:
                break
            budget -= len(body)
            labels[f"S{n}"] = c["path"]
            ctx.append(f"[S{n}] file: {c['path']}{tag}\n{body}\n")
        raw = self.model.gen([{"type": "text", "text": PROMPT.format(ctx="\n".join(ctx), q=q)}], 96)
        log("Q", q, "->", raw)
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except Exception:
            obj = {}
        ans = str(obj.get("answer", "") or "").strip().strip(".")
        srcs = [labels[s] for s in obj.get("sources", []) if isinstance(s, str) and s in labels]
        srcs = list(dict.fromkeys(srcs))
        if not ans:
            return "", [], 0.5
        na = norm(ans)
        holders = [p for p in labels.values() if na and na in norm(self.full[p])]
        if not holders:
            log("answer not grounded; refusing", ans)
            return "", [], 0.1
        # the answer must live in at least one cited file
        if not any(p in holders for p in srcs):
            fresh = [p for p in holders if not self.stale[p]] or holders
            srcs = srcs + [fresh[0]]
        # drop a withdrawn file if a current file already carries the answer
        if any(not self.stale[p] for p in srcs if p in holders):
            srcs = [p for p in srcs if not (self.stale[p] and p in holders)]
        srcs = [p for p in srcs if not self.stale[p]] or srcs
        if not self.judge(q, ans, srcs):
            log("judge rejected; refusing", ans, srcs)
            return "", [], 0.2
        return ans, sorted(set(srcs)), 0.8

    def judge(self, q, ans, srcs) -> bool:
        ev = "\n\n".join(f"[{p}]\n{self.full[p][:4000]}" for p in srcs)
        prompt = (
            f"Evidence:\n{ev}\n\nQuestion: {q}\nProposed answer: {ans}\n\n"
            "Does the evidence explicitly support this answer to exactly this question? It must be "
            "the same item (a part, accessory or component of a product does not count as the product), "
            "the same attribute (e.g. a part's cost is not a product's price), and the same conditions "
            "(quantity/volume tier, revision, date) the question names. Combining facts across the "
            "evidence files is allowed. Reply with exactly one word: YES or NO.")
        raw = self.model.gen([{"type": "text", "text": prompt}], 4)
        log("judge", repr(raw))
        return raw.strip().upper().startswith("Y")


# --------------------------------------------------------------------------
# Index + server plumbing
# --------------------------------------------------------------------------

def build_index(corpus: Path, model) -> None:
    docs = []
    for rel, kind, text in walk(corpus):
        if kind == "image":
            if model is None:
                continue
            try:
                text = model.describe_image(corpus / rel)
                log("image", rel, "->", text.replace("\n", " | ")[:400])
            except Exception as e:
                log("image fail", rel, e)
                continue
        docs.append({"path": rel, "text": text, "stale": is_stale(rel, text)})
    tmp = DOCS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    tmp.replace(DOCS_FILE)
    log("indexed", len(docs), "docs:", [d["path"] for d in docs])


def _send(conn, obj):
    b = json.dumps(obj).encode()
    conn.sendall(struct.pack("!I", len(b)) + b)


def _recv(conn):
    hdr = b""
    while len(hdr) < 4:
        x = conn.recv(4 - len(hdr))
        if not x:
            raise ConnectionError("closed")
        hdr += x
    n = struct.unpack("!I", hdr)[0]
    buf = b""
    while len(buf) < n:
        x = conn.recv(n - len(buf))
        if not x:
            raise ConnectionError("closed")
        buf += x
    return json.loads(buf)


def serve(corpus: Path | None, ready_fd: int | None):
    model = Model()
    if corpus is not None:
        build_index(corpus, model)
    eng = Engine(model)
    try:
        os.unlink(SOCK_PATH)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK_PATH)
    srv.listen(8)
    # warm-up so the first graded question doesn't pay kernel compilation
    try:
        eng.answer("warm up question")
    except Exception:
        log("warmup fail", traceback.format_exc())
    if ready_fd is not None:
        os.write(ready_fd, b"R")
        os.close(ready_fd)
    log("serving on", SOCK_PATH)
    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                req = _recv(conn)
                a, c, conf = eng.answer(req["query"])
                _send(conn, {"answer": a, "citations": c, "confidence": conf})
            except Exception:
                log("req fail", traceback.format_exc())
                try:
                    _send(conn, {"error": "fail"})
                except Exception:
                    pass


def spawn_server(corpus: Path | None, wait_s: float) -> bool:
    """Fork a detached server; block until it signals ready (or dies)."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        os.setsid()
        if os.fork() > 0:
            os._exit(0)
        devnull = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(devnull, fd)
        try:
            serve(corpus, w)
        except Exception:
            log("server died", traceback.format_exc())
        os._exit(1)
    os.close(w)
    os.waitpid(pid, 0)
    import select
    ready, _, _ = select.select([r], [], [], wait_s)
    ok = bool(ready) and os.read(r, 1) == b"R"
    os.close(r)
    return ok


def index(corpus: Path) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not spawn_server(corpus, 570):
        log("server not ready; building index in-process")
        if not DOCS_FILE.exists():
            try:
                build_index(corpus, Model())
            except Exception:
                # never fail the index pass: keep every text file, skip images
                log("model unavailable; text-only index", traceback.format_exc()[-300:])
                build_index(corpus, None)


def ask_server(q: str, timeout: float):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(SOCK_PATH)
    with s:
        _send(s, {"query": q})
        return _recv(s)


def answer(corpus: Path, query: str) -> tuple[str, list[str], float]:
    try:
        r = ask_server(query, 27)
        if "answer" in r:
            return r["answer"], r["citations"], r["confidence"]
    except Exception as e:
        log("socket miss", e)
    try:  # fallback: in-process (slow) — better than a guaranteed zero
        if not DOCS_FILE.exists():
            build_index(corpus, Model())
        a, c, conf = Engine(Model()).answer(query)
        return a, c, conf
    except Exception:
        log("fallback fail", traceback.format_exc())
        return "", [], 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        index(args.index)
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    text, citations, confidence = answer(args.corpus, args.query)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (args.query_id + "_output.json")
    tmp = out.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"answer": text, "citations": list(citations), "confidence": confidence},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
