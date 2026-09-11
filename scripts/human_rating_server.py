"""Serve one blinded Hemera human-study assignment on localhost.

Run one assignment slot per independent rater. Ratings are saved atomically to
the study CSV after every item. The blinding key is never read by this server.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


VOTES = {"left", "tie", "right"}
DIMENSIONS = ("alignment", "quality", "overall")


class StudyStore:
    def __init__(self, path: Path, rater_id: str, slot: int):
        if slot < 0:
            raise ValueError("assignment slot must be non-negative")
        if not rater_id.strip():
            raise ValueError("rater ID must be non-empty")
        self.path = path.resolve()
        self.rater_id = rater_id.strip()
        self.slot = slot
        self.lock = threading.Lock()
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.fieldnames = list(reader.fieldnames or ())
            self.rows = list(reader)
        required = {
            "assignment_id",
            "prompt",
            "category",
            "left_image",
            "right_image",
            "rater_id",
            *DIMENSIONS,
        }
        if not required.issubset(self.fieldnames):
            raise ValueError(
                f"Study CSV is missing fields: {sorted(required - set(self.fieldnames))}"
            )
        self.assignment_rows = {
            row["assignment_id"]: row
            for row in self.rows
            if self._row_slot(row) == slot
        }
        if not self.assignment_rows:
            raise ValueError(f"No assignments found for slot {slot}")
        conflicts = [
            row["assignment_id"]
            for row in self.assignment_rows.values()
            if row["rater_id"].strip()
            and row["rater_id"].strip() != self.rater_id
        ]
        if conflicts:
            raise ValueError(
                f"Slot {slot} already contains ratings by another rater; "
                f"first conflict: {conflicts[0]}"
            )

    @staticmethod
    def _row_slot(row: dict[str, str]) -> int:
        try:
            return int(row["assignment_id"].rsplit("-", maxsplit=1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"Invalid assignment ID {row.get('assignment_id')!r}"
            ) from error

    @staticmethod
    def _complete(row: dict[str, str]) -> bool:
        return bool(row["rater_id"].strip()) and all(
            row[field].strip().lower() in VOTES for field in DIMENSIONS
        )

    def state(self) -> dict[str, Any]:
        assigned = list(self.assignment_rows.values())
        complete = sum(self._complete(row) for row in assigned)
        current = next((row for row in assigned if not self._complete(row)), None)
        if current is None:
            return {"done": True, "complete": complete, "total": len(assigned)}
        return {
            "done": False,
            "complete": complete,
            "total": len(assigned),
            "assignment_id": current["assignment_id"],
            "prompt": current["prompt"],
            "category": current["category"],
            "left_url": f"/image/{current['assignment_id']}/left",
            "right_url": f"/image/{current['assignment_id']}/right",
        }

    def image(self, assignment_id: str, side: str) -> Path:
        if side not in {"left", "right"}:
            raise KeyError(side)
        try:
            row = self.assignment_rows[assignment_id]
        except KeyError as error:
            raise KeyError(assignment_id) from error
        path = Path(row[f"{side}_image"]).resolve()
        if not path.is_file() or path.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }:
            raise FileNotFoundError(path)
        return path

    def rate(self, payload: dict[str, Any]) -> dict[str, Any]:
        assignment_id = str(payload.get("assignment_id", ""))
        votes = {
            field: str(payload.get(field, "")).strip().lower()
            for field in DIMENSIONS
        }
        if not all(vote in VOTES for vote in votes.values()):
            raise ValueError("Every rating must be left, tie, or right")
        with self.lock:
            try:
                row = self.assignment_rows[assignment_id]
            except KeyError as error:
                raise ValueError("Unknown assignment ID") from error
            if self._complete(row):
                raise ValueError("Assignment was already rated")
            row.update(votes)
            row["rater_id"] = self.rater_id
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)
            os.replace(temporary, self.path)
            return self.state()


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hemera blinded rating</title>
<style>
body{font:16px system-ui;margin:0;background:#11131a;color:#eef1f7}
main{max-width:1050px;margin:auto;padding:24px}
.top{display:flex;justify-content:space-between;gap:20px;align-items:center}
.prompt{font-size:1.25rem;line-height:1.45;padding:18px;background:#1c202b;border-radius:12px}
.images{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:20px 0}
.image{background:#1c202b;padding:12px;border-radius:12px;text-align:center}
img{width:100%;max-width:460px;image-rendering:auto}
.dimension{display:grid;grid-template-columns:150px repeat(3,1fr);gap:8px;align-items:center;margin:10px 0}
button{padding:12px;border:1px solid #596075;border-radius:9px;background:#242938;color:#fff;cursor:pointer}
button.selected{background:#7757e8;border-color:#a996ff}
#submit{width:100%;margin-top:16px;background:#16784d;font-weight:700}
.muted{color:#aeb6c8}.done{text-align:center;padding:70px;font-size:1.4rem}
@media(max-width:700px){.images{grid-template-columns:1fr}.dimension{grid-template-columns:1fr repeat(3,1fr)}}
</style></head><body><main>
<div class="top"><h1>Blinded image comparison</h1><div id="progress"></div></div>
<div id="study">
<p class="muted">Judge each dimension independently. “Tie” is valid; do not force a winner.</p>
<div class="prompt"><span id="category"></span><br><strong id="prompt"></strong></div>
<div class="images"><div class="image"><h2>Left</h2><img id="left"></div>
<div class="image"><h2>Right</h2><img id="right"></div></div>
<div id="ratings"></div><button id="submit">Save and continue</button>
</div><div id="done" class="done" hidden>Assignment complete. Thank you.</div>
<p id="error"></p></main>
<script>
const dimensions=[
 ['alignment','Prompt alignment'],['quality','Visual quality'],['overall','Overall preference']
]; let current=null, votes={};
function makeRatings(){const root=document.querySelector('#ratings');root.innerHTML='';
 dimensions.forEach(([key,label])=>{const row=document.createElement('div');row.className='dimension';
 row.innerHTML=`<strong>${label}</strong>`;
 ['left','tie','right'].forEach(value=>{const b=document.createElement('button');b.textContent=value;
 b.onclick=()=>{votes[key]=value;row.querySelectorAll('button').forEach(x=>x.classList.remove('selected'));b.classList.add('selected')};row.appendChild(b)});root.appendChild(row)})}
async function load(){const r=await fetch('/api/state');const s=await r.json();
 document.querySelector('#progress').textContent=`${s.complete} / ${s.total}`;
 if(s.done){document.querySelector('#study').hidden=true;document.querySelector('#done').hidden=false;return}
 current=s.assignment_id;votes={};makeRatings();document.querySelector('#category').textContent=s.category;
 document.querySelector('#prompt').textContent=s.prompt;document.querySelector('#left').src=s.left_url;
 document.querySelector('#right').src=s.right_url}
document.querySelector('#submit').onclick=async()=>{if(dimensions.some(([k])=>!votes[k])){
 document.querySelector('#error').textContent='Rate all three dimensions first.';return}
 const r=await fetch('/api/rate',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({assignment_id:current,...votes})});const s=await r.json();
 if(!r.ok){document.querySelector('#error').textContent=s.error;return}
 document.querySelector('#error').textContent='';await load()};
load().catch(e=>document.querySelector('#error').textContent=e);
</script></body></html>"""


def make_handler(store: StudyStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = PAGE.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/state":
                self._json(store.state())
                return
            parts = [unquote(part) for part in parsed.path.split("/") if part]
            if len(parts) == 3 and parts[0] == "image":
                try:
                    image = store.image(parts[1], parts[2])
                    body = image.read_bytes()
                except (KeyError, FileNotFoundError):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type",
                    mimetypes.guess_type(image.name)[0] or "application/octet-stream",
                )
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/rate":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                self._json(store.rate(payload))
            except (ValueError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--rater-id", required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = StudyStore(args.study, args.rater_id, args.slot)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = f"http://{args.host}:{server.server_port}"
    print(f"Rater: {store.rater_id}; slot: {store.slot}")
    print(f"Study: {store.path}")
    print(f"Open {url} — press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
