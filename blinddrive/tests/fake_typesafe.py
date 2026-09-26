"""A local stand-in for the TypeSafe System One HTTP API, for tests only.

It speaks the same wire format as api.typesafe.ai and answers with fixed,
scripted labels. It is NOT a driving policy and is never used outside tests.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeTypeSafe:
    def __init__(self, steering="STRAIGHT", throttle="ACCELERATE", key="test-key"):
        self.steering = steering
        self.throttle = throttle
        self.key = key
        self.requests = []          # decoded request bodies
        self.headers = []           # request headers
        self.script = []            # queue of (status, body) returned before normal answers
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body, extra=None):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("x-typesafe-request-id", f"req-{len(fake.requests)}")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _auth_ok(self):
                if self.headers.get("Authorization") != f"Bearer {fake.key}":
                    self._send(401, {"message": "Invalid API key"})
                    return False
                return True

            def do_GET(self):
                if not self._auth_ok():
                    return
                if self.path == "/v1/models":
                    self._send(200, {"models": [{"name": "jev-latest", "description": "fake", "release_date": "2026-09-15"}]})
                else:
                    self._send(404, {"message": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                fake.headers.append(dict(self.headers))
                if not self._auth_ok():
                    return
                if fake.script:
                    status, payload = fake.script.pop(0)
                    self._send(status, payload, {"retry-after-ms": "1"} if status in (429, 503) else None)
                    return
                answers = {}
                for name, q in body["questions"].items():
                    label = fake.steering if name == "steering" else fake.throttle
                    labels = list(q["criteria"])
                    probs = {lab: (0.7 if lab == label else 0.3 / (len(labels) - 1)) for lab in labels}
                    answers[name] = {"type": "choice", "choice": label, "confidence": 0.7, "probabilities": probs}
                self._send(200, {"model": body["model"], "usage": {"input_tokens": 100, "output_tokens": 2},
                                 "answers": answers})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
