"""Run the MoneyHeist AI chatbot on this computer with python3 server.py."""

import json
import os
import socket
import ssl
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
PORT = 8000
MODEL = "openai/gpt-oss-20b"
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
PERSONAS = {
    "professor": "You are the Professor, a calm, analytical fictional strategist.",
    "tokyo": "You are Tokyo, a bold and imaginative fictional storyteller.",
    "nairobi": "You are Nairobi, a warm, direct and encouraging fictional mentor.",
}
LOCAL_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}


def get_api_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        lines = (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return ""
    for line in lines:
        name, sep, value = line.partition("=")
        if sep and name.strip() == "GROQ_API_KEY":
            return value.strip().strip("\"'")
    return ""


def groq_payload(data):
    if not isinstance(data, dict):
        raise ValueError("Invalid message.")
    channel = data.get("channel")
    if not isinstance(channel, str) or channel not in PERSONAS:
        raise ValueError("Choose a crew channel.")
    temperature = data.get("temperature", 1.4)
    if type(temperature) not in (float, int) or not 0 <= temperature <= 2:
        raise ValueError("Choose a temperature between 0 and 2.")
    history = data.get("messages")
    if not isinstance(history, list) or not 1 <= len(history) <= 20:
        raise ValueError("Send between 1 and 20 messages.")
    prompt = (PERSONAS[channel] + " You are a fan-made Money Heist chat character. "
              "Help with everyday questions, learning, programming, fiction and puzzles. "
              "Match the user's language, including Hinglish. Stay concise unless asked for detail.")
    messages = [{"role": "system", "content": prompt}]
    for message in history:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            raise ValueError("Invalid message role.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > 16000:
            raise ValueError("Invalid message length.")
        messages.append({"role": message["role"], "content": content.strip()})
    if messages[-1]["role"] != "user":
        raise ValueError("Send a user message.")
    return {"model": MODEL, "messages": messages, "temperature": temperature,
            "max_completion_tokens": 1000}


@lru_cache(maxsize=1)
def groq_tls_context():
    context = ssl.create_default_context()
    paths = ssl.get_default_verify_paths()
    if context.cert_store_stats()["x509_ca"] or paths.capath:
        return context
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return context
    # Some python.org macOS installs have no default CA bundle yet.
    try:
        from certifi import where
    except ImportError:
        bundle = Path("/etc/ssl/cert.pem")
        if not bundle.is_file():
            return context
    else:
        bundle = where()
    context.load_verify_locations(cafile=str(bundle))
    return context


def get_reply(payload, key):
    request = Request(ENDPOINT, data=json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "MoneyHeistAI/1.0",
    })
    with urlopen(request, timeout=30, context=groq_tls_context()) as response:
        result = json.load(response)
    answer = result["choices"][0]["message"]["content"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Empty response.")
    return answer.strip()


class RadioServer(BaseHTTPRequestHandler):
    def respond(self, status, data, media_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("Host") not in LOCAL_HOSTS:
            return self.respond(403, {"error": "Use the local server address."})
        if urlsplit(self.path).path not in ("/", "/MoneyHeist-AI.html"):
            return self.respond(404, {"error": "Page not found."})
        html = (ROOT / "MoneyHeist-AI.html").read_bytes()
        html = html.replace(b'data-groq-server="off"', b'data-groq-server="on"', 1)
        self.respond(200, html, "text/html; charset=utf-8")

    def do_POST(self):
        host = self.headers.get("Host")
        if host not in LOCAL_HOSTS or self.headers.get("Origin") not in (None, f"http://{host}"):
            return self.respond(403, {"error": "Open the chat from the local server."})
        if self.path != "/api/chat":
            return self.respond(404, {"error": "Unknown route."})
        if self.headers.get_content_type() != "application/json":
            return self.respond(415, {"error": "Expected a JSON message."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 512000:
                return self.respond(413, {"error": "The request is empty or too large."})
            payload = groq_payload(json.loads(self.rfile.read(length)))
        except (ValueError, RecursionError):
            return self.respond(400, {"error": "The chat request was invalid."})
        key = get_api_key()
        if not key:
            return self.respond(503, {"error": "Add GROQ_API_KEY to .env beside server.py and try again."})
        try:
            answer = get_reply(payload, key)
        except HTTPError as error:
            messages = {
                401: "Groq rejected the key. Check GROQ_API_KEY in .env.",
                403: "Groq denied access. Check the key and model permissions.",
                429: "Groq’s rate limit was reached. Wait a moment and retry.",
            }
            message = messages.get(error.code, f"Groq returned HTTP {error.code} for {payload['model']}.")
            if error.code in (400, 403, 404):
                try:
                    details = json.loads(error.read(8192))
                except (ValueError, OSError):
                    details = {}
                provider_error = details.get("error") if isinstance(details, dict) else None
                if isinstance(provider_error, dict) and isinstance(provider_error.get("message"), str):
                    reason = provider_error["message"].replace(key, "[redacted]").strip()[:400]
                    if reason:
                        message = f"Groq HTTP {error.code} ({payload['model']}): {reason}"
            return self.respond(error.code, {"error": message})
        except OSError as error:
            reason = error.reason if isinstance(error, URLError) else error
            if isinstance(reason, ssl.SSLCertVerificationError):
                message = f"Could not verify Groq's HTTPS certificate: {reason.verify_message}. Check the trusted certificates for this Python installation."
            elif isinstance(reason, socket.gaierror):
                message = "DNS lookup failed while connecting to Groq. Check your network or proxy, then retry."
            elif isinstance(reason, TimeoutError):
                return self.respond(504, {"error": "The Groq connection timed out. Try again."})
            else:
                message = "Could not reach Groq. Check your network or proxy, then retry."
            return self.respond(502, {"error": message})
        except (KeyError, IndexError, TypeError, ValueError):
            return self.respond(502, {"error": "Groq sent an unreadable reply. Try again."})
        self.respond(200, {"reply": answer})


if __name__ == "__main__":
    try:
        with ThreadingHTTPServer(("127.0.0.1", PORT), RadioServer) as server:
            print(f"Open http://127.0.0.1:{PORT} in your browser. Press Ctrl+C to stop.", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except OSError as error:
        raise SystemExit(f"Could not start the local server on port {PORT}: {error}")
