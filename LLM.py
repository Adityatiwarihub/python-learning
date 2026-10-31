"""A tiny character-level JARVIS language model trained from random weights.

This is an educational starter model, not a capable general-purpose LLM. It uses
NumPy for fast matrix arithmetic; all model weights are created by train_model().
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "training_data.txt"
WEIGHTS_PATH = ROOT / "jarvis_brain.npz"
HOST = "127.0.0.1"
PORT = 4318
HIDDEN = 48
SEQUENCE = 48
BATCH = 8
TRAIN_STEPS = 400

MODEL = None
MODEL_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
STATE = {"modelReady": False, "training": False, "progress": 0, "loss": None, "error": None}


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


class TinyJARVIS:
    """A one-layer character RNN language model with no pretrained weights."""

    def __init__(self, vocabulary: str, weights: dict[str, np.ndarray]):
        self.vocabulary = vocabulary
        self.char_to_id = {char: index for index, char in enumerate(vocabulary)}
        self.weights = weights
        self.hidden_size = weights["Whh"].shape[0]

    def generate(self, prompt: str, max_chars: int = 260, temperature: float = 0.8):
        if not self.vocabulary:
            return
        fallback = self.char_to_id.get(" ", 0)
        ids = [self.char_to_id.get(char, fallback) for char in prompt]
        if not ids:
            ids = [self.char_to_id.get("J", fallback)]
        hidden = np.zeros((1, self.hidden_size), dtype=np.float32)
        for token_id in ids[:-1]:
            hidden, _ = self._step(token_id, hidden, temperature=1.0, sample=False)
        token_id = ids[-1]
        for index in range(max_chars):
            hidden, token_id = self._step(token_id, hidden, temperature=temperature, sample=True)
            char = self.vocabulary[token_id]
            if char == "\n" and index > 18:
                break
            if char == "\n" and index <= 18:
                char = " "
            yield char

    def _step(self, token_id: int, hidden: np.ndarray, temperature: float, sample: bool):
        params = self.weights
        x = np.zeros((1, len(self.vocabulary)), dtype=np.float32)
        x[0, token_id] = 1.0
        hidden = np.tanh(x @ params["Wxh"] + hidden @ params["Whh"] + params["bh"])
        logits = hidden @ params["Why"] + params["by"]
        if sample:
            probs = softmax(logits / max(0.35, temperature))[0].astype(np.float64)
            token_id = int(np.random.choice(len(self.vocabulary), p=probs / probs.sum()))
        else:
            token_id = int(np.argmax(logits[0]))
        return hidden, token_id


def _initial_weights(vocab_size: int, hidden_size: int, seed: int = 20261022):
    rng = np.random.default_rng(seed)
    return {
        "Wxh": (rng.standard_normal((vocab_size, hidden_size)) * 0.06).astype(np.float32),
        "Whh": (rng.standard_normal((hidden_size, hidden_size)) * 0.06).astype(np.float32),
        "bh": np.zeros((hidden_size,), dtype=np.float32),
        "Why": (rng.standard_normal((hidden_size, vocab_size)) * 0.06).astype(np.float32),
        "by": np.zeros((vocab_size,), dtype=np.float32),
    }


def train_model():
    global MODEL
    text = DATA_PATH.read_text(encoding="utf-8")
    vocabulary = "".join(sorted(set(text)))
    if len(text) < 500 or len(vocabulary) < 8:
        raise ValueError("training_data.txt needs at least 500 characters of dialogue.")
    ids = np.fromiter((vocabulary.index(char) for char in text), dtype=np.int32)
    vocab_size = len(vocabulary)
    weights = _initial_weights(vocab_size, HIDDEN)
    first_moment = {name: np.zeros_like(value) for name, value in weights.items()}
    second_moment = {name: np.zeros_like(value) for name, value in weights.items()}
    beta1, beta2, epsilon, learning_rate = 0.9, 0.999, 1e-8, 0.003
    rng = np.random.default_rng(20261022)
    eye = np.eye(vocab_size, dtype=np.float32)
    smoothed_loss = None

    with STATE_LOCK:
        STATE.update({"modelReady": False, "training": True, "progress": 0, "loss": None, "error": None,
                      "trainingChars": len(text), "vocabularySize": vocab_size, "steps": TRAIN_STEPS})

    last_start = max(1, len(ids) - SEQUENCE - 1)
    for step in range(1, TRAIN_STEPS + 1):
        starts = rng.integers(0, last_start, size=BATCH)
        inputs = np.stack([ids[start:start + SEQUENCE] for start in starts])
        targets = np.stack([ids[start + 1:start + SEQUENCE + 1] for start in starts])
        hidden_prev = np.zeros((BATCH, HIDDEN), dtype=np.float32)
        hidden_previous, hidden_values, probabilities = [], [], []
        loss = 0.0

        for tick in range(SEQUENCE):
            x = eye[inputs[:, tick]]
            hidden_previous.append(hidden_prev)
            hidden = np.tanh(x @ weights["Wxh"] + hidden_prev @ weights["Whh"] + weights["bh"])
            hidden_values.append(hidden)
            probs = softmax(hidden @ weights["Why"] + weights["by"])
            probabilities.append(probs)
            loss -= np.log(probs[np.arange(BATCH), targets[:, tick]] + 1e-9).mean() / SEQUENCE
            hidden_prev = hidden

        grads = {name: np.zeros_like(value) for name, value in weights.items()}
        hidden_grad_next = np.zeros((BATCH, HIDDEN), dtype=np.float32)
        for tick in range(SEQUENCE - 1, -1, -1):
            probs = probabilities[tick].copy()
            probs[np.arange(BATCH), targets[:, tick]] -= 1.0
            probs /= (BATCH * SEQUENCE)
            grads["Why"] += hidden_values[tick].T @ probs
            grads["by"] += probs.sum(axis=0)
            hidden_grad = probs @ weights["Why"].T + hidden_grad_next
            raw_grad = hidden_grad * (1.0 - hidden_values[tick] ** 2)
            x = eye[inputs[:, tick]]
            grads["Wxh"] += x.T @ raw_grad
            grads["Whh"] += hidden_previous[tick].T @ raw_grad
            grads["bh"] += raw_grad.sum(axis=0)
            hidden_grad_next = raw_grad @ weights["Whh"].T

        norm = np.sqrt(sum(float(np.sum(grad * grad)) for grad in grads.values()))
        if norm > 1.0:
            for grad in grads.values():
                grad *= 1.0 / norm
        for name, grad in grads.items():
            first_moment[name] = beta1 * first_moment[name] + (1.0 - beta1) * grad
            second_moment[name] = beta2 * second_moment[name] + (1.0 - beta2) * (grad * grad)
            m_hat = first_moment[name] / (1.0 - beta1 ** step)
            v_hat = second_moment[name] / (1.0 - beta2 ** step)
            weights[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        smoothed_loss = loss if smoothed_loss is None else 0.95 * smoothed_loss + 0.05 * loss
        if step % 10 == 0 or step == TRAIN_STEPS:
            with STATE_LOCK:
                STATE["progress"] = int(step * 100 / TRAIN_STEPS)
                STATE["loss"] = round(float(smoothed_loss), 3)

    np.savez_compressed(WEIGHTS_PATH, vocabulary=np.array(list(vocabulary)), **weights)
    model = TinyJARVIS(vocabulary, weights)
    with MODEL_LOCK:
        MODEL = model
    with STATE_LOCK:
        STATE.update({"modelReady": True, "training": False, "progress": 100, "loss": round(float(smoothed_loss), 3), "error": None})
    return model


def load_model():
    loaded = np.load(WEIGHTS_PATH, allow_pickle=False)
    vocabulary = "".join(str(char) for char in loaded["vocabulary"].tolist())
    weights = {name: loaded[name].astype(np.float32) for name in ("Wxh", "Whh", "bh", "Why", "by")}
    return TinyJARVIS(vocabulary, weights)


def model_status():
    with STATE_LOCK:
        status = dict(STATE)
    status.update({"backendAvailable": True, "modelAvailable": status["modelReady"],
                   "modelName": "JARVIS Tiny RNN", "trainedFromScratch": True})
    return status


def prompt_from_messages(messages):
    lines = []
    for message in messages[-8:]:
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        label = "User" if role == "user" else "JARVIS"
        lines.append(f"{label}: {content[-500:]}")
    return "\n".join(lines) + "\nJARVIS:"


def start_training_in_background():
    def runner():
        try:
            train_model()
        except Exception as error:  # surfaced in the local setup panel
            with STATE_LOCK:
                STATE.update({"training": False, "modelReady": False, "error": str(error)})
    threading.Thread(target=runner, name="jarvis-training", daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    server_version = "JARVISLocal/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        return

    def _send_json(self, code: int, data: dict):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _local_only(self):
        if self.headers.get("Host") != f"{HOST}:{PORT}":
            self._send_json(403, {"error": "JARVIS brain accepts local requests only."})
            return False
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{HOST}:{PORT}":
            self._send_json(403, {"error": "JARVIS brain accepts local requests only."})
            return False
        return True

    def do_GET(self):
        if not self._local_only():
            return
        if self.path == "/status":
            self._send_json(200, model_status())
        else:
            self._send_json(404, {"error": "Not found."})

    def do_POST(self):
        if not self._local_only():
            return
        if self.path == "/train":
            if STATE.get("training"):
                self._send_json(409, {"error": "Training is already running."})
                return
            start_training_in_background()
            self._send_json(202, {"started": True})
            return
        if self.path != "/chat":
            self._send_json(404, {"error": "Not found."})
            return
        if not STATE.get("modelReady") or MODEL is None:
            self._send_json(503, {"error": "The scratch-trained model is still getting ready."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 512_000:
                self._send_json(400, {"error": "The conversation is empty or too long."})
                return
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages or len(messages) > 40:
                self._send_json(400, {"error": "Please send a valid conversation."})
                return
            prompt = prompt_from_messages(messages)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            chunk = ""
            with MODEL_LOCK:
                model = MODEL
                for char in model.generate(prompt):
                    chunk += char
                    if len(chunk) >= 8:
                        payload = (json.dumps({"message": {"content": chunk}}, ensure_ascii=False) + "\n").encode("utf-8")
                        self.wfile.write(f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n")
                        self.wfile.flush()
                        chunk = ""
            if chunk:
                payload = (json.dumps({"message": {"content": chunk}}, ensure_ascii=False) + "\n").encode("utf-8")
                self.wfile.write(f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            if self.wfile.closed:
                return
            try:
                self._send_json(500, {"error": f"Local model error: {error}"})
            except Exception:
                return


def serve():
    global MODEL
    if WEIGHTS_PATH.exists():
        try:
            MODEL = load_model()
            STATE.update({"modelReady": True, "training": False, "progress": 100, "error": None,
                          "trainingChars": len(DATA_PATH.read_text(encoding="utf-8")),
                          "vocabularySize": len(MODEL.vocabulary), "steps": TRAIN_STEPS})
        except Exception as error:
            STATE.update({"modelReady": False, "training": False, "error": f"Could not load the model weights: {error}"})
    else:
        start_training_in_background()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"JARVIS brain listening locally on {HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if mode == "train":
        train_model()
        print(f"Saved JARVIS's scratch-trained weights to {WEIGHTS_PATH}", flush=True)
    elif mode == "serve":
        serve()
    else:
        raise SystemExit("Use: jarvis_brain.py [serve|train]")
