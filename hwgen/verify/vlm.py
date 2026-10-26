"""Claude vision client for verification: structured JSON answers, an on-disk
response cache (so reruns are free), refusal handling and cost accounting."""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = "claude-opus-5-5"
PRICE_PER_MTOK = {"claude-opus-5-5": (4.00, 20.00)}  # (input, output) USD


def encode_image(img: np.ndarray, max_side: int = 1568, quality: int = 92) -> tuple[str, bytes]:
    """BGR image -> (media type, JPEG bytes), downscaled so the long side fits
    what the API would resize to anyway (no point paying for more pixels)."""
    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w))
    if k < 1.0:
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "image/jpeg", buf.tobytes()


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    refusals: int = 0
    per_model: dict = field(default_factory=dict)

    def cost(self) -> float:
        total = 0.0
        for model, (i, o) in self.per_model.items():
            pin, pout = PRICE_PER_MTOK.get(model, PRICE_PER_MTOK[DEFAULT_MODEL])
            total += i / 1e6 * pin + o / 1e6 * pout
        return total


class VLMRefusal(Exception):
    pass


class VLMClient:
    """`ask(parts, schema)` where parts is a list of str (text) and np.ndarray
    (BGR image), in the order they should appear in the user message."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: Path = Path("data/cache/vlm"),
                 effort: str = "medium", dry_run: bool = False, dry_run_dir: Path | None = None):
        self.model = model
        self.effort = effort
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.dry_run_dir = dry_run_dir
        self.usage = Usage()
        self._client = None
        if not dry_run:
            import anthropic  # resolves ANTHROPIC_API_KEY / auth profile from the environment

            self._client = anthropic.Anthropic(max_retries=4)

    def _content(self, parts: list) -> tuple[list[dict], str]:
        content, h = [], hashlib.sha256()
        for p in parts:
            if isinstance(p, str):
                content.append({"type": "text", "text": p})
                h.update(p.encode())
            else:
                media, data = encode_image(p)
                content.append({"type": "image", "source": {"type": "base64", "media_type": media,
                                                            "data": base64.standard_b64encode(data).decode()}})
                h.update(data)
        return content, h.hexdigest()

    def ask(self, parts: list, schema: dict, system: str, max_tokens: int = 8000) -> dict:
        content, digest = self._content(parts)
        key = hashlib.sha256(f"{self.model}|{self.effort}|{system}|{json.dumps(schema, sort_keys=True)}|{digest}".encode()).hexdigest()
        cache = self.cache_dir / f"{key}.json"
        if cache.exists():
            self.usage.cached += 1
            return json.loads(cache.read_text())["result"]

        if self.dry_run:
            # Record what would be sent: images and prompt text, no API call.
            if self.dry_run_dir is not None:
                d = Path(self.dry_run_dir) / key[:12]
                d.mkdir(parents=True, exist_ok=True)
                n = 0
                for p in parts:
                    if not isinstance(p, str):
                        cv2.imwrite(str(d / f"image_{n:02d}.jpg"), p)
                        n += 1
                (d / "prompt.txt").write_text(system + "\n\n---\n\n" + "\n".join(p if isinstance(p, str) else "[image]" for p in parts))
            self.usage.calls += 1
            return {}

        resp = self._client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # a declined request is re-run on Anthropic's recommended fallback model
        )
        self.usage.calls += 1
        served = resp.model or self.model
        i, o = resp.usage.input_tokens, resp.usage.output_tokens
        self.usage.input_tokens += i
        self.usage.output_tokens += o
        pi, po = self.usage.per_model.get(served, (0, 0))
        self.usage.per_model[served] = (pi + i, po + o)
        if resp.stop_reason == "refusal":
            self.usage.refusals += 1
            raise VLMRefusal(getattr(resp.stop_details, "category", None))
        text = next(b.text for b in resp.content if b.type == "text")
        result = json.loads(text)
        cache.write_text(json.dumps({"model": served, "result": result,
                                     "usage": {"input": i, "output": o}}, ensure_ascii=False))
        return result
