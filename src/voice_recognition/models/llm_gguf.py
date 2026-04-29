"""Lazy-loaded Llama (GGUF) via llama-cpp-python for short Q&A replies."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

# Llama 3.x Instruct chat framing (matches Meta chat templates for 3.2).
_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer in about two or three short sentences only. "
    "Be accurate and direct. If you do not know, say so briefly."
)


def find_gguf_in_dir(models_dir: Path) -> Optional[Path]:
    """Find a GGUF model, preferring smaller Q4-style files for Jetson latency."""
    if not models_dir.is_dir():
        return None
    ggufs = sorted(models_dir.glob("*.gguf"))
    if not ggufs:
        return None

    def score(path: Path) -> tuple[int, int, int, str]:
        name = path.name.lower()
        # Lower tuple is preferred.
        # 1) Explicit alias wins.
        explicit = 0 if "4qllamasmall" in name else 1
        # 2) Prefer Q4 quant over larger quants for speed/memory.
        q4 = 0 if ("q4" in name or "_q4" in name or "-q4" in name) else 1
        # 3) Prefer smaller files.
        size = path.stat().st_size
        return (explicit, q4, size, name)

    return min(ggufs, key=score)


def build_llama3_instruct_prompt(system: str, user_message: str) -> str:
    return (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message.strip()}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


class LazyLlamaGguf:
    """Thread-safe lazy loader + generate for one GGUF path."""

    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int = 1024,
        n_gpu_layers: int = 0,
        n_batch: int = 128,
        n_threads: Optional[int] = None,
    ):
        self._path = Path(model_path)
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_batch = n_batch
        self._n_threads = n_threads
        self._llm: Any = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> Any:
        with self._lock:
            if self._llm is not None:
                return self._llm
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "llama-cpp-python is required for GGUF answers. "
                    "Install with: pip install llama-cpp-python"
                ) from e
            if not self._path.is_file():
                raise FileNotFoundError(f"GGUF model not found: {self._path}")
            self._llm = Llama(
                model_path=str(self._path),
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                n_batch=self._n_batch,
                n_threads=self._n_threads,
                verbose=False,
            )
            return self._llm

    def generate_short_reply(
        self,
        user_question: str,
        *,
        system: str = _DEFAULT_SYSTEM,
        max_tokens: int = 64,
        temperature: float = 0.15,
    ) -> str:
        """Run the model; return trimmed assistant text."""
        llm = self._ensure_loaded()
        prompt = build_llama3_instruct_prompt(system, user_question)
        with self._lock:
            out = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stop=["<|eot_id|>", "</s>"],
                echo=False,
            )
        text = out["choices"][0].get("text", "").strip()
        return text
