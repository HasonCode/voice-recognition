"""
Example: feed transcript into Llama 3.2 3B Instruct ON THE EDGE using llama.cpp (GGUF).

Runs on Jetson Orin Nano, Raspberry Pi, or any machine with enough RAM.
Uses llama-cpp-python so you can load a GGUF model and generate in process
(e.g. from pipeline_live_test transcript).

Prerequisites:
  1. Download a GGUF model (no Hugging Face gating for many):
     - https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
     - e.g. Llama-3.2-3B-Instruct-Q4_K_M.gguf (~2 GB)
  2. Install:
       pip install llama-cpp-python
     On Jetson with CUDA (optional):
       CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

Usage:
  python llama_transcript_edge_example.py --model path/to/Llama-3.2-3B-Instruct-Q4_K_M.gguf --file transcription.txt
  python llama_transcript_edge_example.py --model model.gguf "Your transcript here"
"""

import argparse
import sys
from pathlib import Path

# Default chat template for Llama 3.2 Instruct (no special tokens required for basic prompt)
def format_prompt(instruction: str, transcript: str) -> str:
    return f"{instruction}\n\nTranscript:\n{transcript}"


def main():
    parser = argparse.ArgumentParser(description="Run Llama 3.2 3B (GGUF) on transcript — edge.")
    parser.add_argument("--model", "-m", required=True, help="Path to .gguf model file")
    parser.add_argument("--file", "-f", type=str, help="Read transcript from file")
    parser.add_argument("text", nargs="?", default=None, help="Transcript text")
    parser.add_argument("--prompt", "-p", type=str, default="Summarize the following transcript briefly:",
                        help="Instruction for the model")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="Layers to put on GPU (-1 = all if available, 0 = CPU only)")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context size")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        transcript = path.read_text(encoding="utf-8").strip()
    elif args.text:
        transcript = args.text.strip()
    else:
        print("Provide transcript as argument or use --file.", file=sys.stderr)
        sys.exit(1)

    if not transcript:
        print("Transcript is empty.", file=sys.stderr)
        sys.exit(1)

    try:
        from llama_cpp import Llama
    except ImportError:
        print("Install llama-cpp-python: pip install llama-cpp-python", file=sys.stderr)
        sys.exit(1)

    print("Loading GGUF model (edge)...")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )
    prompt = format_prompt(args.prompt, transcript)
    print("Generating...")
    out = llm(
        prompt,
        max_tokens=args.max_tokens,
        temperature=0.7,
        stop=["</s>", "<|eot_id|>"],
        echo=False,
    )
    text = out["choices"][0].get("text", "").strip()
    print(text)


if __name__ == "__main__":
    main()
