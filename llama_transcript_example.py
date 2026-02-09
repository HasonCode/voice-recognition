"""
Example: feed transcript text into Llama 3.2 3B Instruct.

Prerequisites:
  1. Hugging Face login and Llama 3.2 license accepted:
       pip install huggingface_hub
       huggingface-cli login
  2. Install: pip install transformers accelerate
     Optional (4-bit, less VRAM): pip install bitsandbytes

Usage:
  python llama_transcript_example.py "Your transcript here"
  python llama_transcript_example.py --file transcription.txt
  python llama_transcript_example.py --file transcription.txt --prompt "Summarize this:"
"""

import argparse
import sys
from pathlib import Path

# Add src for optional use of project packages
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def load_model(use_4bit=True):
    """Load Llama 3.2 3B Instruct and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except ImportError:
            print("bitsandbytes not installed; loading in full precision (need more VRAM).")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    return model, tokenizer


def chat_format(instruction: str, transcript: str) -> str:
    """Format as a single user turn for Llama 3.2 Instruct."""
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\nTranscript:\n{transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def generate(model, tokenizer, prompt: str, max_new_tokens=256, do_sample=True):
    """Run generation and return decoded text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run Llama 3.2 3B Instruct on transcript text.")
    parser.add_argument("text", nargs="?", default=None, help="Transcript text (or use --file)")
    parser.add_argument("--file", "-f", type=str, help="Read transcript from file")
    parser.add_argument("--prompt", "-p", type=str, default="Summarize the following transcript briefly:",
                        help="Instruction before the transcript")
    parser.add_argument("--no-4bit", action="store_true", help="Load full precision (more VRAM)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens to generate")
    args = parser.parse_args()

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

    print("Loading model...")
    model, tokenizer = load_model(use_4bit=not args.no_4bit)
    prompt = chat_format(args.prompt, transcript)
    print("Generating...")
    out = generate(model, tokenizer, prompt, max_new_tokens=args.max_tokens)
    print(out.strip())


if __name__ == "__main__":
    main()
