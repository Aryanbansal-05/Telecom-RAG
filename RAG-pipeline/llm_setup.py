import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# GGUF quantized model -- roughly 2.5GB in memory, safe on 8GB RAM.
# Qwen3-4B-Instruct-2507 specifically benchmarks well on open/closed-book QA
# tasks and instruction-following (important for staying faithful to
# retrieved context) -- a meaningful step up from the Qwen2.5 generation.
MODEL_REPO = "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF"
MODEL_FILE = "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

_llm = None


def load_llm():
    global _llm
    if _llm is not None:
        return _llm

    print(f"Downloading/loading {MODEL_FILE} (first run only, ~2GB)...")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

    print("Loading model into memory...")
    _llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=os.cpu_count(),
        n_gpu_layers=-1,  # uses Metal acceleration on Apple Silicon if available
        verbose=False,
    )
    print("LLM ready.")
    return _llm


def get_llm_info() -> dict:
    """Return live info about the LLM. Safe to call before load."""
    return {
        "model_repo": MODEL_REPO,
        "model_file": MODEL_FILE,
        "model_name": "Qwen3-4B-Instruct",  # human-readable short name
        "quantization": "Q4_K_M (GGUF)",
        "n_ctx": 4096,
        "n_threads": os.cpu_count(),
        "llm_loaded": _llm is not None,
    }


def generate(prompt, max_tokens=400, temperature=0.3):
    """
    Uses llama-cpp-python's chat completion, which reads the chat template
    embedded in the GGUF file itself -- no need to hand-write Qwen's
    instruction format.
    """
    llm = load_llm()
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return output["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    response = generate("What is 5G NR handover? Answer in one sentence.")
    print("Response:", response)

    # Skips Python's normal exit/cleanup path, which currently triggers a
    # known upstream bug in llama.cpp's Metal backend teardown (crashes
    # AFTER the real work above is already done and printed). Not needed
    # once this is imported into a long-running server instead of run
    # standalone like this.
    import os
    os._exit(0)