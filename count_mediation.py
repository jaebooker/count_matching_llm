import argparse
import ast
import math
import os
import random
from collections import defaultdict
from typing import List, Tuple
import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME = "Nexusflow/Starling-LM-7B-beta"
DEVICE = "cpu"  # Force CPU usage to avoid GPU compatibility issues

# Fall‑back vocab if we need to synth‑generate examples
FRUITS = [
    "apple",
    "banana",
    "orange",   
    "grape",
    "pear",
    "peach",
    "plum",
    "cherry",
]
NON_FRUITS = [
    "dog",
    "cat",
    "bus",
    "bowl",
    "chair",
    "book",
    "lamp",
]

# --------------------------------------------------------------------------- #
# Prompt helpers
# --------------------------------------------------------------------------- #

def build_prompt(word_type: str, words: List[str]) -> str:
    return (
        "Given a list of words and a category, count how many words in the list belong to that category.\n"
        f"Category: {word_type}\n"
        f"List: {words}\n"
        "Count: "
    )


def prompt_from_row(row) -> Tuple[str, int]:
    word_type: str = row["type"]
    words: List[str] = ast.literal_eval(row["list"])
    answer: int = int(row["answer"])
    return build_prompt(word_type, words), answer


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #

def collect_last_hidden(model, tokenizer, prompts: List[str], layers: List[int]) -> dict:
    """Collect final‑token hidden states for each layer."""
    H = {ℓ: [] for ℓ in layers}
    for prompt in tqdm(prompts, desc="Collecting hidden states"):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}  # Move to device
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for ℓ in layers:
            H[ℓ].append(out.hidden_states[ℓ][0, -1].cpu())
    return {ℓ: torch.stack(h) for ℓ, h in H.items()}


def fit_linear_probe(X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, float]:
    """Fit a linear probe and return the model + R² score."""
    reg = LinearRegression().fit(X, y)
    r2 = r2_score(y, reg.predict(X))
    return reg, r2


# --------------------------------------------------------------------------- #
# Activation‑patching utilities
# --------------------------------------------------------------------------- #
class PatchLayer:
    """Context‑manager: capture donor activation at layer ℓ and replay into target."""

    def __init__(self, model, layer_id: int):
        self.model = model
        self.ℓ = layer_id
        self.store = None
        self.layer = model.model.layers[layer_id]

    def __enter__(self):
        self.h1 = self.layer.register_forward_hook(self._capture)
        return self

    def _capture(self, module, inp, out):
        # For Mistral, the hidden state is the first element of the output tuple
        if isinstance(out, tuple):
            hidden_state = out[0]
        else:
            hidden_state = out
        self.store = hidden_state.detach().clone()

    def patch(self, *args):
        def patch_hook(module, inp, out):
            if isinstance(out, tuple):
                # Replace the hidden state (first element) while keeping other elements
                return (self.store,) + out[1:]
            return self.store
        return self.layer.register_forward_hook(patch_hook)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h1.remove()


@torch.no_grad()
def forward_logits(model, tokenizer, prompt: str):
    ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model(**ids, output_hidden_states=True)
    # Get the last token's logits
    return outputs.logits[:, -1, :].cpu()


DIGIT_TOKENS = [str(i) for i in range(10)]

def decode_digit(logits, tokenizer):
    digit_ids = tokenizer.convert_tokens_to_ids(DIGIT_TOKENS)
    return logits[:, digit_ids].argmax(dim=-1).item()


def run_patched_pair(model, tokenizer, promptA: str, promptB: str, layer: int) -> Tuple[float, float, float]:
    # donor pass – capture
    with PatchLayer(model, layer) as ctx:
        _ = forward_logits(model, tokenizer, promptB)
        # patch into A
        handle = ctx.patch()
        logits_Apatched = forward_logits(model, tokenizer, promptA)
        handle.remove()
    logits_A = forward_logits(model, tokenizer, promptA)
    logits_B = forward_logits(model, tokenizer, promptB)
    return decode_digit(logits_A, tokenizer), decode_digit(logits_B, tokenizer), decode_digit(logits_Apatched, tokenizer)


# --------------------------------------------------------------------------- #
# Experiment pipeline
# --------------------------------------------------------------------------- #

def load_or_generate(args):
    """Return DataFrame with columns prompt, answer."""
    if args.data_csv and os.path.exists(args.data_csv):
        print(f"Loading data from {args.data_csv}")
        df = pd.read_csv(args.data_csv)
        # drop rows without numeric answer (e.g., "```")
        df = df[pd.to_numeric(df["answer"], errors="coerce").notnull()].copy()
        df["answer"] = df["answer"].astype(int)
        rows = df.apply(prompt_from_row, axis=1).tolist()
        return pd.DataFrame(rows, columns=["prompt", "count"]).sample(n=500, random_state=0)  # Use 500 examples for testing
    else:
        raise ValueError(f"Data file {args.data_csv} not found!")


def mediation_pipeline(args):
    torch.manual_seed(0); random.seed(0); np.random.seed(0)

    print("Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use float32 for CPU
        low_cpu_mem_usage=True,
        device_map="cpu"            # Force CPU usage
    ).eval()

    # 1. Dataset
    df = load_or_generate(args)
    train = df.sample(frac=0.8, random_state=0)

    # 2. Representation search (final token → answer)
    layers = list(range(model.config.num_hidden_layers))
    H = collect_last_hidden(model, tokenizer, train.prompt.tolist(), layers)
    layer_r2 = {}
    y = train["count"].values
    for ℓ in layers:
        reg, r2 = fit_linear_probe(H[ℓ].numpy(), y)
        layer_r2[ℓ] = r2
    top_layers = sorted(layer_r2, key=layer_r2.get, reverse=True)[: args.top_k]
    print("Top layers by R²:", [(ℓ, f"{layer_r2[ℓ]:.3f}") for ℓ in top_layers])

    # 3. Mediation test via activation patching
    results = defaultdict(list)
    for _ in tqdm(range(args.num_mediation_pairs)):
        # choose two prompts with different answers
        A, B = df.sample(2, replace=False).itertuples(index=False)
        if A.count == B.count:
            continue  # skip same‑count pairs
        for ℓ in top_layers:
            yA, yB, yApatched = run_patched_pair(model, tokenizer, A.prompt, B.prompt, ℓ)
            TE = yB - yA
            ME = yApatched - yA
            results[ℓ].append((TE, ME))

    print("\nLayer‑wise proportion mediated (mean ± sem):")
    for ℓ in top_layers:
        pairs = results[ℓ]
        if not pairs:
            print(f"Layer {ℓ:2d}: no valid pairs")
            continue
        te, me = zip(*pairs)
        te = np.array(te); me = np.array(me)
        prop = np.divide(me, te, out=np.zeros_like(me, float), where=te!=0)
        m, se = prop.mean(), prop.std(ddof=1)/math.sqrt(len(prop))
        print(f"Layer {ℓ:2d}: {m:.2f} ± {se:.02f}  (n={len(prop)})")

    # Save results
    results_dict = {
        "layer_r2": {str(k): float(v) for k, v in layer_r2.items()},
        "top_layers": top_layers,
        "mediation_results": {
            str(ℓ): {
                "mean": float(prop.mean()),
                "sem": float(prop.std(ddof=1)/math.sqrt(len(prop))),
                "n": len(prop)
            }
            for ℓ in top_layers
        }
    }
    
    with open("mediation_analysis_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mediation analysis on count-matching benchmark")
    parser.add_argument("--data_csv", type=str, default="count_matching_words_benchmark_with_answers.csv",
                      help="Path to CSV file with benchmark data")
    parser.add_argument("--top_k", type=int, default=4,
                      help="Number of top layers to analyze")
    parser.add_argument("--num_mediation_pairs", type=int, default=50,
                      help="Number of mediation pairs to test")
    args = parser.parse_args()
    
    mediation_pipeline(args)
