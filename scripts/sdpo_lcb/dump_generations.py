"""Sample and save per-problem generations from a base model or a training checkpoint.

The sampling matches the training rollout (train.parquet, 8 samples per problem, temperature
1.0, top_p 1.0, 8192 new tokens, thinking off), which is what Figure 1 counts unique solutions
over. Generations are not graded here; grade_dump.py does that on CPU.

    python scripts/sdpo_lcb/dump_generations.py --model Qwen/Qwen3-14B --out dumps/base-seed0.json.gz
    python scripts/sdpo_lcb/dump_generations.py --ckpt outputs/sdpo_lcb/lcb-sdpo-Qwen3-14B-seed0/global_step_200 \
        --out dumps/lcb-sdpo-Qwen3-14B-seed0-step200.json.gz
"""
import argparse
import glob
import gzip
import json
import os
import subprocess
import sys

MAX_PROMPT_LENGTH = 2048
MAX_RESPONSE_LENGTH = 8192
TEMPERATURE = 1.0
TOP_P = 1.0
N_SAMPLES = 8


def merged_hf_dir(ckpt):
    """FSDP actor shards -> HF weights in <ckpt>/actor/huggingface_merged (cached)."""
    actor = os.path.join(ckpt, "actor")
    target = os.path.join(actor, "huggingface_merged")
    if not glob.glob(os.path.join(target, "*.safetensors")):
        # A separate process, so this one stays CUDA-free until vLLM forks its engine.
        subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                        "--local_dir", actor, "--target_dir", target], check=True)
    return target


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="HF model id or path (the untrained base model)")
    src.add_argument("--ckpt", help="a global_step_<N> directory written by train.sh")
    ap.add_argument("--data", default="data/sdpo_lcb/lcb_exp/train.parquet")
    ap.add_argument("--out", required=True, help="output .json.gz")
    ap.add_argument("--seed", type=int, default=0, help="vLLM sampling seed")
    ap.add_argument("--max_problems", type=int, default=None)
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    weights = args.model or merged_hf_dir(args.ckpt)
    tokenizer = AutoTokenizer.from_pretrained(weights)
    df = pd.read_parquet(args.data)
    if args.max_problems:
        df = df.head(args.max_problems)
    prompts = [tokenizer.apply_chat_template(list(p), add_generation_prompt=True, tokenize=True,
                                             enable_thinking=False) for p in df["prompt"]]
    keep = [i for i, p in enumerate(prompts) if len(p) <= MAX_PROMPT_LENGTH]
    df, prompts = df.iloc[keep], [prompts[i] for i in keep]

    llm = LLM(model=weights, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)
    sampling = SamplingParams(n=N_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P, top_k=-1,
                              max_tokens=MAX_RESPONSE_LENGTH, seed=args.seed)
    outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], sampling)

    rows = []
    for (_, row), out in zip(df.iterrows(), outputs):
        for i, completion in enumerate(out.outputs):
            rows.append({
                "data_source": row["data_source"],
                "problem_index": int(row["extra_info"]["index"]),
                "sample": i,
                "finish_reason": completion.finish_reason,
                "n_tokens": len(completion.token_ids),
                "text": completion.text,
            })
    meta = {
        "weights": weights, "model": args.model, "ckpt": args.ckpt, "data": os.path.abspath(args.data),
        "sampling": {"n": N_SAMPLES, "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": -1,
                     "max_tokens": MAX_RESPONSE_LENGTH, "seed": args.seed, "enable_thinking": False},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with gzip.open(args.out, "wt") as f:
        json.dump({"meta": meta, "rows": rows}, f)
    print(f"wrote {args.out}: {len(df)} problems x {N_SAMPLES} samples")


if __name__ == "__main__":
    main()
