# Per-model settings of the paper runs, sourced by the train_*.sh scripts.
# Every preset keeps the effective batch NUM_PROCESSES x PD x GA = 32 (GRPO: 4 prompts x 8 rollouts).
# ALLOC_CONF is exported as PYTORCH_CUDA_ALLOC_CONF; it only affects memory fragmentation.

opsd_preset() {  # <model basename> <teacher_thinking True|False>
    ACCELERATE_CONFIG=accelerate.yaml VLLM_TP=1 VLLM_MEM=0.6 ALLOC_CONF=
    TEACHER_ARGS=(--fixed_teacher True)
    case "$1" in
        Qwen3-1.7B) NUM_PROCESSES=4 PD=4 GA=2 TOKEN_CLIP=0.05 ;;
        Qwen3-4B) NUM_PROCESSES=8 PD=4 GA=1 TOKEN_CLIP=0.05 ;;
        Qwen3-8B) NUM_PROCESSES=8 PD=2 GA=2 TOKEN_CLIP=0.06 ;;
        Qwen3-14B)
            NUM_PROCESSES=8 PD=1 GA=4 TOKEN_CLIP=0.06 ACCELERATE_CONFIG=accelerate_zero3.yaml
            VLLM_TP=8 VLLM_MEM=0.16 ALLOC_CONF=expandable_segments:True ;;
        Olmo-3-7B-Instruct)
            # Olmo has no thinking switch: the thinking teacher is the separate Olmo-3-7B-Think checkpoint.
            NUM_PROCESSES=8 PD=1 GA=4 TOKEN_CLIP=0.06 ALLOC_CONF=expandable_segments:True
            if [ "$2" = True ]; then
                TEACHER_ARGS=(--teacher_model_name_or_path allenai/Olmo-3-7B-Think)
                ACCELERATE_CONFIG=accelerate_zero3.yaml
            fi ;;
        *) echo "no OPSD preset for $1" >&2; return 1 ;;
    esac
}

grpo_preset() {  # <model basename>
    ACCELERATE_CONFIG=accelerate.yaml PD=1 ALLOC_CONF=
    VLLM_ARGS=()
    case "$1" in
        Qwen3-1.7B) NUM_PROCESSES=4 GA=8 ;;
        Qwen3-4B | Qwen3-8B) NUM_PROCESSES=8 GA=4 ;;
        Qwen3-14B)
            NUM_PROCESSES=8 GA=4 ACCELERATE_CONFIG=accelerate_zero3.yaml ALLOC_CONF=expandable_segments:True
            VLLM_ARGS=(--vllm_tensor_parallel_size 8 --vllm_gpu_memory_utilization 0.3 --vllm_enable_sleep_mode True) ;;
        Olmo-3-7B-Instruct)
            # Olmo's 65k native context would not leave room for vLLM's KV cache at the default utilization.
            NUM_PROCESSES=8 GA=4 VLLM_ARGS=(--vllm_max_model_length 32768) ;;
        *) echo "no GRPO preset for $1" >&2; return 1 ;;
    esac
}

sft_preset() {  # <model basename>
    ACCELERATE_CONFIG=accelerate.yaml ALLOC_CONF=
    case "$1" in
        Qwen3-1.7B) NUM_PROCESSES=4 PD=2 GA=4 ;;
        Qwen3-4B | Qwen3-8B | Olmo-3-7B-Instruct) NUM_PROCESSES=8 PD=2 GA=2 ;;
        Qwen3-14B)
            NUM_PROCESSES=8 PD=1 GA=4 ACCELERATE_CONFIG=accelerate_zero3.yaml ALLOC_CONF=expandable_segments:True ;;
        *) echo "no SFT preset for $1" >&2; return 1 ;;
    esac
}

free_port() {
    python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

launch() {  # run a command, or only print it when DRY_RUN=1
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf '%q ' "$@"
        echo
    else
        "$@"
    fi
}
