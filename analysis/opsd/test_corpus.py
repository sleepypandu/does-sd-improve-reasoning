from corpus import is_orphan_close, parse_file


def test_new_run_names():
    r = parse_file("eval_results_aime24_Qwen3-8B_opsd-model-Qwen3-8B-teacher_thinking-True-privileged_mode-swap"
                   "-loss_mask-last_10pct-seed-2_checkpoint-75_nonthinking_temp1.0_valn12.json")
    assert r == {"model": "Qwen3-8B", "benchmark": "aime24", "eval_thinking": False,
                 "arm": "opsd/tt=T/z=swap/mask=last_10pct", "seed": 2, "step": 75, "val_n": 12}
    r = parse_file("eval_results_hmmt25_Qwen3-14B_grpo-model-Qwen3-14B-enable_thinking-False-seed-0_checkpoint-500"
                   "_thinking_temp1.0_valn12.json")
    assert (r["arm"], r["seed"], r["step"], r["eval_thinking"]) == ("grpo_nothink", 0, 500, True)
    assert parse_file("eval_results_aime25_Olmo-3-7B-Instruct_sft-model-Olmo-3-7B-Instruct-seed-1_checkpoint-20"
                      "_nonthinking_temp1.0_valn12.json")["arm"] == "sft"


def test_paper_run_names():
    r = parse_file("eval_results_aime24_Olmo-3-7B-Instruct_opsd-train-opsd_olmo-model-Olmo-3-7B-Instruct-dataset_name"
                   "-Openthoughts_math_30k_opsd-student_thinking-False-teacher-True-privileged_mode-none-seed-3"
                   "_checkpoint-140_nonthinking_temp1.0_valn12.json")
    assert (r["arm"], r["seed"], r["step"]) == ("opsd/tt=T/z=none", 3, 140)
    r = parse_file("eval_results_aime25_Qwen3-14B_opsd-train-grpo_14b_think-model-Qwen3-14B-dataset_name"
                   "-Openthoughts_math_30k_opsd-seed-1-enable_thinking-False_checkpoint-50_nonthinking_temp1.0_valn12.json")
    assert r["arm"] == "grpo_nothink"
    r = parse_file("eval_results_aime24_Qwen3-4B_opsd-train-grpo_nothink-model-Qwen3-4B-dataset_name"
                   "-Openthoughts_math_30k_opsd-seed-0_checkpoint-50_thinking_temp1.0_valn12.json")
    assert r["arm"] == "grpo_nothink"
    assert parse_file("eval_results_aime24_Qwen3-8B_thinking_temp1.0_valn12.json")["arm"] == "base"


def test_excluded_files():
    for name in [
        "eval_results_aime24_Qwen3-8B_thinking_temp1.0_valn12_seed-3.json",
        "eval_results_aime24_Qwen3-8B_nonthinking_temp1.0_valn12_prompt-priv_none_teacher.json",
        "eval_results_aime24_Qwen3-8B_opsd-train-grpo-model-Qwen3-8B-dataset_name-curatedthoughts_math_30k_opsd"
        ".parquet-seed-0_checkpoint-50_thinking_temp1.0_valn12.json",
        "eval_results_aime24_Qwen3-8B_opsd-train-sft-model-Qwen3-8B-dataset_name-Openthoughts_math_30k_opsd"
        "_checkpoint-20_thinking_temp1.0_valn12.json",
        "eval_results_aime24_Qwen3-8B_opsd-train-sft_cot-model-Qwen3-8B-dataset_name-Openthoughts_math_30k_opsd"
        "-seed-0_checkpoint-20_thinking_temp1.0_valn12.json",
        "eval_results_openthoughts500_Qwen3-8B_thinking_temp1.0_valn1.json",
    ]:
        assert parse_file(name) is None, name


def test_orphan_close():
    assert is_orphan_close("reasoning ... </think> answer")
    assert not is_orphan_close("<think> reasoning </think> answer")
    assert not is_orphan_close("plain answer")
