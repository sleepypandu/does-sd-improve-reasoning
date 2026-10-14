"""Per-module LoRA weight sync from a DeepSpeed ZeRO-3 model into a colocated vLLM engine.

TRL's ``VLLMGeneration.sync_weights`` gathers *every* parameter of a PEFT model at once before
merging the adapters, which under ZeRO-3 materializes the full unsharded model on every rank
next to the vLLM engine. LoRA merging is local to each linear layer, so gathering one tuner
layer at a time pushes the same merged weights with a peak of a single module.
"""

from accelerate.utils import is_peft_model


def iter_merged_params(model, gather):
    """Yield ``(vllm_name, tensor)`` for a ZeRO-3 PEFT model, merging LoRA one module at a time.

    ``gather`` is a context manager factory taking a list of parameters (e.g.
    ``deepspeed.zero.GatheredParameters``). Names are mapped exactly as in TRL's PEFT branch.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    owner = {}
    for module_name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            for param_name, _ in module.named_parameters():
                owner[f"{module_name}.{param_name}" if module_name else param_name] = module

    for raw_name, param in model.named_parameters():
        name = raw_name.removeprefix("base_model.model.").replace(".base_layer", "")
        if model.prefix in name or "original_module" in name:
            continue
        name = name.replace("modules_to_save.default.", "")
        module = owner.get(raw_name)
        if module is None:
            with gather([param]):
                yield name, param.data
        else:
            with gather(list(module.parameters())):
                module.merge()
                yield name, param.data
                module.unmerge()


def patch_grpo_vllm_sync_for_zero3():
    """Route TRL's VLLMGeneration.sync_weights through iter_merged_params under ZeRO-3 + PEFT."""
    try:
        from trl.generation.vllm_generation import VLLMGeneration, empty_cache
    except ImportError:
        return
    if getattr(VLLMGeneration.sync_weights, "_zero3_per_module", False):
        return
    original = VLLMGeneration.sync_weights

    def sync_weights(self):
        if not (getattr(self._dist, "is_zero3", False) and is_peft_model(self.model)):
            return original(self)
        if self.mode == "colocate" and self.enable_sleep_mode:
            empty_cache()
            self.llm.wake_up(tags=["weights"])
        for name, tensor in iter_merged_params(self.model, self._dist.gather_params):
            self._push_param_to_vllm(self._fix_param_name_to_vllm(name), tensor)
        if self.mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.mode == "colocate":
            self.llm.reset_prefix_cache()

    sync_weights._zero3_per_module = True
    VLLMGeneration.sync_weights = sync_weights
