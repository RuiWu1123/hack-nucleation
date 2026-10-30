"""Export only the LoRA adapter of a verl FSDP checkpoint (no merged full model).

Writes <out_dir>/adapter_config.json and adapter_model.safetensors, loadable with
`PeftModel.from_pretrained(<base model>, out_dir)`. The vision patch embedding matches the
"proj" target name but is never trained (and not in the checkpoint), so it is excluded.
Usage: python scripts/export_lora.py runs/<run>/checkpoints/global_step_50/actor <out_dir> --base_model Qwen/Qwen3.5-4B
"""

import json
import os
import shutil

import fire

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def main(actor_dir: str, out_dir: str, base_model: str):
    tmp_dir = out_dir.rstrip("/") + ".tmp"
    config = ModelMergerConfig(
        operation="merge", backend="fsdp", local_dir=actor_dir, target_dir=tmp_dir,
        hf_model_config_path=os.path.join(actor_dir, "huggingface"),
    )
    merger = FSDPModelMerger(config)
    world_size = merger._get_world_size()
    rank_zero_state_dict = merger._load_rank_zero_state_dict(world_size)
    mesh, mesh_dim_names = merger._extract_device_mesh_info(rank_zero_state_dict, world_size)
    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_dim_names)
    state_dict = merger._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)
    merger.save_lora_adapter(state_dict)

    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.move(os.path.join(tmp_dir, "lora_adapter"), out_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    config_path = os.path.join(out_dir, "adapter_config.json")
    with open(config_path) as f:
        adapter_config = json.load(f)
    adapter_config["base_model_name_or_path"] = base_model
    adapter_config["exclude_modules"] = ["model.visual.patch_embed.proj"]
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=4)
    print(f"saved LoRA adapter to {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
