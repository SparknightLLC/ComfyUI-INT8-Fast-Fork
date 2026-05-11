import json
import os

import comfy.sd
import comfy.utils
import folder_paths
import torch
from comfy.cli_args import args


def _is_int8_quantized_module(module):
	if not getattr(module, "_is_quantized", False):
		return False

	weight = getattr(module, "weight", None)
	if not isinstance(weight, torch.Tensor):
		return False

	return weight.dtype == torch.int8


def _collect_modules_for_save_workaround(model_patcher):
	base_model = getattr(model_patcher, "model", None)
	if base_model is None:
		return []

	modules = []
	seen_module_ids = set()

	if hasattr(base_model, "named_modules"):
		for _, module in base_model.named_modules():
			if not _is_int8_quantized_module(module):
				continue
			module_id = id(module)
			if module_id in seen_module_ids:
				continue
			seen_module_ids.add(module_id)
			modules.append(module)

	object_patches = getattr(model_patcher, "object_patches", None)
	if isinstance(object_patches, dict):
		for patch_key, patch_obj in object_patches.items():
			if not isinstance(patch_key, str):
				continue
			if not patch_key.startswith("diffusion_model."):
				continue
			if not _is_int8_quantized_module(patch_obj):
				continue

			try:
				target_module = comfy.utils.get_attr(base_model, patch_key)
			except Exception:
				target_module = None

			if target_module is None:
				continue

			module_id = id(target_module)
			if module_id in seen_module_ids:
				continue
			seen_module_ids.add(module_id)
			modules.append(target_module)

	return modules


def _set_comfy_patched_weights_flag(modules):
	flag_states = []
	for module in modules:
		had_flag = hasattr(module, "comfy_patched_weights")
		old_flag = getattr(module, "comfy_patched_weights", False) if had_flag else False
		flag_states.append((module, had_flag, old_flag))
		module.comfy_patched_weights = True
	return flag_states


def _restore_comfy_patched_weights_flag(flag_states):
	for module, had_flag, old_flag in flag_states:
		if had_flag:
			module.comfy_patched_weights = old_flag
		else:
			delattr(module, "comfy_patched_weights")


class INT8ModelSave:
	def __init__(self):
		self.output_dir = folder_paths.get_output_directory()

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"model": ("MODEL",),
				"filename_prefix": ("STRING", {"default": "int8_models/INT8_Model"}),
			},
			"hidden": {
				"prompt": "PROMPT",
				"extra_pnginfo": "EXTRA_PNGINFO",
			},
		}

	RETURN_TYPES = ()
	FUNCTION = "save"
	OUTPUT_NODE = True
	CATEGORY = "loaders"
	DESCRIPTION = "Save MODEL outputs that include INT8-patched layers with a DynamicVRAM-safe save path."

	def save(self, model, filename_prefix, prompt=None, extra_pnginfo=None):
		full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
			filename_prefix,
			self.output_dir,
		)

		prompt_info = ""
		if prompt is not None:
			prompt_info = json.dumps(prompt)

		metadata = {}
		if not args.disable_metadata:
			metadata["prompt"] = prompt_info
			if extra_pnginfo is not None:
				for key, value in extra_pnginfo.items():
					metadata[key] = json.dumps(value)

		output_checkpoint = f"{filename}_{counter:05}_.safetensors"
		output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

		modules_to_patch = _collect_modules_for_save_workaround(model)
		flag_states = _set_comfy_patched_weights_flag(modules_to_patch)

		try:
			comfy.sd.save_checkpoint(output_checkpoint, model, metadata=metadata)
		finally:
			_restore_comfy_patched_weights_flag(flag_states)

		return {}

