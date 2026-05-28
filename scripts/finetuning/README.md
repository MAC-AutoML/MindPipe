# Compression LoRA Finetuning Scripts

This directory is split by model family:

- `llm/flatquant_lora_llm_gpu.sh`
  - Text-only LLMs.
  - Examples: Llama-2, Llama-3.1, Qwen2.5-Instruct, Qwen3 dense text models.
  - Default SFT format: Alpaca.

- `vlm/flatquant_lora_vlm_gpu.sh`
  - Standard VLMs.
  - Examples: MiniCPM-V, Qwen2.5-VL, Qwen3-VL.
  - Default SFT format: LLaVA.
  - Default LoRA targets: `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`.

- `qwen3_5/flatquant_lora_qwen3_5_gpu.sh`
  - Dense Qwen3.5/Qwen3.6 VLMs with hybrid full/linear attention.
  - Examples: Qwen3.5-4B, Qwen3.6-27B.
  - Not for Qwen3.6-35B-A3B MoE.
  - Default SFT format: LLaVA.
  - Default LoRA targets additionally include `in_proj_qkv in_proj_z in_proj_b in_proj_a out_proj`.
