# EarlyTIR

Research code for **Unveiling Entropy-Performance Decoupling in Agentic RL for Tool-Integrated Reasoning**.

EarlyTIR is an early-intervention mechanism for tool-integrated reasoning (TIR) during agentic reinforcement-learning rollouts. It monitors invalid tool interactions and terminates a trajectory after the configured tolerance is reached, assigning a negative truncation reward. This prevents non-informative error-retry loops while preserving a small amount of useful error-recovery exploration.

> **Paper:** *Unveiling Entropy-Performance Decoupling in Agentic RL for Tool-Integrated Reasoning*  
> **Authors:** Yirong Zeng, Shen You, Yufei Liu, Hao Cong, Yutai Hou, Xiao Ding, Yuxian Wang, Wu Ning, Wang Xu, Bibo Cai, Dandan Tu, Wenyu Zang, and Ting Liu.

## Highlights

The paper studies **entropy-performance decoupling**: late in agentic RL training, policy entropy can keep increasing even when reasoning accuracy has plateaued. EarlyTIR intervenes during **training rollouts**, rather than adding a test-time search procedure.

Selected mathematical reasoning results reported in the paper (accuracy, %):

| Model | AIME24 | AIME25 | HMMT25 | AMO-Bench | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-32B | 70.37 | 54.81 | 29.17 | 8.00 | 40.59 |
| ReTool (Qwen3-32B) | 70.00 | 61.11 | 38.89 | 6.67 | 44.17 |
| **EarlyTIR-32B** | **82.50** | **69.58** | **52.00** | **12.80** | **54.22** |

EarlyTIR results are means over three independent runs. The 54.22% macro-average is **13.63 percentage points** above the base model and **10.05 points** above ReTool.

## What is included

```text
.
├── README.md
└── code/
    ├── eval/
    │   ├── human_eval_tool.py          # HumanEval-style execution evaluator
    │   ├── livecode_eval_tool.py       # LiveCode-style execution evaluator
    │   ├── simple_sandbox_server.py    # Minimal local Python sandbox HTTP service
    │   └── vllm_infer_with_tool.py     # vLLM multi-turn tool-use inference
    └── train/
        └── verl/                       # verl training snapshot used by the experiments
            ├── recipe/early_tir/        # EarlyTIR tool, dataset, reward, and tool config
            ├── recipe/entropy/           # Entropy-analysis training components
            ├── recipe/retool/            # ReTool baseline components and launch scripts
            └── ...                      # Supporting verl implementation and examples
```

The EarlyTIR-specific implementation is in `code/train/verl/recipe/early_tir/`:

- `early_tir.py`: custom sandbox tool, dataset adapter, and reward functions.
- `sandbox_fusion_tool_config.yaml`: tool schema and sandbox endpoint configuration.
- `sandbox.py`: a minimal Flask-based local code-execution endpoint for development.

## Environment

The training code is based on the included `verl` snapshot and contains GPU and Ascend/NPU backend support. The paper reports Ascend 910B hardware; choose a backend-compatible PyTorch and inference-engine stack for your deployment. Training also requires Ray, Transformers, Hugging Face Datasets, and the packages listed in `code/train/verl/requirements.txt` and `code/train/verl/setup.py`.

For a CUDA/vLLM installation, create a compatible Python environment and install the included training package (Python >= 3.10 is required by its package metadata):

```bash
git clone https://github.com/zeng-yirong/EarlyTIR.git
cd EarlyTIR/code/train/verl
python -m pip install -e ".[vllm,math]"
```

The exact vLLM/PyTorch/CUDA combination should be selected for the target hardware. The scripts in this repository are research launch scripts and contain machine-specific defaults; review every path and resource setting before starting a run.

## Data and model paths

The training launch scripts expect local paths for:

- a DAPO-Math-17k-style training set;
- AIME 2024 and AIME 2025 evaluation sets;
- a locally available Qwen or other compatible causal language model checkpoint;
- a reachable code-execution sandbox endpoint.

The paper evaluates mathematical reasoning on AIME24, AIME25, HMMT25, and AMO-Bench, and code generation on HumanEval and LiveCodeBench. Dataset/model weights, processed benchmark files, and experiment logs are not bundled.

The example scripts use environment variables such as `DATA_ROOT`, `HDFS_ROOT`, `MODEL_PATH`, or dataset-specific path variables. Replace these values with paths on your machine. Dataset and model files are intentionally not included in this repository.

## Running EarlyTIR training

From the `code/train/verl` directory, use the ReTool DAPO launcher as a starting point and switch the EarlyTIR-specific paths and rollout options:

```bash
cd code/train/verl

# Start the development execution service in another terminal.
python recipe/early_tir/sandbox.py
```

The relevant `verl.trainer.main_ppo` overrides are:

```text
data.custom_cls.path=recipe/early_tir/early_tir.py
data.custom_cls.name=CustomRLHFDataset
custom_reward_function.path=recipe/early_tir/early_tir.py
custom_reward_function.name=compute_score_truncated
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/early_tir/sandbox_fusion_tool_config.yaml
actor_rollout_ref.rollout.multi_turn.early_truncation_on_tool_error=True
actor_rollout_ref.rollout.multi_turn.early_truncation_probability=1.0
actor_rollout_ref.rollout.multi_turn.max_invalid_tool_calls=3
actor_rollout_ref.rollout.multi_turn.error_penalty_score=-0.2
```

For a complete baseline launcher, see `recipe/retool/run_qwen2_7b_dapo.sh` and `recipe/retool/run_qwen2-32b_dapo.sh`. Copy one of these scripts, update its model/data/checkpoint paths, replace `recipe/retool/retool.py` with `recipe/early_tir/early_tir.py`, replace the tool configuration path, and add the EarlyTIR overrides above. The paper's main EarlyTIR setting permits **τ = 2** invalid calls and terminates when the cumulative count **exceeds** τ, with a truncation penalty of **Rₜ = -0.2**. Set `max_invalid_tool_calls=3` to terminate at the third invalid interaction.

Paper-reported training settings include 512 prompts per rollout batch, 16 responses per prompt, a PPO mini-batch of 128 prompts, a 2,048-token prompt limit, a 16,384-token response limit, and an eight-tool-call cap. Configure the launcher's batch sizes, token limits, and conversation-turn settings for the target experiment.

### Sandbox note

For experiments, configure an isolated execution service and set `sandbox_fusion_url` in `recipe/early_tir/sandbox_fusion_tool_config.yaml` accordingly.

## Evaluation utilities

For the local HTTP service and inference client, install `flask` and `requests` in addition to the vLLM/Transformers inference dependencies. The two execution evaluators themselves use the Python standard library.

The scripts in `code/eval/` operate on JSON/JSONL request files and are intended to be used with the model/inference pipeline that produced the paper's rollouts. Check each script's `--help` output for the supported arguments:

```bash
python code/eval/vllm_infer_with_tool.py --help
python code/eval/human_eval_tool.py --help
python code/eval/livecode_eval_tool.py --help
```

`simple_sandbox_server.py` exposes:

- `GET /health`
- `POST /run_code` with a JSON body containing `code` and optional `timeout`

Example:

```bash
python code/eval/simple_sandbox_server.py
curl -X POST http://127.0.0.1:8080/run_code \
  -H 'Content-Type: application/json' \
  -d '{"code":"print(2 + 2)","timeout":5}'
```

Run execution services in an isolated, trusted environment. The inference client calls `http://127.0.0.1:8080/run_code` directly, so each inference host must have access to that endpoint.

### Inference

Create an input JSONL file with one object per line, for example:

```json
{"id":"example-1","conversations":[{"role":"user","content":"What is 12345 * 6789? Use Python if helpful."}]}
```

Use user/assistant messages in `conversations`; the inference script supplies the tool-use system prompt. On a machine with a compatible model and the local execution server running:

```bash
python code/eval/vllm_infer_with_tool.py \
  --model /path/to/model \
  --input-file /path/to/requests.jsonl \
  --output-file /path/to/predictions.jsonl \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --max-new-tokens 16384 \
  --max-tool-turns 8 \
  --temperature 1.0 \
  --top-p 0.6
```

The output preserves input metadata and stores the conversation in `history`. The evaluators expect **different fields**, so convert generated histories before scoring:

| Script | Required scoring fields |
| --- | --- |
| `human_eval_tool.py` | `test`, `entry_point`, and `responses` (list of strings) or `final_code`; optional `original_prompt` |
| `livecode_eval_tool.py` | `final_code`, `test`, and `entry_point` |

Each evaluator reads a JSONL file; its `test` field contains executable Python tests, not the unprocessed official benchmark archive. Example commands after preparing those fields:

```bash
python code/eval/human_eval_tool.py /path/to/humaneval.jsonl \
  --timeout 3 --output /path/to/humaneval-results.json
python code/eval/livecode_eval_tool.py /path/to/livecode.jsonl \
  --timeout 10 --output /path/to/livecode-results.json
```

These scripts execute generated programs. Run them only in a disposable, isolated environment without secrets or access to important files.

## Reproducibility notes

- Configure distributed accelerator resources to match the selected model and experiment.
- Set `WANDB_API_KEY` or configure another logger only when experiment tracking is desired.
- Checkpoint, dataset, and generated-output directories should remain outside the repository.
- Before public deployment, review the included research scripts and replace development sandbox settings with an isolated executor.

## Citation

```bibtex
@misc{zeng2026earlytir,
  title     = {Unveiling Entropy-Performance Decoupling in Agentic RL for Tool-Integrated Reasoning},
  author    = {Zeng, Yirong and You, Shen and Liu, Yufei and Cong, Hao and Hou, Yutai and Ding, Xiao and Wang, Yuxian and Ning, Wu and Wang, Xu and Cai, Bibo and Tu, Dandan and Zang, Wenyu and Liu, Ting},
  year      = {2026},
  url       = {https://github.com/zeng-yirong/EarlyTIR}
}
```

## License

This work builds on [verl](https://github.com/volcengine/verl) and the included ReTool recipe; their original documentation and copyright notices are preserved.

The included `verl` snapshot retains its upstream Apache-2.0 license. Please check the headers and upstream license files for the applicable terms of each component before redistribution or modification.
