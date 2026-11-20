# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import re
from typing import Any

import datasets

from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op
from verl.tools.base_tool import BaseTool
from verl_recipe.retool.my_sandbox import GLOBAL_SANDBOX_MANAGER

logger = logging.getLogger(__name__)

from pbox import CodeSandBox
import re
import asyncio
import resource
import sys
from typing import Any
import concurrent.futures


class CustomSandboxFusionTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        TIMEOUT_SECONDS = 10
        try:
            raw_code = parameters.get("code", "")
            matches = self.code_pattern.findall(raw_code)
            if matches:
                raw_code = matches[0].strip()

            loop = asyncio.get_running_loop()

            result = await loop.run_in_executor(
                None,
                lambda: GLOBAL_SANDBOX_MANAGER.run_code(instance_id, raw_code, timeout=TIMEOUT_SECONDS)
            )
            result = str(result)
            # print(f"INFO: !!result: {result[:300]}!!")
            return result[:300], None, None

        except Exception as e:
            GLOBAL_SANDBOX_MANAGER.close_instance(instance_id)
            return f"Tool Execution Error: {str(e)}", None, None

    def close(self):
        pass


answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            # dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframe = datasets.load_dataset(parquet_file)["train"]
            data_source = "/".join(parquet_file.split("/")[-2:])
            if data_source in ["AIME_2024", "aime_2025"]:
                dataframe = dataframe.map(
                    self.map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names
                )
            else:
                dataframe = dataframe.map(self.map_fn2, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: dict, *, data_source: str = None):
        if data_source == "AIME_2024":
            problem, answer = row["Problem"], row["Answer"]
        elif data_source == "aime_2025":
            problem, answer = row["problem"], row["answer"]

        prompt = problem + answer_format
        data = {
            "data_source": data_source.split("/")[1].lower(),  # aime_2024, aime_2025
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "tool_agent",
        }
        return data

    def map_fn2(self, row: dict):
        content = row["prompt"][0]["content"]
        row["prompt"][0]["content"] = content + answer_format
        row["agent_name"] = "tool_agent"
        return row

def compute_score_truncated(data_source, solution_str, ground_truth, extra_info, **kwargs):
    if extra_info.get("is_truncated_by_error", False):
        penalty_score = extra_info.get("error_penalty_score", -2.0)
        # print(f"DEBUG [Truncated] penalty: {penalty_score}")

        return {
            "score": penalty_score,
            "acc": False,
            "pred": ""
        }

    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
    # print("DEBUG solution_str :", solution_str)
    num_turns = extra_info.get("num_turns", 1)
    step_penalty = 0.05

    if result["score"] > 0:
        efficiency_score = result["score"] - (num_turns - 1) * step_penalty
        result["score"] = max(0.5, efficiency_score)
    else:
        result["score"] = result["score"]

    if result["pred"] is None:
        result["pred"] = ""

    return result


def compute_score_single_turn(data_source, solution_str, ground_truth, extra_info, **kwargs):
    # use \\boxed{...} answer
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    # num_turns = extra_info["num_turns"]
    if result["score"] < 0:
        # tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = -1.0

    if result["pred"] is None:
        result["pred"] = ""

    return result


def compute_score_source(data_source, solution_str, ground_truth, extra_info, **kwargs):
    # use \\boxed{...} answer
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    num_turns = extra_info.get("num_turns", 0)
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result


import re
import os
import tempfile
import subprocess
import ast
import uuid
import json


def compute_score_code(data_source, solution_str, ground_truth, extra_info, **kwargs):
    if isinstance(ground_truth, str):
        try:
            parsed_gt = ast.literal_eval(ground_truth)
        except Exception as e:
            return {"score": -1.0, "acc": False, "pred": ""}
    else:
        parsed_gt = ground_truth

    if "test_cases" in parsed_gt:
        inputs = parsed_gt["test_cases"].get("inputs", [])
        outputs = parsed_gt["test_cases"].get("outputs", [])
    else:
        inputs = parsed_gt.get("inputs", [])
        outputs = parsed_gt.get("outputs", [])

    if not inputs or len(inputs) != len(outputs):
        return {"score": 0.0, "acc": False, "pred": ""}

    solution_str = re.sub(r'<think>.*?</think>', '', solution_str, flags=re.DOTALL | re.IGNORECASE)

    lower_str = solution_str.lower()
    last_submit_idx = lower_str.rfind("<submit>")
    if last_submit_idx == -1:
        return {
            "score": -1.0,
            "acc": False,
            "pred": ""
        }

    code_part = solution_str[last_submit_idx + 8:]
    end_idx = code_part.lower().find("</submit>")
    if end_idx != -1:
        final_code = code_part[:end_idx].strip()
    else:
        final_code = code_part.strip()

    passed = 0
    total = len(inputs)
    is_timeout_or_error = False

    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(final_code)
        temp_path = f.name

    try:
        for i in range(total):
            inp = inputs[i]
            expected_out = str(outputs[i]).strip()

            try:
                res = subprocess.run(
                    ["python", temp_path],
                    input=inp,
                    text=True,
                    capture_output=True,
                    timeout=1.0
                )
                actual_out = res.stdout.strip()

                if actual_out == expected_out:
                    passed += 1
                else:
                    break

            except subprocess.TimeoutExpired:
                is_timeout_or_error = True
                break
            except Exception:
                is_timeout_or_error = True
                break
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if passed == total:
        base_score = 1.0
    elif passed > 0:
        base_score = -0.5 + (passed / total) * 0.5
    else:
        base_score = -0.6 if not is_timeout_or_error else -0.8

    num_turns = extra_info.get("num_turns", 1)

    if base_score < 0:
        tool_call_reward = max(0, (num_turns - 2) / 2 * 0.1)
        final_score = min(-0.1, base_score + tool_call_reward)
    else:
        final_score = base_score

    return {
        "score": final_score,
        "acc": True if passed > 0 else False,
        "pred": final_code
    }