# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4
import random
import re
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import add_generation_prompt_for_gpt_oss, format_gpt_oss_tool_response_manually
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
            self,
            messages: list[dict[str, Any]],
            image_data: Any,
            metrics: dict[str, Any],
            request_id: str,
            tools_kwargs: dict[str, Any],
            interaction: Optional[BaseInteraction] = None,
            interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Extra fields for dynamic addition
        self.extra_fields: dict[str, Any] = {}
        self.invalid_tool_call_count = 0
        self.last_response_ids: list[int] = []
        self.is_validate = False


@register("code_agent")
class CodeAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.early_truncation_on_tool_error = config.actor_rollout_ref.rollout.multi_turn.get(
            "early_truncation_on_tool_error", False)
        cls.error_penalty_score = config.actor_rollout_ref.rollout.multi_turn.get("error_penalty_score", -10.0)
        cls.early_truncation_probability = config.actor_rollout_ref.rollout.multi_turn.get(
            "early_truncation_probability", 1.0)
        cls.max_invalid_tool_calls = config.actor_rollout_ref.rollout.multi_turn.get("max_invalid_tool_calls", 1)
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = initialize_system_prompt(cls.tokenizer, **cls.apply_chat_template_kwargs)

        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        is_validate = False
        if "meta_info" in kwargs:
            is_validate = kwargs["meta_info"].get("validate", False)
        elif "validate" in kwargs:
            is_validate = kwargs["validate"]

        if not is_validate and "extra_info" in kwargs:
            is_validate = kwargs["extra_info"].get("validate", False)

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        agent_data.extra_fields["is_truncated_by_error"] = False
        agent_data.extra_fields["error_penalty_score"] = getattr(self, "error_penalty_score", 0.0)
        agent_data.is_validate = is_validate

        sp_copy = copy.deepcopy(sampling_params)
        custom_stops = ["</execute>", "</submit>"]
        if "stop" not in sp_copy or sp_copy["stop"] is None:
            sp_copy["stop"] = custom_stops
        elif isinstance(sp_copy["stop"], list):
            for word in custom_stops:
                if word not in sp_copy["stop"]:
                    sp_copy["stop"].append(word)
        elif isinstance(sp_copy["stop"], str):
            sp_copy["stop"] = [sp_copy["stop"]] + custom_stops

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        output.extra_fields.update(agent_data.extra_fields)
        try:
            import os
            import json

            log_dir = "/path"
            os.makedirs(log_dir, exist_ok=True)

            if len(os.listdir(log_dir)) < 500:
                log_file_path = os.path.join(log_dir, f"trace_{agent_data.request_id}.json")

                trace_info = {
                    "request_id": agent_data.request_id,
                    "is_validate": agent_data.is_validate,
                    "total_turns": agent_data.assistant_turns,
                    "invalid_tool_calls": agent_data.invalid_tool_call_count,
                    "is_truncated_by_error": agent_data.extra_fields.get("is_truncated_by_error", False),
                    "messages": agent_data.messages
                }

                with open(log_file_path, "w", encoding="utf-8") as f:
                    json.dump(trace_info, f, ensure_ascii=False, indent=4)

        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(
            self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Decode immediately to check for tags
        assistant_message = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        )

        # 因为 sampling_params.stop 命中时，生成的文本可能不包含这个词，主动补回
        if "<execute>" in assistant_message and "</execute>" not in assistant_message:
            assistant_message += "\n</execute>"
        if "<submit>" in assistant_message and "</submit>" not in assistant_message:
            assistant_message += "\n</submit>"

        add_messages.append({"role": "assistant", "content": assistant_message})
        agent_data.messages.extend(add_messages)

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        # if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
        #     return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        agent_data.tool_calls = []

        if "<submit>" in assistant_message:
            logger.info("Detected <submit> tag. Terminating sequence.")
            return AgentState.TERMINATED

        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED

        execute_match = re.search(r'<execute>\s*(.*?)(?:</execute>|$)', assistant_message, re.DOTALL | re.IGNORECASE)
        if execute_match:
            code_to_run = execute_match.group(1).strip()

            prepend_imports = ""
            first_user_msg = next((m["content"] for m in agent_data.messages if m["role"] == "user"), "")
            if isinstance(first_user_msg, str):
                import_pattern = r'^(?:from\s+[\w\.]+\s+)?import\s+.*$'
                extracted_imports = re.findall(import_pattern, first_user_msg, re.MULTILINE)
                if extracted_imports:
                    prepend_imports = "\n".join(extracted_imports) + "\n"

            full_code = prepend_imports + code_to_run if prepend_imports else code_to_run

            agent_data.tool_calls.append(
                FunctionCall(
                    name="code_interpreter",
                    arguments=json.dumps({"code": full_code})
                )
            )
            return AgentState.PROCESSING_TOOLS

        logger.info("No <execute> or <submit> detected. Emitting format error tool call.")
        agent_data.tool_calls.append(
            FunctionCall(
                name="format_error",
                arguments=json.dumps({
                                         "error": "Format parsing failed: no valid instruction action was detected. Please use <execute> to run the code block strictly, or use <submit> to submit the final code."})
            )
        )
        return AgentState.PROCESSING_TOOLS

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        if getattr(self, "early_truncation_on_tool_error", False) and not agent_data.is_validate:
            has_error = False
            is_repeated_response = False

            # if agent_data.response_ids and agent_data.response_ids == agent_data.last_response_ids:
            #     has_error = True
            #     is_repeated_response = True
            #     logger.info("Repeated identical model response detected. Treating as invalid tool call.")

            # agent_data.last_response_ids = agent_data.response_ids[:]

            invalid_keywords = ["no stdout here", "Traceback", "Error", "Exception"]

            if not has_error:
                for tool_resp, _, res in responses:
                    if res is not None and res.get("is_error", False):
                        has_error = True
                        break

                    if tool_resp and tool_resp.text:
                        if any(keyword in tool_resp.text for keyword in invalid_keywords):
                            has_error = True
                            break

            if has_error:
                agent_data.invalid_tool_call_count += 1
                max_invalid = getattr(self, "max_invalid_tool_calls", 1)

                if agent_data.invalid_tool_call_count >= max_invalid:
                    truncation_prob = getattr(self, "early_truncation_probability", 1.0)
                    if random.random() < truncation_prob:
                        reason = "Repeated model response" if is_repeated_response else "Invalid tool keyword/execution"
                        logger.info(
                            f"{reason} detected {agent_data.invalid_tool_call_count} times. Reached threshold {max_invalid}. Terminating early (prob={truncation_prob}).")
                        agent_data.extra_fields["is_truncated_by_error"] = True
                        return AgentState.TERMINATED
                    else:
                        logger.info(
                            f"Invalid/repeated response detected {agent_data.invalid_tool_call_count} times, but skipped early truncation due to prob={truncation_prob}.")
                else:
                    reason = "Repeated model response" if is_repeated_response else "Invalid tool keyword/execution"
                    logger.info(
                        f"{reason} detected. Current count: {agent_data.invalid_tool_call_count}, threshold: {max_invalid}. Continuing...")
        is_warning_turn = (self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns - 1)

        for i, (tool_response, tool_reward, _) in enumerate(responses):
            tool_name = tool_call_names[i]

            if tool_name == "format_error":
                feedback = tool_response.text
            else:
                exec_result = tool_response.text
                if is_warning_turn:
                    feedback = (
                        f"[Tool Execution Return Result]:\n```text\n{exec_result}\n```\n"
                        "⚠️ Warning: You have reached the maximum limit for tool invocations. In your next response, please directly analyze the aforementioned results and be sure to submit the final complete code using the <submit> tag; no further <execute> tool calls are permitted."
                    )
                else:
                    feedback = (
                        f"[Tool Execution Return Result]:\n```text\n{exec_result}\n```\n"
                        "Please continue to modify the code based on the execution results described above. If you confirm that the code is correct, submit the final complete code using the <submit> tag."
                    )

            message = {"role": "user", "content": feedback}
            add_messages.append(message)

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        # Update prompt with tool/user responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            current_images = new_images_this_turn if new_images_this_turn else None
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True,
                                                           **self.apply_chat_template_kwargs),
            )
            # Clip the initial system prompt part if your tokenizer repeats it
            # Depends on your chat template, normally we only encode the new messages
            # If using `add_messages` directly is returning a full string with system,
            # ensure your tokenizer handles partial appends correctly.
            # In standard verl, this slicing handles it:
            if hasattr(self, 'system_prompt') and len(response_ids) > len(self.system_prompt):
                # Only slice if it looks like the chat template appended everything from scratch
                # Wait, standard verl applies it to ONLY add_messages here. So slicing might not be needed.
                # Just keeping the original verl logic below for safety:
                pass

        # Since standard verl logic does this for purely appendings:
        if not self.processor:
            # Re-encode only the new user message to append to prompt_ids safely
            raw_text = self.tokenizer.apply_chat_template(add_messages, tokenize=False, add_generation_prompt=True,
                                                          **self.apply_chat_template_kwargs)
            response_ids = self.tokenizer.encode(raw_text, add_special_tokens=False)

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt):]

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
            self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool_name = tool_call.name
        tool_args = json.loads(tool_call.arguments)

        if tool_name == "format_error":
            return ToolResponse(text=tool_args.get("error", "")), getattr(self, "error_penalty_score", 0.0), {
                "is_error": True}
        tool, instance_id = None, None
        try:
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                getattr(self, "error_penalty_score", 0.0),
                {"is_error": True},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        # tool_response_text = tool_execution_response.text
        if isinstance(tool_execution_response, str):
            tool_response_text = tool_execution_response
        else:
            tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length:]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map
