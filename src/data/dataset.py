"""Dataset loading and preprocessing for knowledge distillation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_kd_dataset(
    teacher_data_path: Union[str, Path],
    student_data_path: Union[str, Path],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load paired teacher and student datasets.

    Teacher dataset: contains Q+E (with experience) input and high-quality A output
    Student dataset: contains Q (without experience) input (response is ignored)

    Args:
        teacher_data_path: Path to teacher dataset (with experience)
        student_data_path: Path to student dataset (without experience)

    Returns:
        Tuple of (teacher_data, student_data)
    """
    teacher_data = load_jsonl(teacher_data_path)
    student_data = load_jsonl(student_data_path)

    assert len(teacher_data) == len(student_data), \
        f"Dataset length mismatch: teacher={len(teacher_data)}, student={len(student_data)}"

    return teacher_data, student_data


class KDDataset:
    """
    Knowledge Distillation Dataset wrapper.

    Combines teacher and student data for different KD methods:
    - SeqKD: student input + teacher output
    - SKD: paired (teacher input, student input, target output)
    """

    def __init__(
        self,
        teacher_data: List[Dict],
        student_data: List[Dict],
    ):
        self.teacher_data = teacher_data
        self.student_data = student_data

    def __len__(self):
        return len(self.teacher_data)

    def get_teacher_messages(self, idx: int) -> List[Dict[str, str]]:
        """Get teacher messages (with experience) for an index."""
        return self.teacher_data[idx]["messages"]

    def get_student_messages(self, idx: int) -> List[Dict[str, str]]:
        """Get student messages (without experience, but with teacher's response)."""
        student_msgs = self.student_data[idx]["messages"]
        teacher_msgs = self.teacher_data[idx]["messages"]

        # Use student's system and user, but teacher's assistant response
        result = []
        for i, msg in enumerate(student_msgs):
            if msg["role"] == "assistant":
                # Use teacher's response
                result.append(teacher_msgs[i])
            else:
                # Use student's input (no experience)
                result.append(msg)

        return result

    def get_student_input_only(self, idx: int) -> List[Dict[str, str]]:
        """Get only the student input messages (system + user, no assistant)."""
        student_msgs = self.student_data[idx]["messages"]
        return [msg for msg in student_msgs if msg["role"] != "assistant"]


def create_seqkd_dataset(
    teacher_data_path: Union[str, Path],
    student_data_path: Union[str, Path],
) -> Dataset:
    """
    Create dataset for SeqKD (Sequence-Level KD).

    SeqKD trains the student model with:
    - Input (prompt): student's system prompt + user message (Q, no experience)
    - Target (completion): teacher's assistant response (high-quality A)

    Uses prompt-completion format to ensure loss is only computed on completion.

    Args:
        teacher_data_path: Path to teacher dataset
        student_data_path: Path to student dataset

    Returns:
        HuggingFace Dataset with 'prompt' and 'completion' columns (conversational format)
    """
    teacher_data, student_data = load_kd_dataset(
        teacher_data_path, student_data_path)
    kd_dataset = KDDataset(teacher_data, student_data)

    # Create dataset with prompt-completion format (conversational style)
    data = []
    for idx in range(len(kd_dataset)):
        student_msgs = student_data[idx]["messages"]
        teacher_msgs = teacher_data[idx]["messages"]

        # Prompt: all non-assistant messages from student (no experience)
        prompt_messages = [
            msg for msg in student_msgs if msg["role"] != "assistant"]

        # Completion: assistant message from teacher (high-quality response)
        completion_messages = [
            msg for msg in teacher_msgs if msg["role"] == "assistant"]

        data.append({
            "prompt": prompt_messages,
            "completion": completion_messages,
        })

    return Dataset.from_list(data)


def create_skd_dataset(
    teacher_data_path: Union[str, Path],
    student_data_path: Union[str, Path],
) -> Dataset:
    """
    Create dataset for SKD (Supervised Knowledge Distillation).

    SKD requires both teacher and student inputs to compute KL divergence:
    - Teacher input: Q+E (with experience) for computing teacher logits
    - Student input: Q (no experience) for computing student logits
    - Target sequence: teacher's assistant response (A)

    Args:
        teacher_data_path: Path to teacher dataset
        student_data_path: Path to student dataset

    Returns:
        HuggingFace Dataset with 'teacher_messages', 'student_messages' columns
    """
    teacher_data, student_data = load_kd_dataset(
        teacher_data_path, student_data_path)
    kd_dataset = KDDataset(teacher_data, student_data)

    data = []
    for idx in range(len(kd_dataset)):
        teacher_messages = kd_dataset.get_teacher_messages(idx)
        student_messages = kd_dataset.get_student_messages(idx)

        data.append({
            "teacher_messages": teacher_messages,
            "student_messages": student_messages,
        })

    return Dataset.from_list(data)


def prepare_chat_format(
    messages: List[Dict[str, str]],
    tokenizer,
    enable_thinking: bool = False,
    add_generation_prompt: bool = False,
) -> str:
    """
    Apply chat template to messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        tokenizer: Tokenizer with apply_chat_template method
        enable_thinking: Whether to enable Qwen3 thinking mode
        add_generation_prompt: Whether to add generation prompt at the end

    Returns:
        Formatted string
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def get_response_template(tokenizer) -> str:
    """
    Get the response template for masking in SFT.

    This is the string that appears before the assistant's response
    in the formatted chat template.

    For Qwen3, this is typically "<|im_start|>assistant\n"
    """
    # Create a dummy message to find the template
    messages = [
        {"role": "user", "content": "test"},
    ]

    # Apply template with generation prompt
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # The response template is what comes after the user message
    # For Qwen3: "<|im_start|>assistant\n"
    return "<|im_start|>assistant\n"
