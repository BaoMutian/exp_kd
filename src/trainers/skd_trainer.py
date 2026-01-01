"""
SKD (Supervised Knowledge Distillation) Trainer

This trainer implements token-level knowledge distillation where:
- Teacher model: receives Q+E (with experience) input
- Student model: receives Q (without experience) input
- Both models produce logits for the same target sequence A
- Loss: KL divergence between teacher and student distributions

The key insight is that teacher and student have different prompts but
train on the same target response, allowing knowledge transfer from
experience-augmented inputs to experience-free inputs.

IMPORTANT: KL divergence is computed ONLY on the response (assistant) tokens,
not on the prompt tokens, since the prompts are different between teacher and student.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset


class SKDDataCollator:
    """
    Data collator for SKD that handles paired teacher/student inputs.

    Each sample contains:
    - teacher_messages: messages with experience (Q+E)
    - student_messages: messages without experience (Q) but same target (A)

    IMPORTANT: We track response start positions for both teacher and student
    so that KL divergence is computed only on the response tokens.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 4096,
        enable_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.enable_thinking = enable_thinking

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _tokenize_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Tokenize a list of messages."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
        )

        return encoded

    def _find_response_start(
        self,
        input_ids: List[int],
    ) -> int:
        """
        Find the start position of the assistant's response.

        For Qwen3, we look for the pattern:
        <|im_start|>assistant\n
        """
        # Tokenize the assistant header
        assistant_header = "<|im_start|>assistant\n"
        header_ids = self.tokenizer.encode(
            assistant_header, add_special_tokens=False)

        # Find the header in input_ids
        header_len = len(header_ids)
        for i in range(len(input_ids) - header_len + 1):
            if input_ids[i:i + header_len] == header_ids:
                return i + header_len  # Return position after the header

        # Fallback: return 0 (train on full sequence)
        return 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.

        Returns:
            Dict with keys:
            - teacher_input_ids, teacher_attention_mask, teacher_response_start
            - student_input_ids, student_attention_mask, student_response_start
            - labels: target token ids (with -100 for non-response tokens)
        """
        teacher_encodings = []
        student_encodings = []
        teacher_response_starts = []
        student_response_starts = []
        labels_list = []

        for feature in features:
            # Tokenize teacher and student messages
            teacher_enc = self._tokenize_messages(feature["teacher_messages"])
            student_enc = self._tokenize_messages(feature["student_messages"])

            teacher_encodings.append(teacher_enc)
            student_encodings.append(student_enc)

            # Find response start positions
            teacher_resp_start = self._find_response_start(
                teacher_enc["input_ids"])
            student_resp_start = self._find_response_start(
                student_enc["input_ids"])

            teacher_response_starts.append(teacher_resp_start)
            student_response_starts.append(student_resp_start)

            # Create labels (mask non-response tokens with -100)
            student_input_ids = student_enc["input_ids"]
            labels = [-100] * student_resp_start + \
                student_input_ids[student_resp_start:]
            labels_list.append(labels)

        # Pad to same length within batch (separately for teacher and student)
        max_teacher_len = max(len(e["input_ids"]) for e in teacher_encodings)
        max_student_len = max(len(e["input_ids"]) for e in student_encodings)

        # Pad teacher inputs
        teacher_input_ids = []
        teacher_attention_mask = []
        for enc in teacher_encodings:
            padding_len = max_teacher_len - len(enc["input_ids"])
            teacher_input_ids.append(
                enc["input_ids"] + [self.tokenizer.pad_token_id] * padding_len)
            teacher_attention_mask.append(
                enc["attention_mask"] + [0] * padding_len)

        # Pad student inputs
        student_input_ids = []
        student_attention_mask = []
        for enc in student_encodings:
            padding_len = max_student_len - len(enc["input_ids"])
            student_input_ids.append(
                enc["input_ids"] + [self.tokenizer.pad_token_id] * padding_len)
            student_attention_mask.append(
                enc["attention_mask"] + [0] * padding_len)

        # Pad labels
        padded_labels = []
        for lbl in labels_list:
            padding_len = max_student_len - len(lbl)
            padded_labels.append(lbl + [-100] * padding_len)

        return {
            "teacher_input_ids": torch.tensor(teacher_input_ids, dtype=torch.long),
            "teacher_attention_mask": torch.tensor(teacher_attention_mask, dtype=torch.long),
            "teacher_response_start": torch.tensor(teacher_response_starts, dtype=torch.long),
            "student_input_ids": torch.tensor(student_input_ids, dtype=torch.long),
            "student_attention_mask": torch.tensor(student_attention_mask, dtype=torch.long),
            "student_response_start": torch.tensor(student_response_starts, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


class SKDTrainer(Trainer):
    """
    Trainer for Supervised Knowledge Distillation (SKD).

    Implements token-level KL divergence loss between teacher and student models
    where teacher and student receive different inputs (with/without experience)
    but are trained on the same target sequence.

    IMPORTANT: KL divergence is computed only on RESPONSE tokens to avoid
    the prompt length mismatch issue.

    Loss = alpha * KL_loss + (1 - alpha) * CE_loss
    """

    def __init__(
        self,
        model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[SKDDataCollator] = None,
        temperature: float = 2.0,
        alpha: float = 0.5,
        kl_direction: str = "forward",
        **kwargs,
    ):
        """
        Initialize SKD Trainer.

        Args:
            model: Student model to train
            teacher_model: Teacher model (frozen)
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer
            data_collator: Data collator for SKD
            temperature: Temperature for softening distributions (higher = softer)
            alpha: Weight for KL loss (1-alpha for CE loss), default 0.5
            kl_direction: "forward" for KL(T||S), "reverse" for KL(S||T)
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            **kwargs,
        )

        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.kl_direction = kl_direction

        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute SKD loss.

        Loss = alpha * KL_loss + (1 - alpha) * CE_loss

        KL divergence is computed only on response tokens by extracting
        the response portion from both teacher and student logits.
        """
        # Get inputs
        teacher_input_ids = inputs["teacher_input_ids"]
        teacher_attention_mask = inputs["teacher_attention_mask"]
        teacher_response_start = inputs["teacher_response_start"]
        student_input_ids = inputs["student_input_ids"]
        student_attention_mask = inputs["student_attention_mask"]
        student_response_start = inputs["student_response_start"]
        labels = inputs["labels"]

        batch_size = student_input_ids.size(0)

        # Get student logits
        student_outputs = model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
        )
        student_logits = student_outputs.logits

        # Get teacher logits (no gradient)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
            )
            teacher_logits = teacher_outputs.logits

        # Compute KL loss on response tokens only
        total_kl_loss = torch.tensor(0.0, device=student_logits.device)
        total_response_tokens = 0

        for i in range(batch_size):
            # Get response portions
            t_start = teacher_response_start[i].item()
            s_start = student_response_start[i].item()

            # Get teacher response logits
            t_resp_logits = teacher_logits[i, t_start:, :]
            # Get student response logits
            s_resp_logits = student_logits[i, s_start:, :]

            # Align lengths (they should be the same since responses are identical)
            min_resp_len = min(t_resp_logits.size(0), s_resp_logits.size(0))
            if min_resp_len == 0:
                continue

            t_resp_logits = t_resp_logits[:min_resp_len, :]
            s_resp_logits = s_resp_logits[:min_resp_len, :]

            # Create mask for valid tokens (not padding)
            resp_labels = labels[i, s_start:s_start + min_resp_len]
            valid_mask = (resp_labels != -100).float()

            if valid_mask.sum() == 0:
                continue

            # Compute soft probabilities with temperature
            s_log_probs = F.log_softmax(
                s_resp_logits / self.temperature, dim=-1)
            t_probs = F.softmax(t_resp_logits / self.temperature, dim=-1)

            # Compute KL divergence
            if self.kl_direction == "forward":
                # Forward KL: KL(p_teacher || p_student)
                kl = F.kl_div(s_log_probs, t_probs,
                              reduction="none").sum(dim=-1)
            else:
                # Reverse KL: KL(p_student || p_teacher)
                s_probs = F.softmax(s_resp_logits / self.temperature, dim=-1)
                t_log_probs = F.log_softmax(
                    t_resp_logits / self.temperature, dim=-1)
                kl = F.kl_div(t_log_probs, s_probs,
                              reduction="none").sum(dim=-1)

            # Apply mask and accumulate
            masked_kl = (kl * valid_mask).sum()
            total_kl_loss = total_kl_loss + masked_kl
            total_response_tokens += valid_mask.sum().item()

        # Average KL loss and scale by temperature^2
        if total_response_tokens > 0:
            kl_loss = (total_kl_loss / total_response_tokens) * \
                (self.temperature ** 2)
        else:
            kl_loss = torch.tensor(0.0, device=student_logits.device)

        # Compute CE loss on student predictions (standard next-token prediction)
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        # Combined loss
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

        # Log individual losses
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "kl_loss": kl_loss.item() if torch.is_tensor(kl_loss) else kl_loss,
                "ce_loss": ce_loss.item(),
                "total_loss": total_loss.item(),
            })

        if return_outputs:
            return total_loss, student_outputs
        return total_loss

    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """Prepare inputs by moving to correct device."""
        inputs = super()._prepare_inputs(inputs)

        # Also move teacher model to same device if needed
        if hasattr(self, 'teacher_model'):
            device = next(self.model.parameters()).device
            if next(self.teacher_model.parameters()).device != device:
                self.teacher_model = self.teacher_model.to(device)

        return inputs
