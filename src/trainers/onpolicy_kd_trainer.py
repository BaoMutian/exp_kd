"""
On-Policy Knowledge Distillation Trainer

This trainer implements on-policy (online) knowledge distillation where:
- Student model: receives Q (without experience) input, generates responses on-the-fly
- Teacher model: receives Q+E (with experience) input, scores student-generated responses
- Loss: KL/JSD divergence between teacher and student distributions on student-generated sequences

The key insight is that the student learns from its own generated outputs rather than
fixed ground-truth sequences, addressing the train-inference distribution mismatch.

Based on GKD paper: "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
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
from transformers.modeling_utils import unwrap_model
from torch.utils.data import Dataset


class OnPolicyKDDataCollator:
    """
    Data collator for On-Policy KD that handles paired teacher/student prompts.

    Each sample contains:
    - teacher_prompt_messages: prompt with experience (Q+E), no assistant response
    - student_prompt_messages: prompt without experience (Q), no assistant response

    The assistant responses will be generated on-the-fly by the student model.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int = 2048,
        enable_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.enable_thinking = enable_thinking

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _tokenize_prompt(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Tokenize a list of prompt messages (without assistant response)."""
        # Apply chat template with generation prompt (ready for generation)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_prompt_length,
            padding=False,
            return_tensors=None,
        )

        return encoded

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.

        Returns:
            Dict with keys:
            - teacher_input_ids, teacher_attention_mask (prompts with experience)
            - student_input_ids, student_attention_mask (prompts without experience)
        """
        teacher_encodings = []
        student_encodings = []

        for feature in features:
            # Tokenize teacher and student prompts
            teacher_enc = self._tokenize_prompt(
                feature["teacher_prompt_messages"])
            student_enc = self._tokenize_prompt(
                feature["student_prompt_messages"])

            teacher_encodings.append(teacher_enc)
            student_encodings.append(student_enc)

        # Pad to same length within batch (separately for teacher and student)
        max_teacher_len = max(len(e["input_ids"]) for e in teacher_encodings)
        max_student_len = max(len(e["input_ids"]) for e in student_encodings)

        # Pad teacher inputs (left padding for generation)
        teacher_input_ids = []
        teacher_attention_mask = []
        for enc in teacher_encodings:
            padding_len = max_teacher_len - len(enc["input_ids"])
            # Left padding for decoder-only models
            teacher_input_ids.append(
                [self.tokenizer.pad_token_id] * padding_len + enc["input_ids"]
            )
            teacher_attention_mask.append(
                [0] * padding_len + enc["attention_mask"])

        # Pad student inputs (left padding for generation)
        student_input_ids = []
        student_attention_mask = []
        for enc in student_encodings:
            padding_len = max_student_len - len(enc["input_ids"])
            # Left padding for decoder-only models
            student_input_ids.append(
                [self.tokenizer.pad_token_id] * padding_len + enc["input_ids"]
            )
            student_attention_mask.append(
                [0] * padding_len + enc["attention_mask"])

        return {
            "teacher_input_ids": torch.tensor(teacher_input_ids, dtype=torch.long),
            "teacher_attention_mask": torch.tensor(teacher_attention_mask, dtype=torch.long),
            "student_input_ids": torch.tensor(student_input_ids, dtype=torch.long),
            "student_attention_mask": torch.tensor(student_attention_mask, dtype=torch.long),
        }


class OnPolicyKDTrainer(Trainer):
    """
    Trainer for On-Policy Knowledge Distillation.

    Implements on-policy (online) distillation where:
    1. Student generates responses from its current policy
    2. Teacher provides soft labels (probability distribution) on student-generated sequences
    3. Student learns to match teacher's distribution on its own outputs

    This addresses the train-inference distribution mismatch problem in traditional KD.

    Loss = JSD(p_teacher, p_student) on student-generated sequences
         = beta * KL(p_teacher || p_student) + (1 - beta) * KL(p_student || p_teacher)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[OnPolicyKDDataCollator] = None,
        temperature: float = 1.0,
        beta: float = 0.5,
        max_new_tokens: int = 256,
        generation_temperature: float = 0.9,
        num_samples: int = 1,
        **kwargs,
    ):
        """
        Initialize On-Policy KD Trainer.

        Args:
            model: Student model to train
            teacher_model: Teacher model (frozen)
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer
            data_collator: Data collator for On-Policy KD
            temperature: Temperature for softening distributions during KL computation
            beta: JSD interpolation coefficient (0=forward KL, 1=reverse KL, 0.5=symmetric JSD)
            max_new_tokens: Maximum tokens to generate per response
            generation_temperature: Temperature for sampling during generation
            num_samples: Number of response samples to generate per input
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
        self.beta = beta
        self.max_new_tokens = max_new_tokens
        self.generation_temperature = generation_temperature
        self.num_samples = num_samples
        # Store tokenizer with a different name to avoid deprecation warning
        self._tokenizer = tokenizer

        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def _get_generation_model(self) -> PreTrainedModel:
        """
        Get the unwrapped model for generation.
        
        In distributed training (DeepSpeed/FSDP), the model is wrapped.
        We need to unwrap it for generation to work correctly.
        """
        return unwrap_model(self.model)

    def _generate_student_responses(
        self,
        student_input_ids: torch.Tensor,
        student_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate responses from the student model.

        Args:
            student_input_ids: Input token IDs for student prompts
            student_attention_mask: Attention mask for student prompts

        Returns:
            Tuple of (generated_ids, generated_attention_mask) including prompt and response
        """
        # Get unwrapped model for generation
        generation_model = self._get_generation_model()
        
        # Check if we're in distributed training
        is_distributed = self.args.parallel_mode.value != "not_parallel"
        
        # Set model to eval mode for generation (temporarily)
        was_training = generation_model.training
        generation_model.eval()

        with torch.no_grad():
            # Check for NaN in model parameters before generation
            # This can happen if previous gradient update produced NaN
            for name, param in generation_model.named_parameters():
                if torch.isnan(param).any():
                    raise RuntimeError(f"NaN detected in model parameter: {name}")
            
            # Generate responses
            # Note: use_cache should be False when gradient_checkpointing is enabled
            # synced_gpus=True for distributed training to avoid hangs
            try:
                outputs = generation_model.generate(
                    input_ids=student_input_ids,
                    attention_mask=student_attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.generation_temperature,
                    top_p=0.9,
                    num_return_sequences=self.num_samples,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    return_dict_in_generate=False,
                    synced_gpus=is_distributed,
                    use_cache=False,  # Must be False when gradient_checkpointing is enabled
                )
            except RuntimeError as e:
                # If generation fails, return dummy output
                import logging
                logging.warning(f"Generation failed with error: {e}")
                # Return original input as "generated" to avoid crash
                outputs = student_input_ids
                outputs = torch.cat([
                    outputs,
                    torch.full(
                        (outputs.size(0), self.max_new_tokens),
                        self._tokenizer.pad_token_id,
                        device=outputs.device,
                        dtype=outputs.dtype
                    )
                ], dim=1)

        # Restore training mode
        if was_training:
            generation_model.train()

        # Ensure consistent sequence length across all GPUs by padding to max_length
        # This is important for distributed training where different GPUs may
        # generate sequences of different lengths
        prompt_length = student_input_ids.size(1)
        target_length = prompt_length + self.max_new_tokens
        current_length = outputs.size(1)

        if current_length < target_length:
            # Right-pad to target length
            pad_size = target_length - current_length
            outputs = torch.cat([
                outputs,
                torch.full(
                    (outputs.size(0), pad_size),
                    self._tokenizer.pad_token_id,
                    device=outputs.device,
                    dtype=outputs.dtype
                )
            ], dim=1)

        # Create attention mask for generated sequences
        generated_attention_mask = (
            outputs != self._tokenizer.pad_token_id).long()

        return outputs, generated_attention_mask

    def _prepare_teacher_input(
        self,
        teacher_input_ids: torch.Tensor,
        teacher_attention_mask: torch.Tensor,
        generated_response_ids: torch.Tensor,
        student_prompt_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare teacher input by appending student-generated responses to teacher prompts.

        Args:
            teacher_input_ids: Teacher prompt token IDs (with experience)
            teacher_attention_mask: Teacher prompt attention mask
            generated_response_ids: Full generated sequence (prompt + response) from student
            student_prompt_length: Length of student prompt (to extract response portion)

        Returns:
            Tuple of (teacher_full_input_ids, teacher_full_attention_mask)
        """
        # If num_samples > 1, we need to expand teacher inputs
        if self.num_samples > 1:
            teacher_input_ids = teacher_input_ids.repeat_interleave(
                self.num_samples, dim=0
            )
            teacher_attention_mask = teacher_attention_mask.repeat_interleave(
                self.num_samples, dim=0
            )

        # Extract response portion from generated sequence
        response_ids = generated_response_ids[:, student_prompt_length:]
        response_attention_mask = (
            response_ids != self._tokenizer.pad_token_id).long()

        # Concatenate teacher prompt with student-generated response
        teacher_full_input_ids = torch.cat(
            [teacher_input_ids, response_ids], dim=1
        )
        teacher_full_attention_mask = torch.cat(
            [teacher_attention_mask, response_attention_mask], dim=1
        )

        return teacher_full_input_ids, teacher_full_attention_mask

    def _compute_jsd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Generalized Jensen-Shannon Divergence loss.

        JSD_beta(P, Q) = beta * KL(P || M) + (1 - beta) * KL(Q || M)
        where M = beta * P + (1 - beta) * Q

        When beta = 0.5, this is symmetric JSD.
        When beta = 0, this approximates forward KL(Q || P).
        When beta = 1, this approximates reverse KL(P || Q).

        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
            labels_mask: Mask for valid tokens [batch, seq_len]

        Returns:
            Scalar loss value
        """
        # Apply temperature scaling
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature

        # Compute log probabilities (more numerically stable)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        # Compute probabilities
        student_probs = student_log_probs.exp()
        teacher_probs = teacher_log_probs.exp()

        # Compute mixture distribution M = beta * P_teacher + (1 - beta) * P_student
        mixture_probs = self.beta * teacher_probs + (1 - self.beta) * student_probs
        # Use clamp for numerical stability before log
        mixture_log_probs = torch.log(mixture_probs.clamp(min=1e-10))

        # Compute JSD components using manual KL computation for better stability
        # KL(P || M) = sum(P * (log P - log M))
        # KL(P_teacher || M)
        kl_teacher_mixture = (teacher_probs * (teacher_log_probs - mixture_log_probs)).sum(dim=-1)

        # KL(P_student || M)
        kl_student_mixture = (student_probs * (student_log_probs - mixture_log_probs)).sum(dim=-1)

        # Clamp KL values to avoid negative values due to numerical errors
        kl_teacher_mixture = kl_teacher_mixture.clamp(min=0.0)
        kl_student_mixture = kl_student_mixture.clamp(min=0.0)

        # JSD = beta * KL(P_teacher || M) + (1 - beta) * KL(P_student || M)
        jsd = self.beta * kl_teacher_mixture + (1 - self.beta) * kl_student_mixture

        # Apply mask and compute mean loss
        masked_jsd = jsd * labels_mask
        
        # Check for NaN and replace with 0
        if torch.isnan(masked_jsd).any():
            masked_jsd = torch.where(torch.isnan(masked_jsd), torch.zeros_like(masked_jsd), masked_jsd)
        
        loss = masked_jsd.sum() / (labels_mask.sum().clamp(min=1.0))

        # Scale by temperature^2 (standard KD practice)
        loss = loss * (self.temperature ** 2)

        # Final safety check - replace NaN/Inf with zero while keeping computation graph
        if torch.isnan(loss) or torch.isinf(loss):
            # Use a zero loss that's still connected to the computation graph
            loss = student_logits.sum() * 0.0

        return loss

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute On-Policy KD loss.

        Steps:
        1. Generate responses from student model using current policy
        2. Compute teacher logits on (teacher_prompt + student_response)
        3. Compute student logits on (student_prompt + student_response)
        4. Compute JSD loss between teacher and student distributions
        """
        # Get inputs
        teacher_input_ids = inputs["teacher_input_ids"]
        teacher_attention_mask = inputs["teacher_attention_mask"]
        student_input_ids = inputs["student_input_ids"]
        student_attention_mask = inputs["student_attention_mask"]

        device = student_input_ids.device
        student_prompt_length = student_input_ids.size(1)
        teacher_prompt_length = teacher_input_ids.size(1)

        # Step 1: Generate responses from student model
        generated_ids, generated_attention_mask = self._generate_student_responses(
            student_input_ids, student_attention_mask
        )

        # Safety check: ensure generation produced tokens
        generated_length = generated_ids.size(1)
        if generated_length <= student_prompt_length:
            # No new tokens were generated, return zero loss
            # Create a dummy forward pass for gradient computation
            dummy_output = model(
                input_ids=student_input_ids[:, :1],
                attention_mask=student_attention_mask[:, :1],
            )
            loss = dummy_output.logits.sum() * 0.0
            if return_outputs:
                return loss, dummy_output
            return loss

        # Handle num_samples > 1 case
        effective_batch_size = generated_ids.size(0)

        # Step 2: Prepare inputs for computing logits
        # For teacher: teacher_prompt + student_generated_response
        teacher_full_ids, teacher_full_mask = self._prepare_teacher_input(
            teacher_input_ids,
            teacher_attention_mask,
            generated_ids,
            student_prompt_length,
        )

        # Step 3: Compute teacher logits (no gradient)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=teacher_full_ids,
                attention_mask=teacher_full_mask,
            )
            teacher_logits = teacher_outputs.logits

        # Step 4: Compute student logits (with gradient)
        # For student: student_prompt + student_generated_response (same as generated_ids)
        student_outputs = model(
            input_ids=generated_ids,
            attention_mask=generated_attention_mask,
        )
        student_logits = student_outputs.logits

        # Step 5: Align logits and compute loss on response tokens only
        # Safety check for sequence lengths
        student_seq_len = student_logits.size(1)
        teacher_seq_len = teacher_logits.size(1)

        # Ensure we have enough tokens for slicing
        if student_seq_len <= student_prompt_length or teacher_seq_len <= teacher_prompt_length:
            loss = student_logits.sum() * 0.0
            if return_outputs:
                return loss, student_outputs
            return loss

        # Extract response logits (shift by 1 for next-token prediction)
        # Student: logits at positions [prompt_len-1 : -1] predict tokens at [prompt_len:]
        student_response_logits = student_logits[:, student_prompt_length - 1:-1, :]

        # Teacher: logits at positions [teacher_prompt_len-1 : -1] predict tokens at [teacher_prompt_len:]
        teacher_response_logits = teacher_logits[:, teacher_prompt_length - 1:-1, :]

        # Ensure same length (they should be the same since response is the same)
        min_response_len = min(
            student_response_logits.size(1), teacher_response_logits.size(1)
        )

        if min_response_len == 0:
            # No response tokens to compute loss on
            loss = student_logits.sum() * 0.0
            if return_outputs:
                return loss, student_outputs
            return loss

        student_response_logits = student_response_logits[:, :min_response_len, :]
        teacher_response_logits = teacher_response_logits[:, :min_response_len, :]

        # Create mask for valid response tokens (exclude padding)
        # Safety check for indexing
        end_idx = min(student_prompt_length + min_response_len, generated_ids.size(1))
        response_ids = generated_ids[:, student_prompt_length:end_idx]

        # Pad response_ids if needed
        if response_ids.size(1) < min_response_len:
            pad_size = min_response_len - response_ids.size(1)
            response_ids = torch.cat([
                response_ids,
                torch.full((response_ids.size(0), pad_size),
                           self._tokenizer.pad_token_id,
                           device=device, dtype=response_ids.dtype)
            ], dim=1)

        labels_mask = (response_ids != self._tokenizer.pad_token_id).float()

        # Step 6: Compute JSD loss
        loss = self._compute_jsd_loss(
            student_response_logits,
            teacher_response_logits,
            labels_mask,
        )

        # Log metrics
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "jsd_loss": loss.item(),
                "avg_response_length": labels_mask.sum().item() / effective_batch_size,
            })

        if return_outputs:
            return loss, student_outputs
        return loss

    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """Prepare inputs by moving to correct device."""
        inputs = super()._prepare_inputs(inputs)

        # Also move teacher model to same device if needed
        if hasattr(self, "teacher_model"):
            device = next(self.model.parameters()).device
            if next(self.teacher_model.parameters()).device != device:
                self.teacher_model = self.teacher_model.to(device)

        return inputs
