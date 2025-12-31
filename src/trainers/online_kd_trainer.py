"""
Online Knowledge Distillation Trainer for Experience Internalization

This trainer implements on-policy distillation where:
- Student model: receives Q (without experience), samples N responses
- Teacher model: receives Q+E (with experience), evaluates student's responses
- Student learns to match teacher's distribution on its own samples

This is the standard on-policy KD approach, but with a key difference:
- Teacher uses Q+E (with experience) to evaluate student's responses
- This enables true experience internalization

Training process:
1. For each input Q, student samples N responses A₁, A₂, ..., Aₙ
2. Teacher evaluates each response using Q+E context, computing P(Aᵢ|Q+E)
3. Student learns to match teacher's evaluation: P(Aᵢ|Q) → P(Aᵢ|Q+E)
4. Update student model

Key insight: By training on student's own samples with teacher's Q+E feedback,
the student internalizes the experience E into its weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    GenerationConfig,
)
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class OnlineKDDataCollator:
    """
    Data collator for Online KD that handles paired teacher/student inputs.
    
    Each sample contains:
    - teacher_messages: full conversation with experience (Q+E + A)
    - student_messages: full conversation without experience (Q + A)
    
    For on-policy training, we only need the prompts (without assistant response).
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 4096,
        max_new_tokens: int = 256,
        enable_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def _get_prompt_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Extract only prompt messages (system + user, no assistant)."""
        return [msg for msg in messages if msg["role"] != "assistant"]
    
    def _tokenize_prompt(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Tokenize prompt messages with generation prompt."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length - self.max_new_tokens,
            padding=False,
            return_tensors=None,
        )
        
        return encoded
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
        
        Returns:
            Dict with keys:
            - teacher_input_ids, teacher_attention_mask (prompt with experience)
            - student_input_ids, student_attention_mask (prompt without experience)
        """
        teacher_encodings = []
        student_encodings = []
        
        for feature in features:
            # Get prompt-only messages
            teacher_prompt = self._get_prompt_messages(feature["teacher_messages"])
            student_prompt = self._get_prompt_messages(feature["student_messages"])
            
            teacher_enc = self._tokenize_prompt(teacher_prompt)
            student_enc = self._tokenize_prompt(student_prompt)
            
            teacher_encodings.append(teacher_enc)
            student_encodings.append(student_enc)
        
        # Pad teacher inputs (left padding for generation)
        max_teacher_len = max(len(e["input_ids"]) for e in teacher_encodings)
        teacher_input_ids = []
        teacher_attention_mask = []
        for enc in teacher_encodings:
            padding_len = max_teacher_len - len(enc["input_ids"])
            teacher_input_ids.append(
                [self.tokenizer.pad_token_id] * padding_len + enc["input_ids"])
            teacher_attention_mask.append([0] * padding_len + enc["attention_mask"])
        
        # Pad student inputs (left padding for generation)
        max_student_len = max(len(e["input_ids"]) for e in student_encodings)
        student_input_ids = []
        student_attention_mask = []
        for enc in student_encodings:
            padding_len = max_student_len - len(enc["input_ids"])
            student_input_ids.append(
                [self.tokenizer.pad_token_id] * padding_len + enc["input_ids"])
            student_attention_mask.append([0] * padding_len + enc["attention_mask"])
        
        return {
            "teacher_input_ids": torch.tensor(teacher_input_ids, dtype=torch.long),
            "teacher_attention_mask": torch.tensor(teacher_attention_mask, dtype=torch.long),
            "student_input_ids": torch.tensor(student_input_ids, dtype=torch.long),
            "student_attention_mask": torch.tensor(student_attention_mask, dtype=torch.long),
        }


class OnlineKDTrainer(Trainer):
    """
    Online Knowledge Distillation Trainer for Experience Internalization.
    
    Implements true on-policy distillation:
    1. Student samples N responses given Q (without experience)
    2. Teacher evaluates each response using Q+E (with experience)
    3. Student learns to match teacher's evaluation on its own samples
    
    Key feature: Teacher uses Q+E to evaluate, enabling experience internalization.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[OnlineKDDataCollator] = None,
        temperature: float = 1.0,
        max_new_tokens: int = 256,
        num_samples: int = 1,  # Number of samples per input
        generation_config: Optional[Dict[str, Any]] = None,
        kl_type: str = "forward",  # "forward", "reverse", or "jsd"
        beta: float = 0.5,  # For JSD interpolation
        **kwargs,
    ):
        """
        Initialize Online KD Trainer.
        
        Args:
            model: Student model to train
            teacher_model: Teacher model (frozen)
            args: Training arguments
            train_dataset: Training dataset with teacher_messages and student_messages
            tokenizer: Tokenizer
            data_collator: Data collator
            temperature: Temperature for KL computation
            max_new_tokens: Maximum tokens to generate
            num_samples: Number of responses to sample per input (N)
            generation_config: Generation configuration dict
            kl_type: Type of divergence ("forward", "reverse", "jsd")
            beta: Interpolation coefficient for JSD (0=forward KL, 1=reverse KL)
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
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self.kl_type = kl_type
        self.beta = beta
        
        # Generation config for student sampling
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=generation_config.get("temperature", 0.7) if generation_config else 0.7,
            top_p=generation_config.get("top_p", 0.8) if generation_config else 0.8,
            top_k=generation_config.get("top_k", 20) if generation_config else 20,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,  # We'll loop for multiple samples
        )
        
        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        logger.info(f"OnlineKDTrainer initialized with num_samples={num_samples}")
    
    def _sample_from_student(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample N responses from student model.
        
        Args:
            model: Student model
            input_ids: Prompt input IDs [batch_size, prompt_len]
            attention_mask: Prompt attention mask
            
        Returns:
            List of (generated_ids, attention_mask) tuples, length = num_samples
            Each generated_ids is [batch_size, prompt_len + response_len]
        """
        samples = []
        
        for _ in range(self.num_samples):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
            
            generated_ids = outputs.sequences
            generated_mask = torch.ones_like(generated_ids)
            generated_mask[generated_ids == self.tokenizer.pad_token_id] = 0
            
            samples.append((generated_ids, generated_mask))
        
        return samples
    
    def _prepare_teacher_input_with_response(
        self,
        teacher_prompt_ids: torch.Tensor,
        teacher_prompt_mask: torch.Tensor,
        student_response_ids: torch.Tensor,
        student_response_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate teacher prompt (Q+E) with student's sampled response.
        
        This allows teacher to evaluate the student's response with experience context.
        
        Args:
            teacher_prompt_ids: [batch_size, teacher_prompt_len]
            teacher_prompt_mask: [batch_size, teacher_prompt_len]
            student_response_ids: [batch_size, response_len]
            student_response_mask: [batch_size, response_len]
            
        Returns:
            (full_ids, full_mask) for teacher evaluation
        """
        # Concatenate: teacher_prompt + student_response
        full_ids = torch.cat([teacher_prompt_ids, student_response_ids], dim=1)
        full_mask = torch.cat([teacher_prompt_mask, student_response_mask], dim=1)
        
        return full_ids, full_mask
    
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute Online KD loss with on-policy sampling.
        
        Process:
        1. Student samples N responses given Q (without experience)
        2. For each sample:
           - Student computes P(response|Q)
           - Teacher computes P(response|Q+E)
           - Compute KL divergence
        3. Average loss across all samples
        """
        teacher_prompt_ids = inputs["teacher_input_ids"]
        teacher_prompt_mask = inputs["teacher_attention_mask"]
        student_prompt_ids = inputs["student_input_ids"]
        student_prompt_mask = inputs["student_attention_mask"]
        
        batch_size = student_prompt_ids.size(0)
        student_prompt_len = student_prompt_ids.size(1)
        teacher_prompt_len = teacher_prompt_ids.size(1)
        device = student_prompt_ids.device
        
        # Step 1: Student samples N responses
        student_samples = self._sample_from_student(
            model,  # Use the current student model for sampling
            student_prompt_ids,
            student_prompt_mask,
        )
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_tokens = 0
        num_valid_samples = 0
        
        # Step 2: For each sample, compute KL loss
        for sample_idx, (student_full_ids, student_full_mask) in enumerate(student_samples):
            # Extract response part from student's generation
            response_ids = student_full_ids[:, student_prompt_len:]
            response_mask = student_full_mask[:, student_prompt_len:]
            
            response_len = response_ids.size(1)
            if response_len == 0:
                continue
            
            # Prepare teacher input: teacher_prompt (Q+E) + student_response
            teacher_full_ids, teacher_full_mask = self._prepare_teacher_input_with_response(
                teacher_prompt_ids,
                teacher_prompt_mask,
                response_ids,
                response_mask,
            )
            
            # Step 3: Compute logits
            # Student: P(response | Q)
            student_outputs = model(
                input_ids=student_full_ids,
                attention_mask=student_full_mask,
            )
            student_logits = student_outputs.logits
            
            # Teacher: P(response | Q+E)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=teacher_full_ids,
                    attention_mask=teacher_full_mask,
                )
                teacher_logits = teacher_outputs.logits
            
            # Step 4: Compute KL on response tokens
            sample_loss, sample_tokens = self._compute_sample_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_prompt_len=student_prompt_len,
                teacher_prompt_len=teacher_prompt_len,
                response_mask=response_mask,
            )
            
            if sample_tokens > 0:
                total_loss = total_loss + sample_loss
                total_tokens += sample_tokens
                num_valid_samples += 1
        
        # Average across samples and tokens
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
        else:
            avg_loss = total_loss
        
        # Log losses
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "kd_loss": avg_loss.item() if torch.is_tensor(avg_loss) else avg_loss,
                "num_samples": num_valid_samples,
                "total_tokens": total_tokens,
            })
        
        if return_outputs:
            return avg_loss, None
        return avg_loss
    
    def _compute_sample_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_prompt_len: int,
        teacher_prompt_len: int,
        response_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute KL divergence loss for a single sample.
        
        Args:
            student_logits: [batch_size, student_seq_len, vocab_size]
            teacher_logits: [batch_size, teacher_seq_len, vocab_size]
            student_prompt_len: Length of student prompt
            teacher_prompt_len: Length of teacher prompt
            response_mask: [batch_size, response_len]
            
        Returns:
            (loss, num_tokens)
        """
        batch_size = student_logits.size(0)
        response_len = response_mask.size(1)
        device = student_logits.device
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_tokens = 0
        
        for i in range(batch_size):
            # Get valid response length for this sample
            valid_len = response_mask[i].sum().item()
            if valid_len == 0:
                continue
            
            # Student logits for response (shifted for next-token prediction)
            # Position student_prompt_len-1 predicts token at student_prompt_len
            s_start = student_prompt_len - 1
            s_end = student_prompt_len + int(valid_len) - 1
            s_logits = student_logits[i, s_start:s_end, :]
            
            # Teacher logits for response
            t_start = teacher_prompt_len - 1
            t_end = teacher_prompt_len + int(valid_len) - 1
            t_logits = teacher_logits[i, t_start:t_end, :]
            
            # Align lengths (should be same, but safety check)
            min_len = min(s_logits.size(0), t_logits.size(0))
            if min_len == 0:
                continue
            
            s_logits = s_logits[:min_len, :]
            t_logits = t_logits[:min_len, :]
            
            # Compute divergence
            loss = self._compute_divergence(s_logits, t_logits)
            total_loss = total_loss + loss * min_len
            total_tokens += min_len
        
        return total_loss, total_tokens
    
    def _compute_divergence(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute divergence between student and teacher distributions.
        
        Args:
            student_logits: [seq_len, vocab_size]
            teacher_logits: [seq_len, vocab_size]
            
        Returns:
            Scalar loss value (per token)
        """
        # Apply temperature
        s_logits = student_logits / self.temperature
        t_logits = teacher_logits / self.temperature
        
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        t_probs = F.softmax(t_logits, dim=-1)
        t_log_probs = F.log_softmax(t_logits, dim=-1)
        s_probs = F.softmax(s_logits, dim=-1)
        
        if self.kl_type == "forward":
            # Forward KL: KL(teacher || student) = E_t[log(t/s)]
            # Encourages student to cover all modes of teacher
            kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1).mean()
        elif self.kl_type == "reverse":
            # Reverse KL: KL(student || teacher) = E_s[log(s/t)]
            # Encourages student to focus on high-probability modes
            kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1).mean()
        else:  # jsd
            # Generalized JSD
            m_probs = self.beta * t_probs + (1 - self.beta) * s_probs
            m_log_probs = m_probs.log()
            
            kl_t_m = (t_probs * (t_log_probs - m_log_probs)).sum(dim=-1).mean()
            kl_s_m = (s_probs * (s_log_probs - m_log_probs)).sum(dim=-1).mean()
            kl = self.beta * kl_t_m + (1 - self.beta) * kl_s_m
        
        # Scale by temperature^2
        return kl * (self.temperature ** 2)
    
    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """Prepare inputs by moving to correct device."""
        inputs = super()._prepare_inputs(inputs)
        
        # Move teacher model to same device if needed
        if hasattr(self, 'teacher_model'):
            device = next(self.model.parameters()).device
            if next(self.teacher_model.parameters()).device != device:
                self.teacher_model = self.teacher_model.to(device)
        
        return inputs
