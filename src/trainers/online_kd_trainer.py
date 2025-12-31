"""
Online Knowledge Distillation Trainer for Experience Internalization

This trainer implements on-policy distillation where:
- Student model: receives Q (without experience) and generates responses A
- Teacher model: receives Q+E (with experience) and generates high-quality responses A'
- The student learns to match the teacher's output distribution

Key difference from TRL's GKDTrainer:
- Teacher and student use DIFFERENT inputs (teacher has experience E)
- This enables true experience internalization through on-policy learning

Training process:
1. For each batch, student generates responses A given input Q
2. Teacher generates responses A' given input Q+E (or evaluates A with Q+E context)
3. Compute KL divergence loss between student's A and teacher's A'
4. Update student model
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
        
        # Pad teacher inputs
        max_teacher_len = max(len(e["input_ids"]) for e in teacher_encodings)
        teacher_input_ids = []
        teacher_attention_mask = []
        for enc in teacher_encodings:
            padding_len = max_teacher_len - len(enc["input_ids"])
            # Left padding for generation
            teacher_input_ids.append([self.tokenizer.pad_token_id] * padding_len + enc["input_ids"])
            teacher_attention_mask.append([0] * padding_len + enc["attention_mask"])
        
        # Pad student inputs
        max_student_len = max(len(e["input_ids"]) for e in student_encodings)
        student_input_ids = []
        student_attention_mask = []
        for enc in student_encodings:
            padding_len = max_student_len - len(enc["input_ids"])
            # Left padding for generation
            student_input_ids.append([self.tokenizer.pad_token_id] * padding_len + enc["input_ids"])
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
    
    This trainer implements on-policy distillation where the student generates
    responses and learns from the teacher's distribution on the same responses.
    
    Key feature: Teacher uses Q+E (with experience), Student uses Q (without experience)
    
    Training modes:
    1. Teacher generates A': Student learns to match teacher's generation
    2. Teacher evaluates student's A: Student learns from teacher's feedback on its own output
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
            temperature: Temperature for generation and KL computation
            max_new_tokens: Maximum tokens to generate
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
        self.kl_type = kl_type
        self.beta = beta
        
        # Generation config for on-policy sampling
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=generation_config.get("temperature", 0.7) if generation_config else 0.7,
            top_p=generation_config.get("top_p", 0.8) if generation_config else 0.8,
            top_k=generation_config.get("top_k", 20) if generation_config else 20,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def _generate_responses(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate responses from a model.
        
        Returns:
            Tuple of (generated_ids, generated_attention_mask)
            generated_ids includes the prompt + generated tokens
        """
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=False,
            )
        
        generated_ids = outputs.sequences
        # Create attention mask for generated sequence
        generated_attention_mask = torch.ones_like(generated_ids)
        # Mask padding tokens
        generated_attention_mask[generated_ids == self.tokenizer.pad_token_id] = 0
        
        return generated_ids, generated_attention_mask
    
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute Online KD loss.
        
        Process:
        1. Student generates responses A given Q
        2. Teacher generates responses A' given Q+E
        3. Compute divergence loss between student's P(A|Q) and teacher's P(A'|Q+E)
        """
        teacher_input_ids = inputs["teacher_input_ids"]
        teacher_attention_mask = inputs["teacher_attention_mask"]
        student_input_ids = inputs["student_input_ids"]
        student_attention_mask = inputs["student_attention_mask"]
        
        batch_size = student_input_ids.size(0)
        device = student_input_ids.device
        
        # Step 1: Generate responses from teacher (with experience)
        teacher_gen_ids, teacher_gen_mask = self._generate_responses(
            self.teacher_model,
            teacher_input_ids,
            teacher_attention_mask,
        )
        
        # Get only the generated part (after prompt)
        teacher_prompt_len = teacher_input_ids.size(1)
        teacher_response_ids = teacher_gen_ids[:, teacher_prompt_len:]
        teacher_response_mask = teacher_gen_mask[:, teacher_prompt_len:]
        
        # Step 2: Prepare student input with teacher's response
        # Concatenate student prompt with teacher's response for computing logits
        student_prompt_len = student_input_ids.size(1)
        
        # Align response length
        max_response_len = teacher_response_ids.size(1)
        
        # Create full sequence for student: student_prompt + teacher_response
        student_full_ids = torch.cat([student_input_ids, teacher_response_ids], dim=1)
        student_full_mask = torch.cat([student_attention_mask, teacher_response_mask], dim=1)
        
        # Create full sequence for teacher: teacher_prompt + teacher_response
        teacher_full_ids = teacher_gen_ids
        teacher_full_mask = teacher_gen_mask
        
        # Step 3: Compute logits for both models on the response
        # Student: P(response | student_prompt)
        student_outputs = model(
            input_ids=student_full_ids,
            attention_mask=student_full_mask,
        )
        student_logits = student_outputs.logits
        
        # Teacher: P(response | teacher_prompt)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=teacher_full_ids,
                attention_mask=teacher_full_mask,
            )
            teacher_logits = teacher_outputs.logits
        
        # Step 4: Compute KL divergence on response tokens only
        # Extract logits for response positions
        # For student: positions [student_prompt_len : student_prompt_len + response_len]
        # For teacher: positions [teacher_prompt_len : teacher_prompt_len + response_len]
        
        total_loss = torch.tensor(0.0, device=device)
        total_tokens = 0
        
        for i in range(batch_size):
            # Get response length for this sample
            response_len = teacher_response_mask[i].sum().item()
            if response_len == 0:
                continue
            
            # Student logits for response positions (shifted for next-token prediction)
            s_start = student_prompt_len - 1  # -1 for next-token prediction
            s_end = student_prompt_len + int(response_len) - 1
            s_logits = student_logits[i, s_start:s_end, :]
            
            # Teacher logits for response positions
            t_start = teacher_prompt_len - 1
            t_end = teacher_prompt_len + int(response_len) - 1
            t_logits = teacher_logits[i, t_start:t_end, :]
            
            # Align lengths
            min_len = min(s_logits.size(0), t_logits.size(0))
            if min_len == 0:
                continue
            
            s_logits = s_logits[:min_len, :]
            t_logits = t_logits[:min_len, :]
            
            # Compute divergence
            loss = self._compute_divergence(s_logits, t_logits)
            total_loss = total_loss + loss * min_len
            total_tokens += min_len
        
        if total_tokens > 0:
            total_loss = total_loss / total_tokens
        
        # Log losses
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "kd_loss": total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            })
        
        if return_outputs:
            return total_loss, student_outputs
        return total_loss
    
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
            Scalar loss value
        """
        # Apply temperature
        s_logits = student_logits / self.temperature
        t_logits = teacher_logits / self.temperature
        
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        t_probs = F.softmax(t_logits, dim=-1)
        t_log_probs = F.log_softmax(t_logits, dim=-1)
        s_probs = F.softmax(s_logits, dim=-1)
        
        if self.kl_type == "forward":
            # Forward KL: KL(teacher || student)
            kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
        elif self.kl_type == "reverse":
            # Reverse KL: KL(student || teacher)
            kl = F.kl_div(t_log_probs, s_probs, reduction="batchmean")
        else:  # jsd
            # Generalized JSD: beta * KL(t||m) + (1-beta) * KL(s||m)
            # where m = beta * t + (1-beta) * s
            m_probs = self.beta * t_probs + (1 - self.beta) * s_probs
            m_log_probs = m_probs.log()
            
            kl_t_m = F.kl_div(m_log_probs, t_probs, reduction="batchmean")
            kl_s_m = F.kl_div(m_log_probs, s_probs, reduction="batchmean")
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

