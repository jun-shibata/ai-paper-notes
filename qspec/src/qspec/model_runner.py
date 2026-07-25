from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import torch

class QuantizationMode(StrEnum):
    W4A16 = "w4a16"
    W4A4 = "w4a4"


@dataclass(frozen=True)
class GenerationResult:
    """Result of W4A16 greedy generation."""

    prompt_token_ids: torch.Tensor
    answer_token_ids: torch.Tensor
    generated_text: str

    @property
    def prompt_length(self) -> int:
        return self.prompt_token_ids.numel()

    @property
    def answer_length(self) -> int:
        return self.answer_token_ids.numel()


class ModelRunner(ABC):
    """Interface between QSPEC models and evaluation code."""

    @abstractmethod
    def generate_w4a16(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> GenerationResult:
        """Generate a golden answer using W4A16 greedy decoding."""

    @abstractmethod
    def get_answer_logits(
        self,
        prompt_token_ids: torch.Tensor,
        answer_token_ids: torch.Tensor,
        *,
        mode: QuantizationMode,
    ) -> torch.Tensor:
        """Calculate logits predicting the answer tokens.

        Args:
            prompt_token_ids:
                One-dimensional tensor containing prompt token IDs.

            answer_token_ids:
                One-dimensional tensor containing the answer generated
                by W4A16.

            mode:
                Quantization mode used for the forward pass.

        Returns:
            Tensor with shape:

                [answer_length, vocabulary_size]

            Element [t] contains the logits used to predict answer
            token t under the shared W4A16-generated prefix.
        """