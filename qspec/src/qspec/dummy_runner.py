import torch

from qspec.model_runner import (
    GenerationResult,
    ModelRunner,
    QuantizationMode,
)


class DummyModelRunner(ModelRunner):
    """Deterministic runner for local tests."""

    def generate_w4a16(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> GenerationResult:
        if not prompt:
            raise ValueError("prompt must not be empty")

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive"
            )

        # 仮のtoken IDs
        prompt_token_ids = torch.tensor(
            [10, 20, 30],
            dtype=torch.long,
        )

        answer_token_ids = torch.tensor(
            [1, 0, 2, 1],
            dtype=torch.long,
        )

        return GenerationResult(
            prompt_token_ids=prompt_token_ids,
            answer_token_ids=answer_token_ids,
            generated_text="The answer is 42.",
        )

    def get_answer_logits(
        self,
        prompt_token_ids: torch.Tensor,
        answer_token_ids: torch.Tensor,
        *,
        mode: QuantizationMode,
    ) -> torch.Tensor:
        self._validate_token_ids(
            prompt_token_ids,
            answer_token_ids,
        )

        if mode == QuantizationMode.W4A16:
            return torch.tensor([
                [0.1, 0.8, 0.1],  # Top-1: 1
                [0.7, 0.2, 0.1],  # Top-1: 0
                [0.1, 0.2, 0.7],  # Top-1: 2
                [0.1, 0.8, 0.1],  # Top-1: 1
            ])

        if mode == QuantizationMode.W4A4:
            return torch.tensor([
                [0.2, 0.7, 0.1],  # Top-1: 1, match
                [0.1, 0.2, 0.7],  # Top-1: 2, mismatch
                [0.2, 0.1, 0.7],  # Top-1: 2, match
                [0.7, 0.2, 0.1],  # Top-1: 0, mismatch
            ])

        raise ValueError(
            f"Unsupported quantization mode: {mode}"
        )

    @staticmethod
    def _validate_token_ids(
        prompt_token_ids: torch.Tensor,
        answer_token_ids: torch.Tensor,
    ) -> None:
        if prompt_token_ids.ndim != 1:
            raise ValueError(
                "prompt_token_ids must be one-dimensional"
            )

        if answer_token_ids.ndim != 1:
            raise ValueError(
                "answer_token_ids must be one-dimensional"
            )

        if answer_token_ids.numel() != 4:
            raise ValueError(
                "DummyModelRunner expects four answer tokens"
            )