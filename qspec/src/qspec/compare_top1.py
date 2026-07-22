from dataclasses import dataclass
import torch

@dataclass
class AgreementResult:
    num_tokens: int
    num_matches: int
    agreement: float
    top1_w4a16: list[int]
    top1_w4a4: list[int]
    matches: list[bool]


def calculate_top1_agreement(
    logits_w4a16: torch.Tensor,
    logits_w4a4: torch.Tensor,
) -> AgreementResult:
    """
    Args:
        logits_w4a16:
            Shape [answer_length, vocab_size].
        logits_w4a4:
            Shape [answer_length, vocab_size].
    """

    if logits_w4a16.shape != logits_w4a4.shape:
        raise ValueError(
            "W4A16 and W4A4 logits must have the same shape: "
            f"{logits_w4a16.shape} != {logits_w4a4.shape}"
        )

    if logits_w4a16.ndim != 2:
        raise ValueError(
            "Expected logits with shape [answer_length, vocab_size]"
        )

    top1_w4a16 = logits_w4a16.argmax(dim=-1)
    top1_w4a4 = logits_w4a4.argmax(dim=-1)

    matches = top1_w4a16.eq(top1_w4a4)

    num_tokens = matches.numel()
    num_matches = int(matches.sum().item())

    return AgreementResult(
        num_tokens=num_tokens,
        num_matches=num_matches,
        agreement=num_matches / num_tokens if num_tokens else 0.0,
        top1_w4a16=top1_w4a16.tolist(),
        top1_w4a4=top1_w4a4.tolist(),
        matches=matches.tolist(),
    )


def extract_answer_logits(
    logits: torch.Tensor,
    prompt_length: int,
    answer_length: int,
) -> torch.Tensor:
    """
    Args:
        logits:
            Shape [batch_size, full_sequence_length, vocab_size].

    Returns:
        Shape [answer_length, vocab_size].

    Position i predicts the token at position i + 1.
    """

    if logits.ndim != 3:
        raise ValueError(
            "Expected logits with shape "
            "[batch_size, sequence_length, vocab_size]"
        )

    if logits.size(0) != 1:
        raise ValueError("Only batch size 1 is currently supported")

    start = prompt_length - 1
    end = start + answer_length

    if start < 0 or end > logits.size(1):
        raise ValueError(
            f"Invalid answer range: start={start}, end={end}, "
            f"sequence_length={logits.size(1)}"
        )

    return logits[0, start:end, :]