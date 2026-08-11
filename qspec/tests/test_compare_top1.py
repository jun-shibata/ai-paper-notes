import torch
from qspec.compare_top1 import calculate_top1_agreement, extract_answer_logits

def test_calculate_top1_agreement():
    logits_w4a16 = torch.tensor([
        [0.1, 0.8, 0.1],  # token 1
        [0.7, 0.2, 0.1],  # token 0
        [0.1, 0.2, 0.7],  # token 2
        [0.1, 0.8, 0.1],  # token 1
    ])

    logits_w4a4 = torch.tensor([
        [0.2, 0.7, 0.1],  # token 1: match
        [0.1, 0.2, 0.7],  # token 2: mismatch
        [0.2, 0.1, 0.7],  # token 2: match
        [0.7, 0.2, 0.1],  # token 0: mismatch
    ])

    result = calculate_top1_agreement(
        logits_w4a16,
        logits_w4a4,
    )

    assert result.num_tokens == 4
    assert result.num_matches == 2
    assert result.agreement == 0.5
    assert result.top1_w4a16 == [1, 0, 2, 1]
    assert result.top1_w4a4 == [1, 2, 2, 0]
    assert result.matches == [True, False, True, False]

def test_extract_answer_logits():
    logits = torch.arange(
        1 * 5 * 3,
        dtype=torch.float32,
    ).reshape(1, 5, 3)

    answer_logits = extract_answer_logits(
        logits,
        prompt_length=2,
        answer_length=2,
    )

    expected = logits[0, 1:3, :]

    assert torch.equal(answer_logits, expected)