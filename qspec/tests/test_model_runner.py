import pytest
import torch

from qspec.compare_top1 import calculate_top1_agreement
from qspec.dummy_runner import DummyModelRunner
from qspec.model_runner import QuantizationMode


def test_dummy_runner_generation():
    runner = DummyModelRunner()

    result = runner.generate_w4a16(
        "Q: What is 6 multiplied by 7?\nA:",
        max_new_tokens=128,
    )

    assert result.generated_text == "The answer is 42."
    assert result.prompt_length == 3
    assert result.answer_length == 4
    assert result.prompt_token_ids.dtype == torch.long
    assert result.answer_token_ids.dtype == torch.long


def test_dummy_runner_top1_agreement():
    runner = DummyModelRunner()

    generation = runner.generate_w4a16(
        "Q: What is 6 multiplied by 7?\nA:",
        max_new_tokens=128,
    )

    logits_w4a16 = runner.get_answer_logits(
        generation.prompt_token_ids,
        generation.answer_token_ids,
        mode=QuantizationMode.W4A16,
    )

    logits_w4a4 = runner.get_answer_logits(
        generation.prompt_token_ids,
        generation.answer_token_ids,
        mode=QuantizationMode.W4A4,
    )

    agreement = calculate_top1_agreement(
        logits_w4a16,
        logits_w4a4,
    )

    assert logits_w4a16.shape == (4, 3)
    assert logits_w4a4.shape == (4, 3)

    assert agreement.num_tokens == 4
    assert agreement.num_matches == 2
    assert agreement.agreement == 0.5
    assert agreement.matches == [
        True,
        False,
        True,
        False,
    ]


def test_dummy_runner_rejects_empty_prompt():
    runner = DummyModelRunner()

    with pytest.raises(
        ValueError,
        match="prompt must not be empty",
    ):
        runner.generate_w4a16(
            "",
            max_new_tokens=128,
        )