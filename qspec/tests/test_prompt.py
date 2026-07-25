import pytest

from qspec.prompt import (
    GSM8K_8SHOT_EXAMPLES,
    FewShotExample,
    build_gsm8k_prompt,
)


def test_standard_prompt_contains_eight_examples():
    prompt = build_gsm8k_prompt(
        "What is 6 multiplied by 7?"
    )

    # 8-shot exampleと評価対象の1問
    assert prompt.count("Q:") == 9

    # 8-shotの回答と、評価対象直前のA:
    assert prompt.count("A:") == 9

    assert prompt.endswith(
        "Q: What is 6 multiplied by 7?\nA:"
    )


def test_standard_examples_have_expected_size():
    assert len(GSM8K_8SHOT_EXAMPLES) == 8


def test_build_prompt_with_custom_examples():
    examples = [
        FewShotExample(
            question="What is 1 + 1?",
            answer="1 + 1 = 2. The answer is 2.",
        )
    ]

    prompt = build_gsm8k_prompt(
        question="What is 2 + 3?",
        examples=examples,
    )

    expected = (
        "Q: What is 1 + 1?\n"
        "A: 1 + 1 = 2. The answer is 2.\n\n"
        "Q: What is 2 + 3?\n"
        "A:"
    )

    assert prompt == expected


def test_empty_question_is_rejected():
    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        build_gsm8k_prompt("   ")