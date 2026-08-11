from qspec.dataset import GSM8KSample
from qspec.dummy_runner import DummyModelRunner
from qspec.evaluate import evaluate_sample


def test_evaluate_sample():
    sample = GSM8KSample(
        sample_id="test-0",
        question="What is 6 multiplied by 7?",
        reference_answer="6 * 7 = 42.\n#### 42",
    )

    result = evaluate_sample(
        sample,
        DummyModelRunner(),
        max_new_tokens=128,
    )

    assert result.sample_id == "test-0"
    assert result.generated_answer == "The answer is 42."
    assert result.answer_length == 4
    assert result.num_matches == 2
    assert result.agreement == 0.5
    assert result.first_mismatch_position == 1
    assert result.w4a16_is_correct is None