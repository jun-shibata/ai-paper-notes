import pytest

from qspec.dataset import GSM8KSample, convert_gsm8k_row
from qspec.prompt import FewShotExample, build_gsm8k_prompt
import json
from qspec.result import SampleResult, append_result_jsonl

def test_gsm8k_sample():
    sample = GSM8KSample(
        sample_id="test-0",
        question="What is 1 + 1?",
        reference_answer="2",
    )

    assert sample.question == "What is 1 + 1?"
    assert sample.reference_answer == "2"


def test_build_gsm8k_prompt():
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

    assert "Question: What is 1 + 1?" in prompt
    assert "Question: What is 2 + 3?" in prompt
    assert prompt.endswith("Answer:")


def test_append_result_jsonl(tmp_path):
    output_path = tmp_path / "results.jsonl"

    result = SampleResult(
        sample_id="test-0",
        question="What is 1 + 1?",
        reference_answer="1 + 1 = 2.\n#### 2",
        generated_answer="2",
        prompt_length=10,
        answer_length=1,
        num_matches=1,
        agreement=1.0,
        top1_w4a16=[2],
        top1_w4a4=[2],
        matches=[True],
        first_mismatch_position=None,
        w4a16_is_correct=True,
    )

    append_result_jsonl(result, output_path)

    record = json.loads(
        output_path.read_text(encoding="utf-8").strip()
    )

    assert record["sample_id"] == "test-0"
    assert record["agreement"] == 1.0


def test_comvert_ges8k_row():
    row = {
        "question": "What is 6 multiplied by 7?",
        "answer": "6 * 7 = 42.\n#### 42",
    }

    sample = convert_gsm8k_row(
        row,
        split="test",
        index=3,
    )

    assert sample.sample_id == "test-3"
    assert sample.question == "What is 6 multiplied by 7?"
    assert sample.reference_answer == "6 * 7 = 42.\n#### 42"