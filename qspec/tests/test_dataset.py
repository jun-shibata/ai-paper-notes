from qspec.dataset import GSM8KSample

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
        generated_answer="2",
        prompt_length=10,
        answer_length=1,
        num_matches=1,
        agreement=1.0,
        top1_w4a16=[2],
        top1_w4a4=[2],
        matches=[True],
    )

    append_result_jsonl(result, output_path)

    record = json.loads(
        output_path.read_text(encoding="utf-8").strip()
    )

    assert record["sample_id"] == "test-0"
    assert record["agreement"] == 1.0