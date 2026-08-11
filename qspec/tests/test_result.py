import json

from qspec.result import SampleResult, append_result_jsonl


def test_append_result_jsonl(tmp_path):
    output_path = tmp_path / "results.jsonl"

    result = SampleResult(
        sample_id="test-0",
        question="What is 6 multiplied by 7?",
        reference_answer="6 * 7 = 42.\n#### 42",
        generated_answer="The answer is 42.",
        prompt_length=10,
        answer_length=4,
        num_matches=3,
        agreement=0.75,
        top1_w4a16=[10, 20, 30, 40],
        top1_w4a4=[10, 99, 30, 40],
        matches=[True, False, True, True],
        first_mismatch_position=1,
        w4a16_is_correct=True,
    )

    append_result_jsonl(result, output_path)

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["sample_id"] == "test-0"
    assert record["num_matches"] == 3
    assert record["agreement"] == 0.75
    assert record["matches"] == [
        True,
        False,
        True,
        True,
    ]