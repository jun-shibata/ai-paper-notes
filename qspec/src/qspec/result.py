from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class SampleResult:
    sample_id: str
    question: str
    reference_answer: str
    generated_answer: str
    prompt_length: int
    answer_length: int
    num_matches: int
    agreement: float
    top1_w4a16: list[int]
    top1_w4a4: list[int]
    matches: list[bool]
    first_mismatch_position: int | None
    w4a16_is_correct: bool | None


def append_result_jsonl(
    result: SampleResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("a", encoding="utf-8") as file:
        json.dump(
            asdict(result),
            file,
            ensure_ascii=False,
        )
        file.write("\n")