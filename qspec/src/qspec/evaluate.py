from qspec.compare_top1 import (
    calculate_top1_agreement,
)
from qspec.dataset import GSM8KSample
from qspec.model_runner import (
    ModelRunner,
    QuantizationMode,
)
from qspec.prompt import build_gsm8k_prompt
from qspec.result import SampleResult


def find_first_mismatch(
    matches: list[bool],
) -> int | None:
    return next(
        (
            index
            for index, matched in enumerate(matches)
            if not matched
        ),
        None,
    )


def evaluate_sample(
    sample: GSM8KSample,
    runner: ModelRunner,
    *,
    max_new_tokens: int = 512,
) -> SampleResult:
    prompt = build_gsm8k_prompt(
        sample.question
    )

    generation = runner.generate_w4a16(
        prompt,
        max_new_tokens=max_new_tokens,
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

    return SampleResult(
        sample_id=sample.sample_id,
        question=sample.question,
        reference_answer=sample.reference_answer,
        generated_answer=generation.generated_text,
        prompt_length=generation.prompt_length,
        answer_length=generation.answer_length,
        num_matches=agreement.num_matches,
        agreement=agreement.agreement,
        top1_w4a16=agreement.top1_w4a16,
        top1_w4a4=agreement.top1_w4a4,
        matches=agreement.matches,
        first_mismatch_position=find_first_mismatch(
            agreement.matches
        ),
        w4a16_is_correct=None,
    )