"""
Trace token-level W4A4/W4A16 probabilities during QSpec decoding.

Run this script from the root of the QSpec repository.
It monkey-patches SpecDecodeWorker._verify_tokens at runtime,
so no QSpec source file needs to be edited.

The trace is an operational QSpec trace: W4A4 proposes tokens and W4A16 scores the same positions.
`top1_match` implements the greedy acceptance criterion described in the QSpec paper.
`runtime_status` records what the repository's configured speculative sampler actually emitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import torch

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.spec_decode.util import split_batch_by_proposal_len
from vllm.utils import FlexibleArgumentParser


@dataclass
class TokenTrace:
    request_id: str
    cycle_id: int
    batch_row: int
    draft_position: int
    output_position_before_cycle: int
    absolute_output_position: int
    prompt_length: int
    proposed_token_id: int
    proposed_token_text: str
    w4a4_proposed_probability: float
    w4a16_proposed_probability: float
    w4a4_top1_token_id: int
    w4a16_top1_token_id: int
    w4a4_top1_probability: float
    w4a16_top1_probability: float
    top1_match: bool
    w4a4_high_confidence: bool
    w4a16_high_confidence: bool
    both_high_confidence: bool
    runtime_status: str


class QSpecTracer:
    def __init__(self, confidence_threshold: float) -> None:
        self.confidence_threshold = confidence_threshold
        self.cycle_id = 0
        self.rows: List[TokenTrace] = []
        self.tokenizer: Any = None

    def set_tokenizer(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def decode_token(self, token_id: int) -> str:
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )

    @staticmethod
    def _lengths(metadata: Any) -> Tuple[int, int]:
        sequence_data = next(iter(metadata.seq_data.values()))
        return sequence_data.get_prompt_len(), sequence_data.get_output_len()

    def record(
        self,
        metadata_list: Sequence[Any],
        proposals: Any,
        proposal_scores: Any,
        accepted_token_ids: torch.Tensor,
    ) -> None:
        proposal_lens = proposals.proposal_lens.tolist()
        (_, spec_indices), _ = split_batch_by_proposal_len(
            metadata_list, proposal_lens
        )
        if not spec_indices:
            return

        self.cycle_id += 1

        draft_probs = proposals.proposal_probs[spec_indices].detach()
        draft_ids = proposals.proposal_token_ids[spec_indices].detach()
        target_probs = proposal_scores.probs[spec_indices, :-1].detach()

        draft_top_probs, draft_top_ids = draft_probs.max(dim=-1)
        target_top_probs, target_top_ids = target_probs.max(dim=-1)

        gather_ids = draft_ids.unsqueeze(-1)
        selected_draft = draft_probs.gather(-1, gather_ids).squeeze(-1)
        selected_target = target_probs.gather(-1, gather_ids).squeeze(-1)

        tensors = [
            draft_ids,
            draft_top_probs,
            draft_top_ids,
            target_top_probs,
            target_top_ids,
            selected_draft,
            selected_target,
            accepted_token_ids[spec_indices],
        ]
        (
            draft_ids,
            draft_top_probs,
            draft_top_ids,
            target_top_probs,
            target_top_ids,
            selected_draft,
            selected_target,
            emitted_ids,
        ) = [tensor.cpu() for tensor in tensors]

        for local_row, original_index in enumerate(spec_indices):
            metadata = metadata_list[original_index]
            prompt_length, output_length = self._lengths(metadata)
            rejection_seen = False

            for position in range(draft_ids.shape[1]):
                proposed_id = int(draft_ids[local_row, position])
                emitted_id = int(emitted_ids[local_row, position])

                if rejection_seen or emitted_id == -1:
                    runtime_status = "discarded"
                elif emitted_id == proposed_id:
                    runtime_status = "accepted"
                else:
                    runtime_status = "rejected"
                    rejection_seen = True

                p4_top = float(draft_top_probs[local_row, position])
                p16_top = float(target_top_probs[local_row, position])
                top1_match = (
                    int(draft_top_ids[local_row, position])
                    == int(target_top_ids[local_row, position])
                )

                self.rows.append(
                    TokenTrace(
                        request_id=str(metadata.request_id),
                        cycle_id=self.cycle_id,
                        batch_row=original_index,
                        draft_position=position,
                        output_position_before_cycle=output_length,
                        absolute_output_position=output_length + position,
                        prompt_length=prompt_length,
                        proposed_token_id=proposed_id,
                        proposed_token_text=self.decode_token(proposed_id),
                        w4a4_proposed_probability=float(
                            selected_draft[local_row, position]
                        ),
                        w4a16_proposed_probability=float(
                            selected_target[local_row, position]
                        ),
                        w4a4_top1_token_id=int(
                            draft_top_ids[local_row, position]
                        ),
                        w4a16_top1_token_id=int(
                            target_top_ids[local_row, position]
                        ),
                        w4a4_top1_probability=p4_top,
                        w4a16_top1_probability=p16_top,
                        top1_match=top1_match,
                        w4a4_high_confidence=(
                            p4_top > self.confidence_threshold
                        ),
                        w4a16_high_confidence=(
                            p16_top > self.confidence_threshold
                        ),
                        both_high_confidence=(
                            p4_top > self.confidence_threshold
                            and p16_top > self.confidence_threshold
                        ),
                        runtime_status=runtime_status,
                    )
                )


def install_trace_hook(tracer: QSpecTracer) -> None:
    from vllm.spec_decode.spec_decode_worker import SpecDecodeWorker

    original = SpecDecodeWorker._verify_tokens

    def traced_verify_tokens(
        worker: Any,
        seq_group_metadata_list: Sequence[Any],
        proposal_scores: Any,
        proposals: Any,
        max_proposal_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        accepted_ids, logprobs = original(
            worker,
            seq_group_metadata_list,
            proposal_scores,
            proposals,
            max_proposal_len,
        )
        tracer.record(
            seq_group_metadata_list,
            proposals,
            proposal_scores,
            accepted_ids,
        )
        return accepted_ids, logprobs

    SpecDecodeWorker._verify_tokens = traced_verify_tokens


def build_gsm8k_prompts(
    num_samples: Optional[int],
    shots: int,
    seed: int,
    max_tokens: int,
) -> List[Tuple[str, SamplingParams, str]]:
    import datasets
    from vllm import get_conv_template, get_conv_template_name

    train = datasets.load_dataset("openai/gsm8k", "main", split="train")
    test = datasets.load_dataset("openai/gsm8k", "main", split="test")

    examples = []
    for index in range(shots):
        examples.append(
            f"Question: {train[index]['question']}  "
            f"Answer: {train[index]['answer']}"
        )
    prefix = "\n".join(examples) + "\n"

    indices = list(range(len(test)))
    if num_samples is not None and num_samples < len(indices):
        random.Random(seed).shuffle(indices)
        indices = sorted(indices[:num_samples])

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop_token_ids=[128001, 128009],
        stop=["Question:"],
        seed=seed,
    )

    prompts: List[Tuple[str, SamplingParams, str]] = []
    template_name = get_conv_template_name("Meta-Llama3-8B-Instruct")
    for index in indices:
        raw = (
            prefix
            + f"Question: {test[index]['question']} Answer: "
        )
        conversation = get_conv_template(template_name)
        conversation.append_message(conversation.roles[0], raw)
        conversation.append_message(conversation.roles[1], "")
        prompts.append(
            (conversation.get_prompt(), sampling, f"gsm8k-test-{index}")
        )
    return prompts


def load_jsonl_prompts(
    path: Path,
    max_tokens: int,
    seed: int,
) -> List[Tuple[str, SamplingParams, str]]:
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=seed,
    )
    prompts = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item["prompt"]
            request_id = str(
                item.get("sample_id", item.get("request_id", line_number - 1))
            )
            prompts.append((prompt, sampling, request_id))
    return prompts


def run_requests(
    engine: LLMEngine,
    prompts: List[Tuple[str, SamplingParams, str]],
) -> dict[str, Any]:
    pending = list(prompts)
    submitted = 0
    finished = 0
    generated_tokens = 0
    started = time.perf_counter()

    while pending or engine.has_unfinished_requests():
        if pending:
            prompt, sampling, request_id = pending.pop(0)
            engine.add_request(request_id, prompt, sampling)
            submitted += 1

        outputs: List[RequestOutput] = engine.step()
        for output in outputs:
            if output.finished:
                finished += 1
                generated_tokens += len(output.outputs[0].token_ids)

    elapsed = time.perf_counter() - started
    return {
        "submitted_requests": submitted,
        "finished_requests": finished,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "throughput_tokens_per_second": (
            generated_tokens / elapsed if elapsed else math.nan
        ),
    }


def fraction(rows: Iterable[TokenTrace], predicate: Any) -> float:
    values = list(rows)
    if not values:
        return math.nan
    return sum(bool(predicate(row)) for row in values) / len(values)


def summarize(rows: List[TokenTrace], threshold: float) -> dict[str, Any]:
    statuses = defaultdict(int)
    for row in rows:
        statuses[row.runtime_status] += 1

    high_both = [row for row in rows if row.both_high_confidence]
    return {
        "confidence_threshold": threshold,
        "total_draft_tokens": len(rows),
        "top1_agreement_rate": fraction(rows, lambda row: row.top1_match),
        "top1_rejection_rate": fraction(rows, lambda row: not row.top1_match),
        "w4a4_probability_above_threshold_rate": fraction(
            rows, lambda row: row.w4a4_top1_probability > threshold
        ),
        "w4a16_probability_above_threshold_rate": fraction(
            rows, lambda row: row.w4a16_top1_probability > threshold
        ),
        "both_probabilities_above_threshold_rate": fraction(
            rows, lambda row: row.both_high_confidence
        ),
        "top1_agreement_given_both_high_confidence": fraction(
            high_both, lambda row: row.top1_match
        ),
        "runtime_status_counts": dict(statuses),
        "runtime_acceptance_rate": fraction(
            rows, lambda row: row.runtime_status == "accepted"
        ),
        "runtime_rejection_rate": fraction(
            rows, lambda row: row.runtime_status == "rejected"
        ),
        "runtime_discard_rate": fraction(
            rows, lambda row: row.runtime_status == "discarded"
        ),
    }


def write_results(
    output_dir: Path,
    rows: List[TokenTrace],
    run_stats: dict[str, Any],
    threshold: float,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "qspec_token_trace.csv"
    jsonl_path = output_dir / "qspec_token_trace.jsonl"

    fieldnames = list(TokenTrace.__dataclass_fields__)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    report = {
        "experiment": {
            "kind": "operational_qspec_trace",
            "model": args.model,
            "speculative_model": args.speculative_model,
            "num_speculative_tokens": args.num_speculative_tokens,
            "shots": args.shots,
            "num_samples": args.num_samples,
            "seed": args.experiment_seed,
        },
        "run": run_stats,
        "metrics": summarize(rows, threshold),
    }
    with (output_dir / "qspec_token_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trace-output-dir",
        type=Path,
        default=Path("results/qspec-token-trace"),
    )
    parser.add_argument("--prompt-jsonl", type=Path)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--experiment-seed", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)


def main() -> int:
    parser = FlexibleArgumentParser(
        description="Collect token-level W4A4/W4A16 QSpec traces."
    )
    parser = EngineArgs.add_cli_args(parser)
    add_experiment_args(parser)
    args = parser.parse_args()

    if not args.speculative_model:
        parser.error("--speculative-model is required for a QSpec trace")
    if args.num_speculative_tokens is None:
        parser.error("--num-speculative-tokens is required")

    torch.manual_seed(args.experiment_seed)
    torch.cuda.manual_seed_all(args.experiment_seed)

    tracer = QSpecTracer(args.confidence_threshold)
    install_trace_hook(tracer)

    if args.prompt_jsonl:
        prompts = load_jsonl_prompts(
            args.prompt_jsonl,
            args.max_output_tokens,
            args.experiment_seed,
        )
    else:
        prompts = build_gsm8k_prompts(
            args.num_samples,
            args.shots,
            args.experiment_seed,
            args.max_output_tokens,
        )

    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    tracer.set_tokenizer(engine.get_tokenizer())

    run_stats = run_requests(engine, prompts)
    write_results(
        args.trace_output_dir,
        tracer.rows,
        run_stats,
        args.confidence_threshold,
        args,
    )

    print(json.dumps(summarize(tracer.rows, args.confidence_threshold), indent=2))
    print(f"Saved traces to: {args.trace_output_dir}")
    return 0


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    sys.exit(main())