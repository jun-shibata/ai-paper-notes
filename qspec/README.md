# QSpec Top-1 Agreement Evaluation for W4A4 and W4A16

## Overview

This experiment evaluates token-level Top-1 agreement between the W4A4 speculative model and the W4A16 target model during QSpec decoding.

At each speculative decoding position:

1. The W4A4 model proposes a token.
2. The W4A16 model scores the same position.
3. The Top-1 token predicted by each model is recorded.
4. The predictions are counted as an agreement when their Top-1 token IDs are identical.

The tracing script also records the probability assigned to each proposed token, confidence-threshold statistics, and the status produced by QSpec's configured speculative sampler.

The script applies a runtime hook to `SpecDecodeWorker._verify_tokens`. It does not modify the original QSpec source files.


## Results

The evaluation used all 1,319 samples from the GSM8K test split with 8-shot prompts. QSpec was configured to generate up to three speculative tokens per draft–verify cycle. In total, 275,049 speculative positions were traced.

<img src="https://github.com/jun-shibata/ai-paper-notes/blob/main/qspec/results/qspec_figure2_rep_20260810.png" width="60%">

Each point represents a single draft position; points where the Top-1 tokens from both modes match are classified as "Top-1 Match," while those that do not match are classified as "Top-1 Mismatch." The density curves at the top and right of the graph show the marginal distributions of Top-1 probabilities for W4A16 and W4A4, respectively.  
Matching points are concentrated in the high-probability region at the top right, whereas mismatches are more frequent in regions of relatively low probability. This distribution visually demonstrates the relationship between high Top-1 probability and a high match rate.

W4A4 and W4A16 predicted the same Top-1 token at 263,984 positions, resulting in an overall Top-1 agreement rate of 95.98%.

| Metric | Result |
| :-: | :--- |
| Requests | 1,319 |
| Traced speculative positions | 275,049 |
| Top-1 agreements | 263,984 (95.98%) |
| Top-1 disagreements | 11,065 (4.02%) |

The mean Top-1 probabilities were 92.51% for W4A4 and 92.99% for W4A16. Their median probabilities were 99.966% and 99.986%, respectively.

| Metric | W4A4 | W4A16 |
| :-: | :--- | :--- |
| Mean Top-1 probability | 92.51% | 92.99% |
| Median Top-1 probability | 99.966% | 99.986% |

Both models assigned a Top-1 probability greater than 0.8 at 82.58% of the traced positions. Within this high-confidence region, the Top-1 agreement rate reached 99.96%. When both probabilities were at most 0.8, the agreement rate decreased to 70.56%.

These results show that W4A4 and W4A16 produce highly similar token-level predictions during operational QSpec decoding, particularly when both modes are confident in their predictions. This finding is consistent with the token-level similarity reported in the QSpec paper.

## Requirements

The Docker image is based on the following environment:

- Ubuntu 22.04
- NVIDIA CUDA 12.5.1 with cuDNN
- Python 3.10
- `uv` 0.12.3
- An NVIDIA GPU compatible with the configured CUDA architecture
- NVIDIA Container Toolkit

The provided Dockerfile sets:

```text
TORCH_CUDA_ARCH_LIST=8.9
```

This value targets NVIDIA GPUs with compute capability 8.9, such as the NVIDIA L40S. Change this value in the Dockerfile when building for a different GPU architecture.

## Building the Docker Image

The Docker build context must be the root directory of the original QSpec repository because the Dockerfile copies and builds the complete QSpec source tree.

First, clone QSpec and its submodules:

```bash
git clone --recursive <QSPEC_REPOSITORY_URL>
cd QSpec
```

If the repository was cloned without its submodules, initialize them separately:

```bash
git submodule update --init --recursive
```

Copy the Dockerfile from this repository into the QSpec repository root:

```bash
cp /path/to/ai-paper-notes/qspec/src/Dockerfile .
```

Then build the image:

```bash
docker build -t qspec-trace:latest .
```

## Starting the Container

Run the container with GPU access and mount directories containing the model checkpoints and experiment results:

```bash
docker run --gpus all \
  --ipc=host \
  --rm -it \
  -v /path/to/models:/models:ro \
  -v /path/to/ai-paper-notes/qspec/results:/workspace/QSpec/results \
  qspec-trace:latest
```

The container starts in:

```text
/workspace/QSpec
```

## Installing the Tracing Script

Copy the tracing script into the root of the QSpec repository. When using the container described above, this can be done from the host before building the image:

```bash
cp /path/to/ai-paper-notes/qspec/src/trace_qspec_tokens.py \
  /path/to/QSpec/trace_qspec_tokens.py
```

If the image has already been built, mount the script when starting the container:

```bash
docker run --gpus all \
  --ipc=host \
  --rm -it \
  -v /path/to/ai-paper-notes/qspec/src/trace_qspec_tokens.py:/workspace/QSpec/trace_qspec_tokens.py:ro \
  -v /path/to/models:/models:ro \
  -v /path/to/ai-paper-notes/qspec/results:/workspace/QSpec/results \
  qspec-trace:latest
```

Run all commands below from the QSpec repository root.

## Running the Token-Level Trace

The target W4A16 model is specified with `--model`, and the speculative W4A4 model is specified with `--speculative-model`.

A basic run using the GSM8K dataset is:

```bash
python trace_qspec_tokens.py \
  --model <W4A16_MODEL> \
  --speculative-model <W4A4_MODEL> \
  --num-speculative-tokens 4 \
  --num-samples 10 \
  --shots 8 \
  --max-output-tokens 512 \
  --confidence-threshold 0.8 \
  --experiment-seed 0 \
  --trace-output-dir results/qspec-token-trace
```

Replace:

- `<W4A16_MODEL>` with the path or model identifier for the W4A16 target model.
- `<W4A4_MODEL>` with the path or model identifier for the W4A4 speculative model.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python trace_qspec_tokens.py \
  --model /data/models/Llama3_8B_Instruct_QSpec \
  --speculative-model /data/models/Llama3_8B_Instruct_QSpec \
  --num-speculative-tokens 3 \
  --max-num-seqs 1 \
  --trust-remote-code \
  --enforce-eager \
  --num-samples 1319 \
  --shots 8 \
  --confidence-threshold 0.8 \
  --trace-output-dir results/qspec-token-trace-1319
```

Additional QSpec/vLLM engine options can be passed to the script because it accepts the arguments provided by `vllm.EngineArgs`.

To inspect all available arguments:

```bash
python trace_qspec_tokens.py --help
```

### Default Experiment Settings

Unless overridden, the script uses:

| Argument | Default | Description |
|---|---:|---|
| `--trace-output-dir` | `results/qspec-token-trace` | Directory in which trace files are saved |
| `--num-samples` | `10` | Number of GSM8K test samples |
| `--shots` | `8` | Number of GSM8K training examples included in each prompt |
| `--experiment-seed` | `0` | Seed used for sample selection and generation |
| `--max-output-tokens` | `512` | Maximum number of generated tokens per request |
| `--confidence-threshold` | `0.8` | Threshold used for the high-confidence metrics |

The following arguments are required:

```text
--speculative-model
--num-speculative-tokens
```

The target model must be provided through `--model` according to the QSpec/vLLM configuration.

## Using Custom Prompts

Instead of GSM8K, prompts can be supplied as a JSON Lines file with `--prompt-jsonl`.

Each non-empty line must contain a JSON object with a `prompt` field:

```json
{"sample_id": "example-001", "prompt": "Explain speculative decoding."}
{"sample_id": "example-002", "prompt": "What is weight quantization?"}
```

`sample_id` is optional. The script also accepts `request_id`. If neither is present, a zero-based line index is used as the request ID.

Run the trace with custom prompts as follows:

```bash
python trace_qspec_tokens.py \
  --model <W4A16_MODEL> \
  --speculative-model <W4A4_MODEL> \
  --num-speculative-tokens 4 \
  --prompt-jsonl /path/to/prompts.jsonl \
  --max-output-tokens 512 \
  --experiment-seed 0 \
  --trace-output-dir results/custom-prompt-trace
```

When `--prompt-jsonl` is supplied, the GSM8K-specific `--num-samples` and `--shots` settings are not used.

## Output Files

The script creates the output directory automatically and writes three files:

```text
results/qspec-token-trace/
├── qspec_token_trace.csv
├── qspec_token_trace.jsonl
└── qspec_token_summary.json
```

### `qspec_token_trace.csv`

A tabular token-level trace suitable for analysis and plotting.

### `qspec_token_trace.jsonl`

The same token-level records in JSON Lines format.

Each trace record includes:

- Request and decoding-cycle identifiers
- Draft and absolute output positions
- W4A4 proposed token ID and decoded text
- Probability assigned to the proposed token by W4A4
- Probability assigned to the same token by W4A16
- W4A4 and W4A16 Top-1 token IDs
- W4A4 and W4A16 Top-1 probabilities
- Top-1 agreement
- Confidence-threshold flags
- Runtime acceptance status

### `qspec_token_summary.json`

A summary containing the experiment configuration, runtime statistics, and aggregate metrics, including:

- Total number of draft tokens
- Top-1 agreement and rejection rates
- High-confidence rates for W4A4 and W4A16
- Top-1 agreement conditioned on both models being highly confident
- Runtime acceptance, rejection, and discard rates
- Total generated tokens and generation throughput

The same metric summary is also printed to the terminal when the run finishes.

## Interpretation of the Metrics

`top1_agreement_rate` is the fraction of traced speculative positions at which W4A4 and W4A16 predict the same Top-1 token:

```text
W4A4 Top-1 token ID == W4A16 Top-1 token ID
```

`top1_agreement_given_both_high_confidence` applies the same comparison only to positions where both models' Top-1 probabilities are greater than the configured confidence threshold.

The confidence comparison is strict. With:

```bash
--confidence-threshold 0.8
```

a probability must be greater than `0.8`, rather than equal to it, to be classified as high confidence.

`runtime_status` describes the behavior of QSpec's configured speculative sampler:

- `accepted`: the proposed token was emitted at that position.
- `rejected`: a different token was emitted.
- `discarded`: the position was not considered after an earlier rejection, or no token was emitted for that position.

The runtime acceptance rate and Top-1 agreement rate measure related but distinct properties. Top-1 agreement compares greedy predictions, whereas `runtime_status` records the sampler's actual behavior.

## Notes on Reproducibility

- Generation uses greedy decoding with `temperature=0.0` and `top_p=1.0`.
- The default experiment seed is `0`.
- The GSM8K test subset is selected deterministically from the experiment seed.
- The first `--shots` examples from the GSM8K training split are used as few-shot examples.
- The script loads `openai/gsm8k` through the Hugging Face `datasets` library, so network access or an existing local cache is required for the first GSM8K run.
- Model checkpoints, QSpec revisions, GPU type, CUDA environment, and command-line arguments should be recorded alongside reported results.
- The trace is an operational QSpec trace collected during decoding; it is not a standalone comparison of independently generated W4A4 and W4A16 sequences.
