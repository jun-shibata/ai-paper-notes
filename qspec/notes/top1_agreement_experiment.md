# QSpec Top-1 Agreement Experiment

## Objective

Measure token-level Top-1 prediction agreement between W4A16 and
W4A4 under the same prefix.

## Protocol

1. Generate a golden answer using W4A16 greedy decoding.
2. Concatenate the prompt and the W4A16-generated answer.
3. Run teacher-forced forward passes using W4A16 and W4A4.
4. Extract logits that predict the answer tokens.
5. Compare the Top-1 token IDs at every answer position.

## Initial configuration

- Model: Llama-3-8B-Instruct
- Dataset: GSM8K
- Prompting: 8-shot
- Decoding: greedy
- Quantization: Atom
- Weight precision: INT4
- Activation precision: FP16 / INT4
- Group size: 128
- Initial samples: 10
- Maximum generated tokens: 512

## Primary metric

Top-1 agreement:

agreement = number of matching Top-1 tokens / number of answer tokens

## Additional metrics

- Per-sample agreement
- Sequence-level agreement
- First mismatch position
- W4A16 Top-1 margin
- Agreement for correct/incorrect W4A16 answers