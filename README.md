# ai-paper-notes
A collection of AI paper summaries, technical explanations, and implementation notes focused on LLMs, inference optimization, quantization, MLIR, and compiler technologies.

## Experiments

### QSpec: Top-1 Agreement Between W4A4 and W4A16

[View the QSpec experiment](https://github.com/jun-shibata/ai-paper-notes/tree/main/qspec)

This experiment evaluates the token-level similarity between W4A4 and W4A16 during QSpec speculative decoding. QSpec uses the faster W4A4 mode to generate draft tokens and the higher-precision W4A16 mode to verify them.

The evaluation traced 275,049 speculative positions across all 1,319 samples in the GSM8K test split. The two modes predicted the same Top-1 token at 95.98% of the evaluated positions. When both modes assigned a Top-1 probability greater than 0.8, their agreement rate reached 99.96%.
