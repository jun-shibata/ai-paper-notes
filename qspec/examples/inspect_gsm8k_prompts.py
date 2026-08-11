from qspec.dataset import load_gsm8k_samples
from qspec.prompt import build_gsm8k_prompt


def main() -> None:
    samples = load_gsm8k_samples(
        split="test",
        limit=10,
    )

    print(f"Loaded {len(samples)} samples")

    for sample in samples:
        prompt = build_gsm8k_prompt(
            sample.question
        )

        print("=" * 80)
        print(f"Sample ID: {sample.sample_id}")
        print(f"Prompt length in characters: {len(prompt)}")
        print()
        print(prompt)
        print()


if __name__ == "__main__":
    main()