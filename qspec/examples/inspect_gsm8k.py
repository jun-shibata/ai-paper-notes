from qspec.dataset import load_gsm8k_samples

def main() -> None:
  samples = load_gsm8k_samples(
    split="test",
    limit=10,
  )

  print(f"Loaded {len(samples)} samples")

  for sample in samples:
    print("=" * 80)
    print(f"ID: {sample.sample_id}")
    print()
    print("Question:")
    print(sample.question)
    print()
    print("Reference answer:")
    print(sample.reference_answer)


if __name__ == "__main__":
  main()