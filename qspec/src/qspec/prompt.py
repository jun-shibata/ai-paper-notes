from dataclasses import dataclass


@dataclass(frozen=True)
class FewShotExample:
    question: str
    answer: str


GSM8K_8SHOT_EXAMPLES = [
    FewShotExample(
        question=(
            "There are 15 trees in the grove. Grove workers will plant "
            "trees in the grove today. After they are done, there will "
            "be 21 trees. How many trees did the grove workers plant today?"
        ),
        answer=(
            "There are 15 trees originally. Then there were 21 trees "
            "after some more were planted. So there must have been "
            "21 - 15 = 6. The answer is 6."
        ),
    ),
    FewShotExample(
        question=(
            "If there are 3 cars in the parking lot and 2 more cars "
            "arrive, how many cars are in the parking lot?"
        ),
        answer=(
            "There are originally 3 cars. 2 more cars arrive. "
            "3 + 2 = 5. The answer is 5."
        ),
    ),
    FewShotExample(
        question=(
            "Leah had 32 chocolates and her sister had 42. If they "
            "ate 35, how many pieces do they have left in total?"
        ),
        answer=(
            "Originally, Leah had 32 chocolates. Her sister had 42. "
            "So in total they had 32 + 42 = 74. After eating 35, "
            "they had 74 - 35 = 39. The answer is 39."
        ),
    ),
    FewShotExample(
        question=(
            "Jason had 20 lollipops. He gave Denny some lollipops. "
            "Now Jason has 12 lollipops. How many lollipops did "
            "Jason give to Denny?"
        ),
        answer=(
            "Jason started with 20 lollipops. Then he had 12 after "
            "giving some to Denny. So he gave Denny "
            "20 - 12 = 8. The answer is 8."
        ),
    ),
    FewShotExample(
        question=(
            "Shawn has five toys. For Christmas, he got two toys "
            "each from his mom and dad. How many toys does he have now?"
        ),
        answer=(
            "Shawn started with 5 toys. If he got 2 toys each from "
            "his mom and dad, then that is 4 more toys. "
            "5 + 4 = 9. The answer is 9."
        ),
    ),
    FewShotExample(
        question=(
            "There were nine computers in the server room. Five more "
            "computers were installed each day, from Monday to Thursday. "
            "How many computers are now in the server room?"
        ),
        answer=(
            "There were originally 9 computers. For each of 4 days, "
            "5 more computers were added. So 5 * 4 = 20 computers "
            "were added. 9 + 20 is 29. The answer is 29."
        ),
    ),
    FewShotExample(
        question=(
            "Michael had 58 golf balls. On Tuesday, he lost 23 golf "
            "balls. On Wednesday, he lost 2 more. How many golf balls "
            "did he have at the end of Wednesday?"
        ),
        answer=(
            "Michael started with 58 golf balls. After losing 23 on "
            "Tuesday, he had 58 - 23 = 35. After losing 2 more, he "
            "had 35 - 2 = 33 golf balls. The answer is 33."
        ),
    ),
    FewShotExample(
        question=(
            "Olivia has $23. She bought five bagels for $3 each. "
            "How much money does she have left?"
        ),
        answer=(
            "Olivia had 23 dollars. 5 bagels for 3 dollars each will "
            "be 5 * 3 = 15 dollars. So she has 23 - 15 dollars left. "
            "23 - 15 is 8. The answer is 8."
        ),
    ),
]


def build_gsm8k_prompt(
    question: str,
    examples: list[FewShotExample] | None = None,
) -> str:
    """Build an 8-shot Chain-of-Thought prompt for GSM8K.

    Args:
        question:
            Evaluation question to append after the few-shot examples.
        examples:
            Few-shot examples. The standard fixed GSM8K examples are
            used when this argument is omitted.

    Returns:
        Prompt formatted as alternating Q/A sections.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    if examples is None:
        examples = GSM8K_8SHOT_EXAMPLES

    sections: list[str] = []

    for example in examples:
        sections.append(
            f"Q: {example.question}\n"
            f"A: {example.answer}"
        )

    sections.append(
        f"Q: {question}\n"
        "A:"
    )

    return "\n\n".join(sections)