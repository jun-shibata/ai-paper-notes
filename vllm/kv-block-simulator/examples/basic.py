from simulator import KVBlockSimulator


def main() -> None:
    simulator = KVBlockSimulator(
        num_blocks=6,
        block_size=4,
    )

    simulator.create_request("A")
    simulator.append_tokens("A", count=6)
    simulator.print_state()

    simulator.create_request("B")
    simulator.append_tokens("B", count=3)
    simulator.print_state()

    simulator.append_tokens("A", count=4)
    simulator.print_state()

    simulator.finish_request("B")
    simulator.print_state()

    simulator.create_request("C")
    simulator.append_tokens("C", count=5)
    simulator.print_state()

    simulator.finish_request("A")
    simulator.finish_request("C")
    simulator.print_state()


if __name__ == "__main__":
    main()