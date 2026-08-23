from dataclasses import dataclass, field

@dataclass(slots=True)
class Request:
    """Logical state owned by one inference request."""
    request_id: str
    block_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0
    finished: bool = False

    def add_block(self, block_id: int) -> None:
        if self.finished:
            raise RuntimeError(
                f"Request {self.request_id!r} has already finished"
            )
        if block_id in self.block_ids:
            raise RuntimeError(
                f"Block {block_id} is already assigned "
                f"to request {self.request_id!r}"
            )

        self.block_ids.append(block_id)

    def record_token(self) -> None:
        if self.finished:
            raise RuntimeError(f"Request {self.request_id!r} has already finished")
        self.num_tokens += 1

    def mark_finished(self) -> None:
        if self.finished:
            raise RuntimeError(f"Request {self.request_id!r} has already finished")
        self.finished = True