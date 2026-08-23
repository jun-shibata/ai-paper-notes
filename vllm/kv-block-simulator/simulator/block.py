from dataclasses import dataclass

@dataclass(slots=True)
class Block:
    """A fixed-size physical KV-cache block."""
    block_id: int
    capacity: int
    request_id: str | None = None
    used_slots: int = 0

    @property
    def is_free(self) -> bool:
        return self.request_id is None

    @property
    def is_full(self) -> bool:
        return self.used_slots == self.capacity

    @property
    def remaining_slots(self) -> int:
        return self.capacity - self.used_slots

    @property
    def utilization(self) -> float:
        if self.is_free:
            return 0.0
        return self.used_slots / self.capacity

    def assign(self, request_id: str) -> None:
        """Assign this block to a request."""
        if not self.is_free:
            raise RuntimeError(
                f"Block {self.block_id} is already assigned "
                f"to request {self.request_id!r}"
            )
        self.request_id = request_id
        self.used_slots = 0

    def append_token(self) -> None:
        """Consume one slot in this block."""
        if self.is_free:
            raise RuntimeError(
                f"Cannot append a token to free block {self.block_id}"
            )
        if self.is_full:
            raise RuntimeError(
                f"Block {self.block_id} is already full"
            )
        self.used_slots += 1

    def reset(self) -> None:
        """Release all state and make this block free."""
        self.request_id = None
        self.used_slots = 0