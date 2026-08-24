from dataclasses import dataclass, field

@dataclass(slots=True)
class Block:
    """A fixed-size physical KV-cache block."""
    block_id: int
    capacity: int

    token_ids: list[int] = field(default_factory=list)
    block_hash: str | None = None
    request_ids: set[str] = field(default_factory=set)

    @property
    def used_slots(self) -> int:
        return len(self.token_ids)

    @property
    def remaining_slots(self) -> int:
        return self.capacity - self.used_slots

    @property
    def is_full(self) -> bool:
        return self.used_slots == self.capacity

    @property
    def ref_count(self) -> int:
        return len(self.request_ids)

    @property
    def is_referenced(self) -> bool:
        return self.ref_count > 0

    @property
    def is_cached(self) -> bool:
        return self.block_hash is not None

    @property
    def is_free(self) -> bool:
        return self.block_bash is not None

    @property
    def is_free(self) -> bool:
        """
        True only when the block contains no data and has no references.
        A cached block with ref_count == 0 is reclaimable, but it is not
        considered campletely free because it still contains cached data.
        """

        return (
            self.ref_count == 0
            and not self.token_ids
            and self.block_hash is None
        )

    @property
    def is_reclaimable(self) -> bool:
        """
        A block can be reclaimed when no active request references it.
        This includes cached blocks whose ref_count has reached zero.
        """

        return self.ref_count == 0

    @property
    def utilization(self) -> float:
        return self.used_slots / self.capacity

    def assign(self, request_id: str) -> None:
        """Assign an empty block to a new request."""
        if not self.is_free:
            raise RuntimeError(
                f"Block {self.block_id} is not free"
            )
        self.request_ids.add(request_id)

    def acquire_cached(self, request_id: str) -> None:
        """Add a reference to a complete cached block."""
        if not self.is_cached:
            raise RuntimeError(
                f"Block {self.block_id} is not cached"
            )
        if not self.is_full:
            raise RuntimeError(
                f"Cached block {self.block_id} is not full"
            )
        if request_id in self.request_ids:
            raise RuntimeError(
                f"Request {request_id!r} already references "
                f"block {self.block_id}"
            )

    def append_token(self, token_id: int) -> None:
        """Append one token to a block under construction."""
        if self.ref_count == 0:
            raise RuntimeError(
                f"Cannot append a token to unreferenced block "
                f"{self.block_id}"
            )

        if self.ref_count > 1:
            raise RuntimeError(
                f"Cannot modify shared block {self.block_id}"
            )

        if self.is_cached:
            raise RuntimeError(
                f"Cannot modify cached block {self.block_id}"
            )

        if self.is_full:
            raise RuntimeError(
                f"Block {self.block_id} is already full"
            )

        self.token_ids.append(token_id)

    def set_block_hash(self, block_hash: str) -> None:
        """Mark a complete block as prefix-cacheable."""

        if not self.is_full:
            raise RuntimeError(
                f"Cannot cache incomplete block {self.block_id}"
            )

        if self.block_hash is not None:
            raise RuntimeError(
                f"Block {self.block_id} is already cached"
            )

        if not block_hash:
            raise ValueError("block_hash must not be empty")

        self.block_hash = block_hash

    def release_reference(self, request_id: str) -> int:
        """
        Release one request reference.

        Returns the remaining reference count.
        """

        if request_id not in self.request_ids:
            raise RuntimeError(
                f"Request {request_id!r} does not reference "
                f"block {self.block_id}"
            )

        self.request_ids.remove(request_id)
        return self.ref_count

    def reset(self) -> None:
        """Remove cached data and return the block to an empty state."""

        if self.ref_count != 0:
            raise RuntimeError(
                f"Cannot reset referenced block {self.block_id}: "
                f"ref_count={self.ref_count}"
            )

        self.token_ids.clear()
        self.block_hash = None
        self.request_ids.clear()