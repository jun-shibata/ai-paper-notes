from collections import deque
from .block import Block

class OutOfBlocksError(RuntimeError):
    """Raised when no free KV-cache block is available."""

class BlockPool:
    """A pool of fixed-size physical KV-cache blocks."""
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero")

        if block_size <= 0:
            raise ValueError("block_size must be greater than zero")

        self.num_blocks = num_blocks
        self.block_size = block_size

        self._blocks = [
            Block(
                block_id=block_id,
                capacity=block_size,
            ) for block_id in range(num_blocks)
        ]

        self._free_block_ids = deque(range(num_blocks))

    @property
    def free_block_count(self) -> int:
        return len(self._free_block_ids)

    @property
    def allocated_block_count(self) -> int:
        return self.num_blocks - self.free_block_count

    @property
    def total_slots(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def used_slots(self) -> int:
        return sum(block.used_slots for block in self._blocks)

    @property
    def allocated_slots(self) -> int:
        return self.allocated_block_count * self.block_size

    @property
    def physical_utilization(self) -> float:
        """Utilization across the entire KV-cache capacity."""
        if self.total_slots == 0:
            return 0.0
        return self.used_slots / self.total_slots

    @property
    def allocated_block_utilization(self) -> float:
        """Utilization within blocks currently assigned to requests."""
        if self.allocated_slots == 0:
            return 0.0

        return self.used_slots / self.allocated_slots

    def allocate(self, request_id: str) -> Block:
        """Allocate one free block to a request."""
        if not self._free_block_ids:
            raise OutOfBlocksError(
                f"No free KV blocks are available for request {request_id!r}"
            )
        block_id = self._free_block_ids.popleft()
        block = self._blocks[block_id]
        block.assign(request_id)

        return block

    def release(self, block_id: int, expected_request_id: str | None = None,) -> None:
        """Release one block back to the free list."""
        block = self.get(block_id)
        if block.is_free:
            raise RuntimeError(
                f"Block {block_id} is already free"
            )

        if (expected_request_id is not None and block.request_id != expected_request_id):
            raise RuntimeError(
                f"Block {block_id} belongs to request "
                f"{block.request_id!r}, not {expected_request_id}"
            )

        block.reset()
        self._free_block_ids.append(block_id)

    def get(self, block_id: int) -> Block:
        """Return a block by its physical ID"""
        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError(f"Invalid block ID: {block_id}")

        return self._blocks[block_id]

    def blocks(self) -> tuple[Block, ...]:
        """Return a read-only view of all blocks."""
        return tuple(self._blocks)

