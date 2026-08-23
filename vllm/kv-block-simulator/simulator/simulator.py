from .block_pool import BlockPool
from .request import Request

class KVBlockSimulator:
    """A simplified paged KV-cache simulator."""
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_pool = BlockPool(
            num_blocks=num_blocks,
            block_size=block_size,
        )
        self._requests: dict[str, Request] = {}

    def create_request(self, request_id: str) -> Request:
        """Register a new request without allocating a block yet."""
        if request_id in self._requests:
            raise ValueError(f"Request {request_id!r} already exists")
        request = Request(request_id=request_id)
        self._requests[request_id] = request

        return request

    def append_tokens(self, request_id: str, count: int = 1,) -> None:
        """Append tokens, allocating new blocks when necessary."""
        if count < 0:
            raise ValueError("count must not be negative")

        request = self.get_request(request_id)

        if request.finished:
            raise RuntimeError(
                f"Request {request_id!r} has already finished"
            )

        for _ in range(count):
            block = self._get_or_allocate_last_block(request)
            block.append_token()
            request.record_token()

    def finish_request(self, request_id: str) -> None:
        """Finish a request and release all of its blocks."""
        request = self.get_request(request_id)

        if request.finished:
            raise RuntimeError(
                f"Request {request_id!r} has already finished"
            )

        for block_id in request.block_ids:
            self.block_pool.release(
                block_id,
                expected_request_id=request_id,
            )

        request.mark_finished()

    def remove_finished_request(self, request_id: str) -> None:
        """Remove an already-finished request from the history."""
        request = self.get_request(request_id)
        if not request.finished:
            raise RuntimeError(f"Request {request_id!r} has not finished")
        del self._requests[request_id]

    def get_request(self, request_id: str) -> Request:
        try:
            return self._requests[request_id]
        except KeyError as error:
            raise KeyError(
                f"Unknown request: {request_id!r}"
            ) from error

    def active_requests(self) -> tuple[Request, ...]:
        return tuple(
            request
            for request in self._requests.values()
            if not request.finished
        )

    def state(self) -> dict:
        """Return the current simulator state."""

        active_requests = self.active_requests()

        return {
            "num_blocks": self.block_pool.num_blocks,
            "block_size": self.block_pool.block_size,
            "free_blocks": self.block_pool.free_block_count,
            "allocated_blocks": self.block_pool.allocated_block_count,
            "used_slots": self.block_pool.used_slots,
            "total_slots": self.block_pool.total_slots,
            "physical_utilization": (
                self.block_pool.physical_utilization
            ),
            "allocated_block_utilization": (
                self.block_pool.allocated_block_utilization
            ),
            "active_requests": len(active_requests),
        }

    def print_state(self) -> None:
        """Print blocks and logical-to-physical mappings."""

        state = self.state()

        print("\n=== KV cache state ===")
        print(
            f"Blocks: {state['allocated_blocks']} allocated, "
            f"{state['free_blocks']} free"
        )
        print(
            f"Token slots: {state['used_slots']}/"
            f"{state['total_slots']}"
        )
        print(
            "Physical utilization: "
            f"{state['physical_utilization']:.1%}"
        )
        print(
            "Allocated-block utilization: "
            f"{state['allocated_block_utilization']:.1%}"
        )

        print("\nPhysical blocks:")

        for block in self.block_pool.blocks():
            if block.is_free:
                owner = "free"
            else:
                owner = block.request_id

            print(
                f"  Block {block.block_id:>2}: "
                f"owner={owner!s:<10} "
                f"slots={block.used_slots}/{block.capacity}"
            )

        print("\nRequest block tables:")

        active_requests = self.active_requests()

        if not active_requests:
            print("  No active requests")
            return

        for request in active_requests:
            print(
                f"  {request.request_id}: "
                f"tokens={request.num_tokens}, "
                f"blocks={request.block_ids}"
            )

    def _get_or_allocate_last_block(
        self,
        request: Request,
    ):
        if request.block_ids:
            last_block_id = request.block_ids[-1]
            last_block = self.block_pool.get(last_block_id)

            if not last_block.is_full:
                return last_block

        new_block = self.block_pool.allocate(request.request_id)
        request.add_block(new_block.block_id)

        return new_block
        
        
