# KV Block Simulator
A small Python simulator for learning how vLLM manages paged KV-cache blocks.
The simulator does not run a language model or store real key/value tensors. It focuses only on logical and physical block allocation, token-slot usage, block release, and reuse.

## Concepts
During autoregressive generation, an LLM stores the key and value states of previously processed tokens in a KV cache. Reserving one contiguous region for the maximum possible sequence length can waste memory because the final sequence length is not known in advance.  
Paged KV-cache management divides the cache into fixed-size physical blocks. Requests allocate additional blocks only when needed. A request's physical blocks do not need to be contiguous because a block table maps logical blocks to physical blocks.

For example:

```
Request A block table: [0, 1, 3]
Logical block 0 -> Physical block 0
Logical block 1 -> Physical block 1
Logical block 2 -> Physical block 3
```

## Running the Example
From the project root, run:

```
python -m examples.basic
```

The example uses:

```
Number of physical blocks: 6
Slots per block:           4
Total KV-cache capacity:  24 tokens
```