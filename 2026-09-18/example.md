# Worked example: KernelBench task 47

Two questions only: **how we get a specification out of the PyTorch source**,
and **how the specification and the kernel become a proof**. Files: `kernel.rs`,
`proof.rs` (17 verified, 0 errors).

The task, in full:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x, dim=self.dim, keepdim=True)

batch_size, dim1, dim2 = 128, 4096, 4095
reduce_dim = 1
```

Worked here in its 2-D instance: `x` is `(rows, cols)`, output cell `r` is the
sum of row `r`. `cols = 4095` is kept, and the tile width is 256, so the last
tile of every row runs off the end.

---

# Part 1 — PyTorch → specification

> **These steps are done by hand here, and that is not the intent.** The
> specification is eventually meant to be derived *mechanically* from the
> PyTorch source — the way ProofWright traces the program to a graph and maps
> each operation through MLRocq to get a Rocq specification, but landing in
> Verus instead. Deriving it by hand first is how you find out what the
> mechanical version has to produce: which decisions it must make, and where
> it can go wrong. It should not end up being an LLM, for the reason in
> `plan.md`.

### Step 0. The Python file *is* the specification

Nothing else defines correctness — no requirements document, no reference
kernel. So the job is **denotation**: what mathematical function does this
program denote?

### Step 1. Strip the framework

`torch.sum(x, dim=1, keepdim=True)` says one thing:

> for each output cell, sum the elements along axis 1 that map to it

`nn.Module`, `__init__`, `keepdim` do not affect values. What remains is *which
elements feed each output cell, and what is done with them*.

### Step 2. Fix the element type

Elements are `i32`; sums are computed in `int`, Verus's mathematical integer.

If the sum returned `i32`, overflow would wrap silently *inside the
specification*, and the spec would quietly become "the sum, mod 2³²". In `int`
it says *the sum*, and overflow resurfaces later as an explicit obligation.

### Step 3. Decide the representation

Verus has no tensor type, so a tensor is **a flat `Seq<i32>` plus its shape
numbers**, row-major.

Load-bearing: this has to match the layout the kernel assumes. Get it wrong and
you prove a true theorem about the wrong program.

### Step 4. Name the elements that feed one output cell

The step that does the real work — it turns an *indexing* question into a
*sequence* question.

```rust
pub open spec fn row(x: Seq<i32>, r: int, cols: int) -> Seq<i32> {
    x.subrange(r * cols, r * cols + cols)
}
```

Everything downstream talks about sequences, never indices.

This is also the **only** place the 2-D/3-D difference lives. The benchmark's
real tensor is 3-D and `dim=1` is the middle axis, so the summands are strided,
not contiguous — `row` becomes a strided gather and nothing else in the
development changes.

### Step 5. Say what "sum" means

```rust
pub open spec fn fold_sum(s: Seq<i32>) -> int
    decreases s.len()
{
    if s.len() == 0 { 0int } else { s[0] as int + fold_sum(s.skip(1)) }
}
```

A structural left fold: the standard denotation, defined by recursion on the
sequence so induction works directly, with `decreases` making it total.

### Step 6. Compose

```rust
pub open spec fn torch_sum_dim1(x: Seq<i32>, rows: int, cols: int) -> Seq<int> {
    Seq::new(rows as nat, |r: int| fold_sum(row(x, r, cols)))
}
```

Done. Steps 4–6 are just *which elements* → *what operation* → *for every
output cell*.

### Step 7. Audit it

The test: **does it mention anything from the implementation?** Scan for tiles,
block widths, program ids, accumulation order. It has none — only rows, columns
and addition.

This matters because the whole exercise is worthless if the specification is
read off the kernel: you would prove the kernel agrees with itself.

---

# Part 2 — (specification, kernel) → proof

### What the kernel does that the specification does not

```rust
let mut acc: i32 = 0;
for j in 0i32..num_tiles {
    let tx = x_part.load([row, j]);      // 256 elements
    acc = acc + tile_to_scalar(reduce_sum(tx, 1i32).reshape(shape![]));
}
out.store(scalar_to_tile(acc).reshape(shape![1]));
```

`fold_sum` is one flat left fold over the whole row. The kernel folds each
256-element tile and then sums the 16 partials. **Different bracketings of the
same additions** — so the proof obligation is a *reassociation*. That is the
mathematical content of the exercise, and the reason the element type had to be
`int`: over floats, addition is not associative and the two are genuinely not
equal.

Two other facts about the kernel matter:

- **`reduce_sum` has no Rust body.** It is one of 163 cuTile operations declared
  as `unreachable!()` and filled in by the compiler. There is nothing to verify.
- **There is no output index in the body.** `out` is this instance's own
  exclusive slice, so a kernel copy cannot address another's cell.

### Layer 1 — meanings (definitions; nothing trusted)

`fold_sum`, `row`, `torch_sum_dim1` from Part 1, plus two describing how the
kernel chops the work up:

```rust
pub open spec fn tile_elem(rw: Seq<i32>, kt: int, b: int, j: int) -> i32 {
    if 0 <= kt * b + j < rw.len() { rw[kt * b + j] } else { 0 }   // zero padding
}
pub open spec fn covered(kt: int, b: int, cols: int) -> int {
    min_int(kt * b, cols)                    // how far tiles 0..kt reach
}
```

These are a **different species** from `torch_sum_dim1`:

| from PyTorch | from cuTile |
|---|---|
| `fold_sum`, `row`, `torch_sum_dim1` | `tile_elem`, `tile`, `covered` |
| *what must be computed* | *how it is being computed* |

The cuTile-side ones appear only in invariants and intermediate lemmas. The
top-level theorem mentions only the PyTorch side — implementation vocabulary is
scaffolding you climb and then kick away.

### Layer 2 — the operations, as axioms (trusted)

No bodies, so they get `ensures` clauses instead, attached from outside with
`assume_specification` (the cuTile crate is a source and is never edited):

```rust
pub assume_specification [cutile_core::reduce_sum] (t: &Vec<i32>) -> (s: i32)
    requires i32::MIN <= fold_sum(t@) <= i32::MAX,
    ensures  s as int == fold_sum(t@),
;
```

This is the trust boundary and it is machine-enumerable — `--no-cheating`
rejects assumed specifications, so running it prints exactly what is assumed:

```
proof.rs:64   load_tile
proof.rs:74   reduce_sum
proof.rs:79   store_scalar
```

Three entries. Everything else is proved.

### Layer 3 — lemmas (proved from layer 1)

```
lemma_fold_concat   fold_sum(s1 + s2) == fold_sum(s1) + fold_sum(s2)
lemma_fold_split    fold_sum(s.take(j)) == fold_sum(s.take(i)) + fold_sum(s.subrange(i,j))
lemma_fold_zeros    an all-zero sequence folds to 0
lemma_fold_tile     a tile's fold == the fold of the row slice it covers, overhang included
```

`lemma_fold_split` **is** the reassociation. `lemma_fold_zeros` is what makes
the overhanging tail tile harmless: padded elements contribute nothing, so a
tile running off the end still folds to exactly the slice it covers.

### The theorem

The per-row kernel maintains

```rust
acc as int == fold_sum(rw.take(covered(kt as int, b as int, cols as int)))
```

— *the accumulator equals the fold of everything the first `kt` tiles cover*.
Each step advances it by `lemma_fold_tile` then `lemma_fold_split`. On exit
`covered(ntiles, ..) == cols`, so `acc == fold_sum(row)`, and the grid gives

```rust
ensures
    forall|r: int| 0 <= r < rows ==>
        final(o)[r] as int == torch_sum_dim1(x@, rows as int, cols as int)[r],
```

the PyTorch denotation itself, not a restatement of it.


---

## Caveats and running it

`proof.rs` is a hand-written model of `kernel.rs`, not the kernel with
annotations; the correspondence is by inspection.The three assumed specifications
are unchecked against what the compiler emits. 

```
tools/verus-0.2026.09.13.671956e/verus \
    --rlimit 100000 --triggers-mode silent \
    projects/cutile-rust/draft/v7/proof.rs
```

    17 verified, 0 errors        0.44s

