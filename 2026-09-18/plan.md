# Formal Verification in the context of RUST GPU Programming

*The intuition only. `example.md` has the worked detail.*

---

## The problem

Someone writes a program in PyTorch — think of it as notation for a
mathematical function on arrays:

```python
def forward(self, x):
    return torch.sum(x, dim=1, keepdim=True)
```

It runs, but slowly. To make it fast you rewrite it as a **GPU kernel**: a
function that runs in thousands of concurrent copies, each computing one piece
of the output. That is specialist work, so increasingly an LLM is asked to do
it — *here is the PyTorch program, write me a fast GPU kernel that does the same
thing.*

- **Is it right?** Does it compute the same function as the PyTorch program?

---

## Where are the formal specs?

KernelBench is a benchmark of PyTorch programs — one Python file each. The
Python file encodes specification - the intended behaviour.

---

## ProofWright: arXiv:2511.12294

The authors claim to ship a library called MLRocq, a dictionary that gives ~100
PyTorch operations a formal definition in Rocq. They say they will make this artifact open-source, however it's not available yet. 

You can think of this work as equivalent to building PyTorch -> Verus Spec library. Instead they do it for Rocq.

---

## What we want to do

cuTile Rust kernels are **Rust**, and Verus verifies **Rust**. The specification
language and the implementation language are the same language. So there is
nothing to translate between formalisms — the derived specification attaches
directly to the kernel, and Verus discharges it.

---

## The plan

**PyTorch → specification.** PyTorch -> Verus Spec instead of Natural Lang -> Verus Spec. But we want it to be deterministic like ProofWright does, instead of LLM generated. However, I haven't been able to engineer the process yet. 

**PyTorch → kernel.** The LLM's job, same analogy as NL -> Code task. 

**(specification, kernel) → proof.** Attach the first to the second and let
Verus discharge it. The one thing to know here: cuTile's tile operations have no
Rust bodies — they are compiler primitives — so each one we use gets an assumed
specification instead. That is the trusted base, it is small per kernel, and
Verus will print the list for you.

---

## Modelling now, annotating later

The example I prepared is a hand-written model of the kernel, not an in-place annotated version of the
kernel itself. This is to keep things simple for now. But in-place annotations are possible, just a little more engineering work.

