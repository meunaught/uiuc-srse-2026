# A walkthrough: what the lynette gate actually does to ProofGen results

I use commit `fce8da7` because we need the dataset snapshot at the time of ProofGen
evaluation. New PRs have modified the dataset partially.

---

## 1/ How the pipeline works

Input is `(code_spec.rs, prompt.md)`, output is a candidate proof, which then passes through
two checks in sequence.

```
code_spec.rs + spec_code2proof.md  ->  candidate_proof.rs  ->  lynette  ->  verus
```

The prompt template is filled with `code_spec.rs` verbatim — `{{verus_program}}` in
`run_gpt55.py:81-84`. Scoring is `test_spec_code2proof.py:108`:

```python
origin = (problem_dir / "code_spec.rs").read_text()
safe   = code_change_is_safe(origin, candidate, ..., target_mode=True)   # lynette compare -t
if safe is not True:
    return {"status": "unsafe_code_or_spec_change"}                      # never reaches verus
res = run_verus_verify(candidate, verify_timeout)                        # only if safe
```

`code_spec.rs` therefore serves twice over: it is what the model is shown, and what its output
is measured against. A lynette rejection means Verus never runs at all.

## 2/ Let's check gpt5.5's result for lc1688

What goes to the LLM is `benchmark/leetcode/lc1688/code_spec.rs` plus the prompt. The recorded
response lands in

```
spec_code2proof/result/gpt55/s0/lc1688.md
```

and `extract_code` pulls the ```` ```rust ```` fence out of it, giving
`lc1688_gpt55_candidate.rs`.

## 3/ What is the next step?

```
lynette compare -t  benchmark/leetcode/lc1688/code_spec.rs  lc1688_gpt55_candidate.rs
Files are different
exit 1
```

Immediately rejected. Recorded status: `unsafe_code_or_spec_change`.

## 4/ But does `lc1688_gpt55_candidate.rs` verify?

```
verus --no-cheating --rlimit 100000  lc1688_gpt55_candidate.rs
verification results:: 3 verified, 0 errors
```

It does. And `--no-cheating` means no `assume`, no `admit`, no `external_body` — it is a real
proof.

## 5/ Should we accept it, then? Let's read it by hand

First, just diff the two files directly — 32 lines in, 101 lines out:

```diff
  use vstd::prelude::*;
+ use vstd::arithmetic::div_mod::*;

  ...

+ pub proof fn lemma_matches_spec(n: int)
+     requires 1 <= n,
+     ensures  Self::matches_spec(n) == n - 1,
+     decreases n,
+ {
+     ...  60 lines of assert / lemma_fundamental_div_mod calls  ...
+ }

  pub fn number_of_matches(n: i32) -> (result: i32)
      requires 1 <= n <= 200,
      ensures result == Self::matches_spec(n as int),
  {
+     proof {
+         Self::lemma_matches_spec(n as int);
+     }
      n - 1
  }
```

Everything except the import seems to be proof blocks — one new `proof fn`, and one
`proof { }` that calls it. The executable body is still `n - 1`, untouched.

But *seems* is not good enough. Let's confirm with what lynette actually compares.

**Deghosting** is the step lynette runs before comparing: it walks the AST and strips
everything that is proof — `proof` blocks, `assert`s, loop `invariant`/`decreases`, `proof fn`s,
`let ghost` — leaving the executable program behind. With `-t` the contracts
(`requires`/`ensures`/`decreases`) are kept and compared too. If two files deghost to the same
thing, nothing but proof was added.

`lynette compare -t -v` prints both files after deghosting — literally what it compares — one
after the other, separated by the `} // verus!` line. Split them at the first separator and
diff:

```bash
lynette compare -t -v code_spec.rs lc1688_gpt55_candidate.rs \
  | grep -v '^Files are different$' \
  | awk 'f {print > "/tmp/candidate.dg"; next}
         {print > "/tmp/original.dg"}
         /^} \/\/ verus!$/ {f=1}'

diff -u -B /tmp/original.dg /tmp/candidate.dg
```

```diff
  use vstd::prelude::*;
+ use vstd::arithmetic::div_mod::*;
```

That is all of it. One line.

Apart from proof blocks, the sole change is an **import** — the added proof calls a vstd
lemma, so the module has to be in scope. Not one spec clause moved, not one executable
statement. We should indeed accept this one.

It means lynette is too strict here.

## 6/ And curiously — we apply this standard, where we reject proofs if lynette rejects them, to a ground-truth dataset?

Every problem ships a `verified.rs`: the benchmark's own reference proof, a known-correct
answer to the exact task. Which means we can point the identical check at it. Run it on
`cf1200C/{code_spec.rs, verified.rs}`:

```
lynette compare -t code_spec.rs verified.rs
Files are different
exit 1
```

Striking — identical outcome. And the deghosted diff looks familiar:

```diff
  use vstd::prelude::*;
+ use vstd::arithmetic::div_mod::*;
```

One import line — this time in the benchmark's own answer.

Hypothetically: if a very good model had produced exactly the artifact in `verified.rs`, the
pipeline would falsely reject it.

Is the ground truth wrong, then? No. What we are watching is lynette applying too strict a
rule. **The ground truth is fine, and so is `lc1688_gpt55_candidate.rs`.**

### What does it mean for our ground truth, then?

Three things we have been taking for granted:

- `verified.rs` is designed not to tamper with `code_spec.rs` — is that actually so?
- `code_spec.rs` is consistent — untampered — with `(spec.rs, code.rs)` — is that so?
- And if lynette does get it right in some cases, or if we assume a perfect anti-tamper
  filter, then we should be hunting for a *real* tampering bug in `verified.rs` against
  `code_spec.rs`. See step 8.

## 7/ At this point you may be thinking — these must be edge cases for lynette?

Turns out this is not the case. Try `cf1538B/{code_spec.rs, verified.rs}`.

The gold proof leaves the executable content of the `if` untouched and adds proof around it —
including an `else` that holds nothing but a proof block:

```rust
if a[i] > t {
    proof { assert((cnt + 1) as int <= n as int); }
    cnt = cnt + 1;
} else {
    proof { assert(cnt as int == count_gt_prefix(a@, t as int, (i + 1) as int)); }
}
```

And here is the deghosted diff — nothing added, three lines *removed*:

```diff
          while i < n {
-             if a[i] > t {
-                 cnt = cnt + 1;
-             }
              i = i + 1;
          }
```

The whole `if` statement has vanished from lynette's view of the file — `cnt = cnt + 1`
included. An all-ghost `else` deghosts to `None`, and `None.map(...)` is `None`.

This is not an edge case. It is a whole class.

## 8/ The gate is clearly not doing what we want — but what about when lynette is right?

Now, should you ask, why do we bother at all? We could drop it. But what about the cases where
lynette gets it right?

Let us take `benchmark/leetcode/lc1095/code_spec.rs` against
`spec_code2proof/result/gpt55/s0/lc1095.md` — again, fence stripped to
`lc1095_gpt55_candidate.rs`. By simply putting the files side by side, what do we see?

```diff
      let n = mountain_arr.length();
-     let mut left: i32 = 0;
-     let mut right: i32 = n - 1;
-     while left < right {                       // phase 1: binary search for the peak
-         let mid = left + (right - left) / 2;
-         if mountain_arr.get(mid) < mountain_arr.get(mid + 1) { left = mid + 1; }
-         else { right = mid; }
-     }
-     let peak = left;
-     ...                                        // phase 2: binary search ascending side
-     ...                                        // phase 3: binary search descending side
+     let mut i: i32 = 0;
+     while i < n {
+         if mountain_arr.get(i) == target { return i; }
+         i = i + 1;
+     }
      -1
```

A clear code-level tampering attempt by gpt55 — binary search (`code__given`) swapped out for
linear search (`code__tampered`). Perfectly tampered so that `verus --no-cheating` verifies!

Note the shape of the diff: real executable code on *both* sides. That is what separates a
genuine edit from the erasure artifact in step 7, where lines only vanish and nothing takes
their place.

Here the benchmark's own `verified.rs` **passes** — the gate was satisfiable on this problem,
so the rejection is the candidate's own doing.

And if gpt55 can do such tampering in pass@1, what would stop an agentic-loop based approach
from doing this more aggressively?

## 9/ Now we can agree

A better anti-tamper gate is required to truly benchmark ProofGen capabilities of LLMs.

With all that said, it feels like — umm — just like Mijhalo found pre/post-condition bugs,
what about the probability of finding tampering bugs in the ground-truth dataset itself?
