// Verus artifact for KernelBench L1 task 47. See README.md for the walkthrough.
//
// Difference from v6: the row length `cols` is NOT assumed to be a multiple of
// the tile width `b`. The final tile runs off the end of the row and cuTile
// zero-fills it (`padding::Zero`). Task 47's head shape is (128, 4096, 4095),
// and 4095 is odd on purpose, so the tail is the case that actually occurs.

use vstd::prelude::*;

// The GPU operations. Bodies belong to the cuTile compiler, not to us.
#[verifier::external]
mod cutile_core {
    pub fn load_tile(_x: &Vec<i32>, _r: usize, _kt: usize, _b: usize, _cols: usize) -> Vec<i32> {
        unreachable!()
    }
    pub fn reduce_sum(_t: &Vec<i32>) -> i32 { unreachable!() }
    pub fn store_scalar(_o: &mut Vec<i32>, _r: usize, _v: i32) { unreachable!() }
}

verus! {

// ===========================================================================
// 1. Meanings
// ===========================================================================

pub open spec fn fold_sum(s: Seq<i32>) -> int
    decreases s.len()
{
    if s.len() == 0 { 0int } else { s[0] as int + fold_sum(s.skip(1)) }
}

pub open spec fn min_int(a: int, b: int) -> int { if a < b { a } else { b } }

/// Row `r` of a rows x cols tensor held in row-major order.
pub open spec fn row(x: Seq<i32>, r: int, cols: int) -> Seq<i32> {
    x.subrange(r * cols, r * cols + cols)
}

/// Element `j` of tile `kt`. Past the end of the row it reads 0 -- this is
/// cuTile's `padding::Zero`, and it is what makes the tail tile well defined.
pub open spec fn tile_elem(rw: Seq<i32>, kt: int, b: int, j: int) -> i32 {
    if 0 <= kt * b + j < rw.len() { rw[kt * b + j] } else { 0 }
}

pub open spec fn tile(rw: Seq<i32>, kt: int, b: int) -> Seq<i32> {
    Seq::new(b as nat, |j: int| tile_elem(rw, kt, b, j))
}

/// How much of the row tiles 0..kt cover. Caps at `cols` because the last one
/// overhangs.
pub open spec fn covered(kt: int, b: int, cols: int) -> int {
    min_int(kt * b, cols)
}

/// The denotation of `torch.sum(x, dim=1, keepdim=True)` -- the specification.
pub open spec fn torch_sum_dim1(x: Seq<i32>, rows: int, cols: int) -> Seq<int> {
    Seq::new(rows as nat, |r: int| fold_sum(row(x, r, cols)))
}

// ===========================================================================
// 2. The GPU operations, as axioms
// ===========================================================================

pub assume_specification [cutile_core::load_tile]
    (x: &Vec<i32>, r: usize, kt: usize, b: usize, cols: usize) -> (t: Vec<i32>)
    requires
        0 <= r * cols,
        r * cols + cols <= x.len(),
    ensures
        t.len() == b,
        t@ =~= tile(row(x@, r as int, cols as int), kt as int, b as int),
;

pub assume_specification [cutile_core::reduce_sum] (t: &Vec<i32>) -> (s: i32)
    requires i32::MIN <= fold_sum(t@) <= i32::MAX,
    ensures s as int == fold_sum(t@),
;

pub assume_specification [cutile_core::store_scalar] (o: &mut Vec<i32>, r: usize, v: i32)
    requires r < old(o).len(),
    ensures
        final(o).len() == old(o).len(),
        final(o)[r as int] == v,
        forall|k: int| 0 <= k < final(o).len() && k != r ==> final(o)[k] == old(o)[k],
;

// ===========================================================================
// 3. Lemmas
// ===========================================================================

pub proof fn lemma_fold_concat(s1: Seq<i32>, s2: Seq<i32>)
    ensures fold_sum(s1 + s2) == fold_sum(s1) + fold_sum(s2),
    decreases s1.len(),
{
    if s1.len() == 0 {
        assert(s1 + s2 =~= s2);
    } else {
        assert((s1 + s2).skip(1) =~= s1.skip(1) + s2);
        lemma_fold_concat(s1.skip(1), s2);
    }
}

pub proof fn lemma_fold_split(s: Seq<i32>, i: int, j: int)
    requires 0 <= i <= j <= s.len(),
    ensures fold_sum(s.take(j)) == fold_sum(s.take(i)) + fold_sum(s.subrange(i, j)),
{
    assert(s.take(j) =~= s.take(i) + s.subrange(i, j));
    lemma_fold_concat(s.take(i), s.subrange(i, j));
}

/// Padding contributes nothing. Without this the tail tile would be opaque.
pub proof fn lemma_fold_zeros(s: Seq<i32>)
    requires forall|i: int| 0 <= i < s.len() ==> s[i] == 0,
    ensures fold_sum(s) == 0,
    decreases s.len(),
{
    if s.len() == 0 {
    } else {
        assert forall|i: int| 0 <= i < s.skip(1).len() implies s.skip(1)[i] == 0 by {
            assert(s.skip(1)[i] == s[i + 1]);
        }
        lemma_fold_zeros(s.skip(1));
    }
}

/// A tile's fold equals the fold of the row slice it covers -- overhang and
/// all. This is the tail case discharged.
pub proof fn lemma_fold_tile(rw: Seq<i32>, kt: int, b: int, cols: int)
    requires
        b > 0,
        0 <= kt * b,
        rw.len() == cols,
    ensures
        fold_sum(tile(rw, kt, b))
            == fold_sum(rw.subrange(covered(kt, b, cols), covered(kt + 1, b, cols))),
{
    let lo = covered(kt, b, cols);
    let hi = covered(kt + 1, b, cols);
    assert((kt + 1) * b == kt * b + b) by (nonlinear_arith);

    let head = rw.subrange(lo, hi);
    let pad = Seq::new((b - (hi - lo)) as nat, |_j: int| 0i32);

    assert(tile(rw, kt, b) =~= head + pad) by {
        assert forall|j: int| 0 <= j < b implies
            tile(rw, kt, b)[j] == (head + pad)[j] by {
            if j < hi - lo {
                assert(head[j] == rw[lo + j]);
            }
        }
    }
    lemma_fold_concat(head, pad);
    lemma_fold_zeros(pad);
}

// ===========================================================================
// 4. One tile program: one output row
// ===========================================================================

fn row_sum_kernel(
    o: &mut Vec<i32>, x: &Vec<i32>,
    r: usize, b: usize, ntiles: usize, cols: usize,
)
    requires
        b > 0,
        r < old(o).len(),
        cols <= ntiles * b,
        0 <= r * cols,
        r * cols + cols <= x.len(),
        forall|k: int| 0 <= k <= cols ==>
            i32::MIN <= #[trigger] fold_sum(row(x@, r as int, cols as int).take(k)) <= i32::MAX,
        forall|kt: int| 0 <= kt < ntiles ==>
            i32::MIN <= #[trigger] fold_sum(tile(row(x@, r as int, cols as int), kt, b as int))
                     <= i32::MAX,
    ensures
        final(o).len() == old(o).len(),
        final(o)[r as int] as int == fold_sum(row(x@, r as int, cols as int)),
        forall|k: int| 0 <= k < final(o).len() && k != r ==> final(o)[k] == old(o)[k],
{
    let ghost rw = row(x@, r as int, cols as int);
    let mut acc: i32 = 0;
    let mut kt: usize = 0;

    assert(rw.len() == cols);
    assert(covered(0int, b as int, cols as int) == 0) by (nonlinear_arith);
    assert(rw.take(0int) =~= Seq::<i32>::empty());

    while kt < ntiles
        invariant
            b > 0,
            kt <= ntiles,
            cols <= ntiles * b,
            rw == row(x@, r as int, cols as int),
            rw.len() == cols,
            0 <= r * cols,
            r * cols + cols <= x.len(),
            0 <= covered(kt as int, b as int, cols as int) <= cols,
            acc as int == fold_sum(rw.take(covered(kt as int, b as int, cols as int))),
            forall|k: int| 0 <= k <= cols ==>
                i32::MIN <= #[trigger] fold_sum(rw.take(k)) <= i32::MAX,
            forall|k: int| 0 <= k < ntiles ==>
                i32::MIN <= #[trigger] fold_sum(tile(rw, k, b as int)) <= i32::MAX,
        decreases ntiles - kt,
    {
        proof {
            assert(0 <= kt * b) by (nonlinear_arith) requires kt >= 0, b > 0;
            assert((kt + 1) * b == kt * b + b) by (nonlinear_arith);
            // the tile covers exactly the next stretch of the row ...
            lemma_fold_tile(rw, kt as int, b as int, cols as int);
            // ... and that stretch is what extends the prefix.
            lemma_fold_split(
                rw,
                covered(kt as int, b as int, cols as int),
                covered(kt as int + 1, b as int, cols as int),
            );
        }

        let t = cutile_core::load_tile(x, r, kt, b, cols);
        let s = cutile_core::reduce_sum(&t);
        acc = acc + s;
        kt = kt + 1;
    }

    proof {
        assert(covered(ntiles as int, b as int, cols as int) == cols);
        assert(rw.take(cols as int) =~= rw);
    }
    cutile_core::store_scalar(o, r, acc);
}

// ===========================================================================
// 5. The grid, against the PyTorch denotation
// ===========================================================================

fn sum_grid(
    o: &mut Vec<i32>, x: &Vec<i32>,
    rows: usize, cols: usize, b: usize, ntiles: usize,
)
    requires
        b > 0,
        cols <= ntiles * b,
        old(o).len() == rows,
        x.len() == rows * cols,
        forall|r: int, k: int| 0 <= r < rows && 0 <= k <= cols ==>
            i32::MIN <= #[trigger] fold_sum(row(x@, r, cols as int).take(k)) <= i32::MAX,
        forall|r: int, kt: int| 0 <= r < rows && 0 <= kt < ntiles ==>
            i32::MIN <= #[trigger] fold_sum(tile(row(x@, r, cols as int), kt, b as int))
                     <= i32::MAX,
    ensures
        final(o).len() == rows,
        final(o)@.len() == torch_sum_dim1(x@, rows as int, cols as int).len(),
        forall|r: int| 0 <= r < rows ==>
            #[trigger] final(o)[r] as int == torch_sum_dim1(x@, rows as int, cols as int)[r],
{
    let mut r: usize = 0;
    while r < rows
        invariant
            b > 0,
            r <= rows,
            cols <= ntiles * b,
            o.len() == rows,
            x.len() == rows * cols,
            forall|rr: int, k: int| 0 <= rr < rows && 0 <= k <= cols ==>
                i32::MIN <= #[trigger] fold_sum(row(x@, rr, cols as int).take(k)) <= i32::MAX,
            forall|rr: int, kt: int| 0 <= rr < rows && 0 <= kt < ntiles ==>
                i32::MIN <= #[trigger] fold_sum(tile(row(x@, rr, cols as int), kt, b as int))
                         <= i32::MAX,
            forall|rr: int| 0 <= rr < r ==>
                #[trigger] o[rr] as int == fold_sum(row(x@, rr, cols as int)),
        decreases rows - r,
    {
        proof {
            assert(r * cols + cols <= rows * cols) by (nonlinear_arith)
                requires r < rows, cols >= 0;
            assert(0 <= r * cols) by (nonlinear_arith) requires r >= 0, cols >= 0;
        }
        row_sum_kernel(o, x, r, b, ntiles, cols);
        r = r + 1;
    }

    assert forall|rr: int| 0 <= rr < rows implies
        #[trigger] o[rr] as int == torch_sum_dim1(x@, rows as int, cols as int)[rr] by {
        assert(torch_sum_dim1(x@, rows as int, cols as int)[rr]
               == fold_sum(row(x@, rr, cols as int)));
    }
}

// The benchmark's own shape, checked admissible: 4095 columns, 256-wide tiles,
// 16 tiles. 16*256 = 4096, so the last tile overhangs by one element -- the
// tail case, which is why `cols` is not assumed to divide.
pub proof fn task47_shape_is_admissible() {
    assert(4095int <= 16int * 256int);
    assert(covered(16int, 256int, 4095int) == 4095int);
    assert(covered(15int, 256int, 4095int) == 3840int);
}

fn main() {}

} // verus!
