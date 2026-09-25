/*
 * The cuTile Rust kernel for KernelBench L1 task 47. This is the artifact an
 * LLM produces from the PyTorch source; `proof.rs` is what Verus checks.
 *
 * IT DOES NOT BUILD ON THIS MACHINE and has never been compiled or run.
 * cuTile requires Linux, CUDA 13.3 and an sm_80+ GPU. It is written against
 * the idiom in repo/cutile-examples/examples/rms_norm.rs, which is the
 * closest upstream example -- a tiled accumulation loop over a partitioned
 * input, closed with `reduce_sum`.
 */
use cutile::prelude::*;

#[cutile::module]
mod sum_module {
    use cutile::core::*;

    /// out[row] = sum over the row's COLS elements.
    ///
    /// The launch grid is (ROWS, 1, 1): one tile program per output row. The
    /// output is partitioned one element per program, so `out` arrives as an
    /// exclusive `&mut Tensor` over just this program's cell -- which is why
    /// there is no output index anywhere in the body.
    #[cutile::entry()]
    fn sum_rows<const COLS: i32, const BLOCK: i32>(
        out: &mut Tensor<i32, { [1] }>,
        x: &Tensor<i32, { [-1, COLS] }>,
    ) {
        let tile_shape: Shape<{ [1, BLOCK] }> = shape![1, BLOCK];
        // Ceiling division: the last tile overhangs the row end when BLOCK
        // does not divide COLS. cuTile zero-fills the overhang.
        let num_tiles: i32 = (COLS + BLOCK - 1) / BLOCK;

        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let x_part: Partition<i32, { [1, BLOCK] }> = x.partition(tile_shape);

        // The tiled accumulation. This is the line that makes the kernel a
        // REASSOCIATION of the PyTorch sum rather than a transcription of it.
        let mut acc: i32 = 0;
        for j in 0i32..num_tiles {
            let tx: Tile<i32, { [1, BLOCK] }> = x_part.load([row, j]);
            let s: Tile<i32, { [1] }> = reduce_sum(tx, 1i32);
            let s: Tile<i32, { [] }> = s.reshape(shape![]);
            acc = acc + tile_to_scalar(s);
        }

        out.store(scalar_to_tile(acc).reshape(shape![1]));
    }
}

use sum_module::sum_rows;

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    // The benchmark's shape, in its 2-D instance. 4095 is not a multiple of
    // 256, so the tail tile is exercised.
    let (rows, cols) = (128usize, 4095usize);

    let x: Arc<Tensor<i32>> = api::ones::<i32>(&[rows * cols])
        .sync_on(&stream)?
        .reshape(&[rows, cols])
        .unwrap()
        .into();
    let out = api::zeros::<i32>(&[rows]).partition([1]);

    let (out, _x) = sum_rows(out, x).generics((cols as i32, 256i32)).sync_on(&stream)?;
    let host: Vec<i32> = out.unpartition().to_host_vec(&stream)?;

    // With x all ones, every row sums to cols.
    for (r, &v) in host.iter().enumerate() {
        assert_eq!(v, cols as i32, "row {}: {} != {}", r, v, cols);
    }
    println!("sum_rows: all rows correct");
    Ok(())
}
