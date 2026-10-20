//! Inputs the size of a large model for the benchmarks, from a fixed seed so that they are the
//! same on every run: about a million terms, names or values.

use std::ffi::c_int;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, Int32Array, StringArray, StringViewArray};

pub const NNZ: usize = 1_000_000;
pub const TERMS_PER_ROW: usize = 4;
pub const WIDTH: usize = 1_000_000;

/// A fixed-seed generator, the same as the unit tests use, so inputs are identical on every run.
struct Lcg(u64);

impl Lcg {
    fn new() -> Self {
        Self(0x9e37_79b9_7f4a_7c15)
    }

    fn next(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % bound as u64) as usize
    }
}

/// CSR rows of `TERMS_PER_ROW` terms against `WIDTH` columns. With `repeat_every` set, every
/// such row repeats its first variable in its second term, from `first_row` on.
pub fn csr(repeat_every: Option<usize>, first_row: usize) -> (Vec<usize>, Vec<c_int>, Vec<f64>) {
    let rows = NNZ / TERMS_PER_ROW;
    let mut rng = Lcg::new();
    let cbeg: Vec<usize> = (0..rows).map(|r| r * TERMS_PER_ROW).collect();
    let mut cind: Vec<c_int> = (0..NNZ).map(|_| rng.next(WIDTH) as c_int).collect();
    if let Some(every) = repeat_every {
        for row in (first_row..rows).step_by(every) {
            cind[row * TERMS_PER_ROW + 1] = cind[row * TERMS_PER_ROW];
        }
    }
    let cval: Vec<f64> = (0..NNZ).map(|_| rng.next(1000) as f64 / 10.0).collect();
    (cbeg, cind, cval)
}

/// `NNZ` names like `x[O123,F45,W6]`, as one contiguous `Utf8` chunk or one `Utf8View` chunk.
pub fn names(view: bool) -> Vec<ArrayRef> {
    let mut rng = Lcg::new();
    let texts = (0..NNZ).map(|_| {
        format!(
            "x[O{},F{},W{}]",
            rng.next(250_000),
            rng.next(50),
            rng.next(52)
        )
    });
    if view {
        vec![Arc::new(StringViewArray::from_iter_values(texts))]
    } else {
        vec![Arc::new(StringArray::from_iter_values(texts))]
    }
}

/// `NNZ` indices in `chunks` equal pieces.
pub fn indices(chunks: usize) -> Vec<ArrayRef> {
    let mut rng = Lcg::new();
    let all: ArrayRef = Arc::new(Int32Array::from_iter_values(
        (0..NNZ).map(|_| rng.next(WIDTH) as c_int),
    ));
    split(all, chunks)
}

/// `NNZ` coefficients in `chunks` equal pieces.
pub fn f64s(chunks: usize) -> Vec<ArrayRef> {
    let mut rng = Lcg::new();
    let all: ArrayRef = Arc::new(Float64Array::from_iter_values(
        (0..NNZ).map(|_| rng.next(1000) as f64 / 10.0),
    ));
    split(all, chunks)
}

fn split(all: ArrayRef, chunks: usize) -> Vec<ArrayRef> {
    let each = all.len().div_ceil(chunks);
    (0..all.len())
        .step_by(each)
        .map(|start| all.slice(start, each.min(all.len() - start)))
        .collect()
}
