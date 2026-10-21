//! CodSpeed benchmarks of the pure Rust stages of a call, on inputs the size of a large model:
//! about a million terms, names or values, built outside the measured code. The variants cover
//! the paths that differ in cost: rows with and without repeated variables, contiguous and view
//! strings, one chunk and many.
//!
//! The `CodSpeed` workflow builds them with `cargo codspeed build`, which instruments them for
//! CodSpeed's CPU simulation or memory profiling. Locally, `cargo bench --bench stages` times
//! them with plain divan.

use std::ffi::c_int;
use std::sync::Arc;

use _native::columns::{cstrings_of, f64s_of, indices_of};
use _native::model::{merge_repeats, validate};
use arrow_array::{Array, ArrayRef, Float64Array, Int32Array, StringArray, StringViewArray};
use arrow_schema::DataType;
use divan::Bencher;

const NNZ: usize = 1_000_000;
const TERMS_PER_ROW: usize = 4;
const WIDTH: usize = 1_000_000;

fn main() {
    divan::main();
}

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

type Csr = (Vec<usize>, Vec<c_int>, Vec<f64>);

/// CSR rows of `TERMS_PER_ROW` terms against `WIDTH` columns. With `repeat_every` set, every
/// such row repeats its first variable in its second term, from `first_row` on.
fn csr(repeat_every: Option<usize>, first_row: usize) -> Csr {
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
fn names(view: bool) -> Vec<ArrayRef> {
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

/// `all` in `chunks` equal pieces.
fn split(all: ArrayRef, chunks: usize) -> Vec<ArrayRef> {
    let each = all.len().div_ceil(chunks);
    (0..all.len())
        .step_by(each)
        .map(|start| all.slice(start, each.min(all.len() - start)))
        .collect()
}

/// `NNZ` indices in `chunks` equal pieces.
fn indices(chunks: usize) -> Vec<ArrayRef> {
    let mut rng = Lcg::new();
    let all = Int32Array::from_iter_values((0..NNZ).map(|_| rng.next(WIDTH) as c_int));
    split(Arc::new(all), chunks)
}

/// `NNZ` coefficients in `chunks` equal pieces.
fn f64s(chunks: usize) -> Vec<ArrayRef> {
    let mut rng = Lcg::new();
    let all = Float64Array::from_iter_values((0..NNZ).map(|_| rng.next(1000) as f64 / 10.0));
    split(Arc::new(all), chunks)
}

#[divan::bench(args = ["no_repeats", "repeats_throughout", "repeats_at_end_only"])]
fn bench_merge_repeats(bencher: Bencher, case: &str) {
    let input = || match case {
        "no_repeats" => csr(None, 0),
        "repeats_throughout" => csr(Some(10), 0),
        _ => csr(Some(10), NNZ / TERMS_PER_ROW - 1000),
    };
    bencher
        .with_inputs(input)
        .bench_local_refs(|(cbeg, cind, cval)| merge_repeats(cbeg, cind, cval, WIDTH));
}

#[divan::bench]
fn bench_validate(bencher: Bencher) {
    bencher
        .with_inputs(|| csr(None, 0))
        .bench_local_refs(|(cbeg, cind, _)| validate(cbeg, cind, WIDTH).unwrap());
}

#[divan::bench(args = ["utf8", "utf8view"])]
fn bench_cstrings(bencher: Bencher, case: &str) {
    bencher
        .with_inputs(|| names(case == "utf8view"))
        .bench_local_refs(|chunks| cstrings_of("names", chunks, Some(NNZ)).unwrap().ptrs());
}

#[divan::bench(args = [1, 8])]
fn bench_indices(bencher: Bencher, chunks: usize) {
    bencher
        .with_inputs(|| indices(chunks))
        .bench_local_refs(|c| {
            indices_of("vars", &DataType::Int32, c, Some(NNZ))
                .unwrap()
                .len()
        });
}

#[divan::bench(args = [1, 8])]
fn bench_f64s(bencher: Bencher, chunks: usize) {
    bencher.with_inputs(|| f64s(chunks)).bench_local_refs(|c| {
        f64s_of("coefs", &DataType::Float64, c, Some(NNZ))
            .unwrap()
            .len()
    });
}
