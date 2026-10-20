//! Instruction and allocation counts of the pure Rust stages of a call, measured under Valgrind
//! with iai-callgrind, so that they are reproducible on any machine rather than subject to load.
//!
//! Each benchmark takes inputs the size of a large model, built outside the measured function:
//! about a million terms, names or values. The instruction counts cover the measured function
//! only; DHAT's allocation totals cover the whole process, so they include building the inputs,
//! which is the same on every run. The variants cover the paths that differ in cost:
//! rows with and without repeated variables, contiguous and view strings, one chunk and many.
//!
//!     cargo bench --bench native                 # all benchmarks
//!     cargo bench --bench native -- --save-baseline=base
//!     cargo bench --bench native -- --baseline=base   # compare against the saved run
//!
//! Needs Valgrind 3.19 or later, which demangles Rust's v0 symbols (earlier versions never find
//! the benchmark functions and count zero), and `cargo install iai-callgrind-runner --version
//! <same as Cargo.toml>`.
//! Linux only: the CI job in `.github/workflows/bench.yml` runs it against the base branch.

use std::ffi::c_int;
use std::hint::black_box;
use std::sync::Arc;

use _native::columns::{cstrings_of, f64s_of, indices_of};
use _native::model::{merge_repeats, validate};
use arrow_array::{Array, ArrayRef, Float64Array, Int32Array, StringArray, StringViewArray};
use arrow_schema::DataType;
use iai_callgrind::{
    Dhat, LibraryBenchmarkConfig, library_benchmark, library_benchmark_group, main,
};

const NNZ: usize = 1_000_000;
const TERMS_PER_ROW: usize = 4;
const WIDTH: usize = 1_000_000;

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
fn csr(repeat_every: Option<usize>, first_row: usize) -> (Vec<usize>, Vec<c_int>, Vec<f64>) {
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

#[library_benchmark]
#[bench::no_repeats(csr(None, 0))]
#[bench::repeats_throughout(csr(Some(10), 0))]
#[bench::repeats_at_end_only(csr(Some(10), NNZ / TERMS_PER_ROW - 1000))]
fn bench_merge_repeats((cbeg, cind, cval): (Vec<usize>, Vec<c_int>, Vec<f64>)) {
    black_box(merge_repeats(&cbeg, &cind, &cval, WIDTH));
}

#[library_benchmark]
#[bench::million(csr(None, 0))]
fn bench_validate((cbeg, cind, _): (Vec<usize>, Vec<c_int>, Vec<f64>)) {
    black_box(validate(&cbeg, &cind, WIDTH).unwrap());
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

#[library_benchmark]
#[bench::utf8(names(false))]
#[bench::utf8view(names(true))]
fn bench_cstrings(chunks: Vec<ArrayRef>) {
    let strings = cstrings_of("names", &chunks, Some(NNZ)).unwrap();
    black_box(strings.ptrs());
}

/// `NNZ` indices in `chunks` equal pieces.
fn indices(chunks: usize) -> Vec<ArrayRef> {
    let mut rng = Lcg::new();
    let all: ArrayRef = Arc::new(Int32Array::from_iter_values(
        (0..NNZ).map(|_| rng.next(WIDTH) as c_int),
    ));
    split(all, chunks)
}

/// `NNZ` coefficients in `chunks` equal pieces.
fn f64s(chunks: usize) -> Vec<ArrayRef> {
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

#[library_benchmark]
#[bench::one_chunk(indices(1))]
#[bench::eight_chunks(indices(8))]
fn bench_indices(chunks: Vec<ArrayRef>) {
    black_box(indices_of("vars", &DataType::Int32, &chunks, Some(NNZ)).unwrap());
}

#[library_benchmark]
#[bench::one_chunk(f64s(1))]
#[bench::eight_chunks(f64s(8))]
fn bench_f64s(chunks: Vec<ArrayRef>) {
    black_box(f64s_of("coefs", &DataType::Float64, &chunks, Some(NNZ)).unwrap());
}

library_benchmark_group!(
    name = stages;
    benchmarks = bench_merge_repeats, bench_validate, bench_cstrings, bench_indices, bench_f64s
);

main!(
    // DHAT alongside callgrind: total bytes and blocks allocated, which the memory-oriented
    // changes to these stages aim at.
    config = LibraryBenchmarkConfig::default().tool(Dhat::default());
    library_benchmark_groups = stages
);
