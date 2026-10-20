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

#[path = "inputs.rs"]
mod inputs;

use std::ffi::c_int;
use std::hint::black_box;

use _native::columns::{cstrings_of, f64s_of, indices_of};
use _native::model::{merge_repeats, validate};
use arrow_array::ArrayRef;
use arrow_schema::DataType;
use iai_callgrind::{
    Dhat, LibraryBenchmarkConfig, library_benchmark, library_benchmark_group, main,
};
use inputs::{NNZ, TERMS_PER_ROW, WIDTH, csr, f64s, indices, names};

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

#[library_benchmark]
#[bench::utf8(names(false))]
#[bench::utf8view(names(true))]
fn bench_cstrings(chunks: Vec<ArrayRef>) {
    let strings = cstrings_of("names", &chunks, Some(NNZ)).unwrap();
    black_box(strings.ptrs());
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
