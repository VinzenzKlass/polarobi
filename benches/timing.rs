//! Wall-clock timings of the pure Rust stages, on the same inputs as `native.rs`. Where that
//! counts instructions under Valgrind, this measures time: the number that users feel, and the
//! one that catches changes that trade instructions for cache misses or threads.
//!
//! Timings are noisy wherever they run, so each stage is timed `REPEATS` times after a warm-up
//! and the minimum and median are reported. Compare builds by running them alternately on the
//! same machine, as `.github/workflows/bench.yml` does against the base commit, never by two
//! separate sessions.
//!
//!     cargo bench --bench timing                    # a table on stdout
//!     cargo bench --bench timing -- --json          # one JSON line per stage, for scripts
//!
//! Runs on any platform; nothing but `std` is needed.

#[path = "inputs.rs"]
mod inputs;

use std::ffi::c_int;
use std::hint::black_box;
use std::time::{Duration, Instant};

use _native::columns::{cstrings_of, f64s_of, indices_of};
use _native::model::{merge_repeats, validate};
use arrow_array::ArrayRef;
use arrow_schema::DataType;
use inputs::{NNZ, TERMS_PER_ROW, WIDTH, csr, f64s, indices, names};

const REPEATS: usize = 20;

type Csr = (Vec<usize>, Vec<c_int>, Vec<f64>);

/// `f` timed `REPEATS` times on `input`, after one untimed run.
fn time<I>(input: &I, f: impl Fn(&I)) -> Vec<Duration> {
    f(input);
    (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            f(black_box(input));
            start.elapsed()
        })
        .collect()
}

fn merge((cbeg, cind, cval): &Csr) {
    black_box(merge_repeats(cbeg, cind, cval, WIDTH));
}

fn check((cbeg, cind, _): &Csr) {
    black_box(validate(cbeg, cind, WIDTH).unwrap());
}

fn strings(chunks: &Vec<ArrayRef>) {
    black_box(cstrings_of("names", chunks, Some(NNZ)).unwrap().ptrs());
}

fn ints(chunks: &Vec<ArrayRef>) {
    black_box(indices_of("vars", &DataType::Int32, chunks, Some(NNZ)).unwrap());
}

fn floats(chunks: &Vec<ArrayRef>) {
    black_box(f64s_of("coefs", &DataType::Float64, chunks, Some(NNZ)).unwrap());
}

/// Every stage with the same names as in `native.rs`, timed in turn.
fn run() -> Vec<(&'static str, Vec<Duration>)> {
    vec![
        ("bench_merge_repeats no_repeats", time(&csr(None, 0), merge)),
        (
            "bench_merge_repeats repeats_throughout",
            time(&csr(Some(10), 0), merge),
        ),
        (
            "bench_merge_repeats repeats_at_end_only",
            time(&csr(Some(10), NNZ / TERMS_PER_ROW - 1000), merge),
        ),
        ("bench_validate million", time(&csr(None, 0), check)),
        ("bench_cstrings utf8", time(&names(false), strings)),
        ("bench_cstrings utf8view", time(&names(true), strings)),
        ("bench_indices one_chunk", time(&indices(1), ints)),
        ("bench_indices eight_chunks", time(&indices(8), ints)),
        ("bench_f64s one_chunk", time(&f64s(1), floats)),
        ("bench_f64s eight_chunks", time(&f64s(8), floats)),
    ]
}

fn main() {
    let json = std::env::args().any(|arg| arg == "--json");
    if !json {
        println!("{:40} {:>12} {:>12}", "stage", "min", "median");
    }
    for (name, mut times) in run() {
        times.sort_unstable();
        let (min, median) = (times[0], times[times.len() / 2]);
        if json {
            println!(
                "{{\"name\":\"{name}\",\"min_ns\":{},\"median_ns\":{}}}",
                min.as_nanos(),
                median.as_nanos()
            );
        } else {
            println!("{name:40} {min:>12.3?} {median:>12.3?}");
        }
    }
}
