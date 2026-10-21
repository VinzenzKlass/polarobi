//! The pure Rust stages of a call under divan, for CodSpeed: the same stages and inputs as
//! `native.rs` and `timing.rs`, measured with CodSpeed's CPU simulation, memory and wall-time
//! instruments by `.github/workflows/codspeed.yml`.
//!
//! Inputs are built once, outside the measured functions, and shared by every iteration.
//!
//!     cargo bench --bench stages                     # plain divan, anywhere
//!     cargo codspeed build --bench stages && cargo codspeed run --bench stages

#[path = "inputs.rs"]
mod inputs;

use std::ffi::c_int;
use std::sync::LazyLock;

use _native::columns::{cstrings_of, f64s_of, indices_of};
use _native::model::{merge_repeats, validate};
use arrow_array::ArrayRef;
use arrow_schema::DataType;
use divan::black_box;
use inputs::{NNZ, TERMS_PER_ROW, WIDTH, csr, f64s, indices, names};

type Csr = (Vec<usize>, Vec<c_int>, Vec<f64>);

static NO_REPEATS: LazyLock<Csr> = LazyLock::new(|| csr(None, 0));
static REPEATS_THROUGHOUT: LazyLock<Csr> = LazyLock::new(|| csr(Some(10), 0));
static REPEATS_AT_END_ONLY: LazyLock<Csr> =
    LazyLock::new(|| csr(Some(10), NNZ / TERMS_PER_ROW - 1000));
static UTF8: LazyLock<Vec<ArrayRef>> = LazyLock::new(|| names(false));
static UTF8VIEW: LazyLock<Vec<ArrayRef>> = LazyLock::new(|| names(true));
static INDICES_ONE: LazyLock<Vec<ArrayRef>> = LazyLock::new(|| indices(1));
static INDICES_EIGHT: LazyLock<Vec<ArrayRef>> = LazyLock::new(|| indices(8));
static F64S_ONE: LazyLock<Vec<ArrayRef>> = LazyLock::new(|| f64s(1));
static F64S_EIGHT: LazyLock<Vec<ArrayRef>> = LazyLock::new(|| f64s(8));

fn main() {
    divan::main();
}

/// Runs `f` on the input, forcing the input first so that building it is never measured.
fn bench_on<I: Sync + 'static, O>(
    bencher: divan::Bencher,
    input: &'static LazyLock<I>,
    f: impl Fn(&I) -> O + Sync,
) {
    let input: &I = LazyLock::force(input);
    bencher.bench(|| f(black_box(input)));
}

mod merge_repeats {
    use super::*;

    fn merge((cbeg, cind, cval): &Csr) {
        black_box(merge_repeats(cbeg, cind, cval, WIDTH));
    }

    #[divan::bench]
    fn no_repeats(bencher: divan::Bencher) {
        bench_on(bencher, &NO_REPEATS, merge);
    }

    #[divan::bench]
    fn repeats_throughout(bencher: divan::Bencher) {
        bench_on(bencher, &REPEATS_THROUGHOUT, merge);
    }

    #[divan::bench]
    fn repeats_at_end_only(bencher: divan::Bencher) {
        bench_on(bencher, &REPEATS_AT_END_ONLY, merge);
    }
}

mod validate {
    use super::*;

    #[divan::bench]
    fn million(bencher: divan::Bencher) {
        bench_on(bencher, &NO_REPEATS, |(cbeg, cind, _)| {
            validate(cbeg, cind, WIDTH).unwrap()
        });
    }
}

mod cstrings {
    use super::*;

    fn strings(chunks: &Vec<ArrayRef>) {
        black_box(cstrings_of("names", chunks, Some(NNZ)).unwrap().ptrs());
    }

    #[divan::bench]
    fn utf8(bencher: divan::Bencher) {
        bench_on(bencher, &UTF8, strings);
    }

    #[divan::bench]
    fn utf8view(bencher: divan::Bencher) {
        bench_on(bencher, &UTF8VIEW, strings);
    }
}

mod indices {
    use super::*;

    fn ints(chunks: &Vec<ArrayRef>) {
        black_box(indices_of("vars", &DataType::Int32, chunks, Some(NNZ)).unwrap());
    }

    #[divan::bench]
    fn one_chunk(bencher: divan::Bencher) {
        bench_on(bencher, &INDICES_ONE, ints);
    }

    #[divan::bench]
    fn eight_chunks(bencher: divan::Bencher) {
        bench_on(bencher, &INDICES_EIGHT, ints);
    }
}

mod f64s {
    use super::*;

    fn floats(chunks: &Vec<ArrayRef>) {
        black_box(f64s_of("coefs", &DataType::Float64, chunks, Some(NNZ)).unwrap());
    }

    #[divan::bench]
    fn one_chunk(bencher: divan::Bencher) {
        bench_on(bencher, &F64S_ONE, floats);
    }

    #[divan::bench]
    fn eight_chunks(bencher: divan::Bencher) {
        bench_on(bencher, &F64S_EIGHT, floats);
    }
}
