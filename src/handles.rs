//! gurobipy handles for elements added through the C API.
//!
//! gurobipy keeps one handle per variable and constraint in a list next to Gurobi's model, and
//! reads every entry on `update()`. Creating the handles here saves the Python loop around the
//! constructor; the constructor itself is gurobipy's and costs the same as in `Model.addVars`.

use std::ffi::c_int;
use std::panic;
use std::thread;

use pyo3::prelude::*;
use pyo3::types::PyList;

/// Below this many elements a thread costs more than the overlap it buys.
const OVERLAP_MIN: usize = 1 << 14;

/// Append `cls(cmodel, i)` for every `i` in `start..start + count` to gurobipy's list `pool`.
/// The handles are created first and appended after, so a failure leaves `pool` unchanged.
#[pyfunction]
pub fn register(
    pool: &Bound<'_, PyList>,
    cls: &Bound<'_, PyAny>,
    cmodel: &Bound<'_, PyAny>,
    start: usize,
    count: usize,
) -> PyResult<()> {
    append(pool, create(cls, cmodel, start, count)?)
}

fn create<'py>(
    cls: &Bound<'py, PyAny>,
    cmodel: &Bound<'py, PyAny>,
    start: usize,
    count: usize,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    (start..start + count)
        .map(|i| cls.call1((cmodel, i)))
        .collect()
}

fn append<'py>(pool: &Bound<'py, PyList>, handles: Vec<Bound<'py, PyAny>>) -> PyResult<()> {
    for handle in handles {
        pool.append(handle)?;
    }
    Ok(())
}

/// A Gurobi call that may run on another thread, built only through the `unsafe` [`Self::new`].
///
/// The call deliberately does not run on the calling thread for large calls: Gurobi allows any
/// thread to use a model as long as no two use it at once, and the calling thread keeps the GIL
/// to create handles meanwhile, so no Python state crosses threads. Like gurobipy itself, this
/// assumes the user does not drive one model from several Python threads at once.
pub struct GurobiCall<F>(F);

// SAFETY: `new`'s contract: the closure captures no Python objects, only raw pointers and plain
// data that nothing else touches until it returns, so moving it to one other thread is sound.
unsafe impl<F> Send for GurobiCall<F> {}

impl<F: FnOnce() -> c_int> GurobiCall<F> {
    /// # Safety
    ///
    /// `call` must not capture Python objects (`Bound`, `Py`, `Python`) or anything else that is
    /// not safe to use from another thread, and everything it points into must stay alive and
    /// unmodified by other code until it returns. It may run on a thread other than the caller's.
    pub unsafe fn new(call: F) -> Self {
        Self(call)
    }

    fn run(self) -> c_int {
        (self.0)()
    }
}

/// Run `call`, which adds `count` elements, and register their handles `cls(cmodel, i)` in `pool`
/// if `pool` and `cls` are given. Handle creation needs the GIL and the Gurobi call does not, so
/// for large calls the two run at the same time; the handles are appended only once the call
/// succeeds. Returns the call's return code.
pub fn with_handles<F: FnOnce() -> c_int>(
    cmodel: &Bound<'_, PyAny>,
    handles: Option<(&Bound<'_, PyList>, &Bound<'_, PyAny>)>,
    count: usize,
    call: GurobiCall<F>,
) -> PyResult<c_int> {
    let Some((pool, cls)) = handles else {
        return Ok(call.run());
    };
    let start = pool.len();
    let (rc, created) = if count < OVERLAP_MIN {
        let rc = call.run();
        if rc != 0 {
            return Ok(rc);
        }
        (rc, create(cls, cmodel, start, count))
    } else {
        thread::scope(|s| {
            let worker = s.spawn(move || call.run());
            let created = create(cls, cmodel, start, count);
            let rc = worker.join().unwrap_or_else(|p| panic::resume_unwind(p));
            (rc, created)
        })
    };
    if rc == 0 {
        append(pool, created?)?;
        // Nothing else may add to gurobipy's list while its new elements are being added.
        debug_assert_eq!(
            pool.len(),
            start + count,
            "handle pool changed during the call"
        );
    }
    Ok(rc)
}
