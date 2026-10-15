//! gurobipy handles for elements added through the C API.
//!
//! gurobipy keeps one handle per variable and constraint in a list next to Gurobi's model, and
//! reads every entry on `update()`. Creating the handles here saves the Python loop around the
//! constructor; the constructor itself is gurobipy's and costs the same as in `Model.addVars`.

use pyo3::prelude::*;
use pyo3::types::PyList;

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
    let handles = (start..start + count)
        .map(|i| cls.call1((cmodel, i)))
        .collect::<PyResult<Vec<_>>>()?;
    for handle in handles {
        pool.append(handle)?;
    }
    Ok(())
}
