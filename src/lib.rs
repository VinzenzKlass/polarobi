mod columns;
mod gurobi;
mod model;

use pyo3::prelude::*;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<model::ArrowArray>()?;
    m.add_function(wrap_pyfunction!(gurobi::load, m)?)?;
    m.add_function(wrap_pyfunction!(model::add_vars, m)?)?;
    m.add_function(wrap_pyfunction!(model::add_constrs, m)?)?;
    m.add_function(wrap_pyfunction!(model::get_attr, m)?)?;
    m.add_function(wrap_pyfunction!(model::set_attr, m)?)?;
    Ok(())
}
