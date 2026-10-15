//! Bulk calls into a gurobipy-owned `GRBmodel*`. Registering the matching Python handles is left
//! to the Python layer, which owns gurobipy's bookkeeping.

use std::borrow::Cow;
use std::ffi::{CString, c_char, c_int};
use std::ptr;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int32Array};
use arrow_schema::{Field, FieldRef};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};
use pyo3_arrow::PyChunkedArray;
use pyo3_arrow::ffi::to_array_pycapsules;

use crate::columns;
use crate::gurobi::{GrbModel, api};

/// The `GRBmodel*` behind gurobipy's `Model._cmodel` capsule.
fn model_ptr(cmodel: &Bound<'_, PyAny>) -> PyResult<*mut GrbModel> {
    let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(cmodel.as_ptr(), ptr::null()) };
    if ptr.is_null() {
        return Err(PyErr::take(cmodel.py())
            .unwrap_or_else(|| PyValueError::new_err("Model._cmodel holds a null pointer")));
    }
    Ok(ptr.cast())
}

fn c_int_len(name: &str, n: usize) -> PyResult<c_int> {
    c_int::try_from(n).map_err(|_| PyValueError::new_err(format!("too many {name} for one call: {n}")))
}

fn opt_ptr<T: Clone>(values: &Option<Cow<'_, [T]>>) -> *const T {
    values.as_deref().map_or(ptr::null(), <[T]>::as_ptr)
}

fn name_ptrs(names: &Option<Vec<CString>>) -> Option<Vec<*const c_char>> {
    names.as_ref().map(|names| names.iter().map(|s| s.as_ptr()).collect())
}

#[pyfunction]
#[pyo3(signature = (cmodel, n, obj=None, lb=None, ub=None, vtype=None, names=None))]
#[allow(clippy::too_many_arguments)]
pub fn add_vars(
    cmodel: &Bound<'_, PyAny>,
    n: usize,
    obj: Option<PyChunkedArray>,
    lb: Option<PyChunkedArray>,
    ub: Option<PyChunkedArray>,
    vtype: Option<PyChunkedArray>,
    names: Option<PyChunkedArray>,
) -> PyResult<()> {
    let (api, model) = (api()?, model_ptr(cmodel)?);
    let obj = obj.as_ref().map(|c| columns::f64s("obj", c, Some(n))).transpose()?;
    let lb = lb.as_ref().map(|c| columns::f64s("lb", c, Some(n))).transpose()?;
    let ub = ub.as_ref().map(|c| columns::f64s("ub", c, Some(n))).transpose()?;
    let vtype = vtype.as_ref().map(|c| columns::chars("vtype", c, Some(n))).transpose()?;
    let names = names.as_ref().map(|c| columns::cstrings("names", c, Some(n))).transpose()?;
    let names = name_ptrs(&names);
    let rc = unsafe {
        (api.addvars)(
            model,
            c_int_len("variables", n)?,
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            opt_ptr(&obj),
            opt_ptr(&lb),
            opt_ptr(&ub),
            opt_ptr(&vtype),
            names.as_ref().map_or(ptr::null(), |v| v.as_ptr()),
        )
    };
    api.check(cmodel.py(), model, rc)
}

/// Add constraints from CSR: row `i` holds `cind[cbeg[i]..cbeg[i+1]]` with coefficients `cval`.
#[pyfunction]
#[pyo3(signature = (cmodel, cbeg, cind, cval, sense, rhs, names=None))]
pub fn add_constrs(
    cmodel: &Bound<'_, PyAny>,
    cbeg: PyChunkedArray,
    cind: PyChunkedArray,
    cval: PyChunkedArray,
    sense: PyChunkedArray,
    rhs: PyChunkedArray,
    names: Option<PyChunkedArray>,
) -> PyResult<()> {
    let (api, model) = (api()?, model_ptr(cmodel)?);
    let cbeg = columns::offsets("cbeg", &cbeg)?;
    let n = cbeg.len();
    let cind = columns::indices("cind", &cind, None)?;
    let cval = columns::f64s("cval", &cval, Some(cind.len()))?;
    // Gurobi trusts cbeg blindly, so a bad offset would read out of bounds.
    let mut prev = 0;
    for &beg in cbeg.iter() {
        if beg < prev || beg > cind.len() {
            return Err(PyValueError::new_err("cbeg must be non-decreasing and within cind"));
        }
        prev = beg;
    }
    let sense = columns::chars("sense", &sense, Some(n))?;
    let rhs = columns::f64s("rhs", &rhs, Some(n))?;
    let names = names.as_ref().map(|c| columns::cstrings("names", c, Some(n))).transpose()?;
    let names = name_ptrs(&names);
    let (cbeg, cind, cval) = match merge_repeats(&cbeg, &cind, &cval) {
        Some((beg, ind, val)) => (Cow::Owned(beg), Cow::Owned(ind), Cow::Owned(val)),
        None => (cbeg, cind, cval),
    };
    let rc = unsafe {
        (api.xaddconstrs)(
            model,
            c_int_len("constraints", n)?,
            cind.len(),
            cbeg.as_ptr(),
            cind.as_ptr(),
            cval.as_ptr(),
            sense.as_ptr(),
            rhs.as_ptr(),
            names.as_ref().map_or(ptr::null(), |v| v.as_ptr()),
        )
    };
    api.check(cmodel.py(), model, rc)
}

type Csr = (Vec<usize>, Vec<c_int>, Vec<f64>);

/// gurobipy accepts a variable more than once per row and sums its coefficients left to right,
/// while Gurobi's bulk call rejects repeats. Merge them the same way; `None` if there are none.
/// Gurobi itself then drops coefficients that are zero or tiny, for either path alike.
fn merge_repeats(cbeg: &[usize], cind: &[c_int], cval: &[f64]) -> Option<Csr> {
    if cind.iter().any(|&v| v < 0) {
        return None; // Gurobi reports the invalid index.
    }
    let width = cind.iter().map(|&v| v as usize + 1).max().unwrap_or(0);
    let rows = || {
        let ends = cbeg.iter().skip(1).copied().chain(std::iter::once(cind.len()));
        cbeg.iter().copied().zip(ends)
    };
    // `seen[v]` is the last row that used column `v`, offset by one so zero means none.
    let mut seen = vec![0usize; width];
    let repeats = rows().enumerate().any(|(row, (start, end))| {
        cind[start..end].iter().any(|&v| std::mem::replace(&mut seen[v as usize], row + 1) == row + 1)
    });
    if !repeats {
        return None;
    }
    seen.fill(0);
    let mut slot = vec![0usize; width];
    let (mut beg, mut ind, mut val) = (Vec::with_capacity(cbeg.len()), Vec::new(), Vec::new());
    for (row, (start, end)) in rows().enumerate() {
        beg.push(ind.len());
        for (&v, &coef) in cind[start..end].iter().zip(&cval[start..end]) {
            let column = v as usize;
            if seen[column] == row + 1 {
                val[slot[column]] += coef;
            } else {
                seen[column] = row + 1;
                slot[column] = ind.len();
                ind.push(v);
                val.push(coef);
            }
        }
    }
    Some((beg, ind, val))
}

/// An Arrow array handed to Python through the Arrow PyCapsule interface, which polars imports
/// without copying.
#[pyclass(frozen, name = "ArrowArray", module = "polarobi._native")]
pub struct ArrowArray {
    field: FieldRef,
    array: ArrayRef,
}

#[pymethods]
impl ArrowArray {
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        Ok(to_array_pycapsules(py, self.field.clone(), &self.array, requested_schema)?)
    }
}

/// Read attribute `attr` for the elements at `ind`. Gurobi writes straight into the buffer that
/// backs the returned Float64 array, or Int32 array if `int` is set.
#[pyfunction]
#[pyo3(signature = (cmodel, attr, ind, int=false))]
pub fn get_attr(cmodel: &Bound<'_, PyAny>, attr: &str, ind: PyChunkedArray, int: bool) -> PyResult<ArrowArray> {
    let (api, model, py) = (api()?, model_ptr(cmodel)?, cmodel.py());
    let ind = columns::indices("ind", &ind, None)?;
    let len = c_int_len("elements", ind.len())?;
    let name = CString::new(attr)?;
    let (rc, array): (c_int, ArrayRef) = if int {
        let mut out = vec![0; ind.len()];
        let rc = unsafe { (api.getintattrlist)(model, name.as_ptr(), len, ind.as_ptr(), out.as_mut_ptr()) };
        (rc, Arc::new(Int32Array::from(out)))
    } else {
        let mut out = vec![0.0; ind.len()];
        let rc = unsafe { (api.getdblattrlist)(model, name.as_ptr(), len, ind.as_ptr(), out.as_mut_ptr()) };
        (rc, Arc::new(Float64Array::from(out)))
    };
    api.check(py, model, rc)?;
    let field = Arc::new(Field::new(attr, array.data_type().clone(), false));
    Ok(ArrowArray { field, array })
}

/// Set attribute `attr` for the elements at `ind`. Float64 values use the double API, Int32
/// values the int API.
#[pyfunction]
pub fn set_attr(cmodel: &Bound<'_, PyAny>, attr: &str, ind: PyChunkedArray, values: PyChunkedArray) -> PyResult<()> {
    let (api, model, py) = (api()?, model_ptr(cmodel)?, cmodel.py());
    let ind = columns::indices("ind", &ind, None)?;
    let len = c_int_len("elements", ind.len())?;
    let name = CString::new(attr)?;
    let rc = match values.field().data_type() {
        arrow_schema::DataType::Int32 => {
            let values = columns::indices("values", &values, Some(ind.len()))?;
            unsafe { (api.setintattrlist)(model, name.as_ptr(), len, ind.as_ptr(), values.as_ptr()) }
        }
        _ => {
            let values = columns::f64s("values", &values, Some(ind.len()))?;
            unsafe { (api.setdblattrlist)(model, name.as_ptr(), len, ind.as_ptr(), values.as_ptr()) }
        }
    };
    api.check(py, model, rc)
}
