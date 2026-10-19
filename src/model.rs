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

use crate::gurobi::{GrbModel, api};
use crate::{columns, gurobi};

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
    c_int::try_from(n)
        .map_err(|_| PyValueError::new_err(format!("too many {name} for one call: {n}")))
}

fn opt_ptr<T: Clone>(values: &Option<Cow<'_, [T]>>) -> *const T {
    values.as_deref().map_or(ptr::null(), <[T]>::as_ptr)
}

/// The `char **` for an optional name column. The shadowed `names` stays alive, so the pointers
/// are valid for the call that follows.
fn name_ptrs(names: &Option<columns::CStrings>) -> Option<Vec<*const c_char>> {
    names.as_ref().map(columns::CStrings::ptrs)
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
    let obj = obj
        .as_ref()
        .map(|c| columns::f64s("obj", c, Some(n)))
        .transpose()?;
    let lb = lb
        .as_ref()
        .map(|c| columns::f64s("lb", c, Some(n)))
        .transpose()?;
    let ub = ub
        .as_ref()
        .map(|c| columns::f64s("ub", c, Some(n)))
        .transpose()?;
    let vtype = vtype
        .as_ref()
        .map(|c| columns::chars("vtype", c, Some(n)))
        .transpose()?;
    let names = names
        .as_ref()
        .map(|c| columns::cstrings("names", c, Some(n)))
        .transpose()?;
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

/// Add one constraint per row of the List columns `vars` and `coefs`, which hold its terms. Their
/// Arrow offsets are the row starts of the CSR that `GRBXaddconstrs` takes, and their items the
/// variable indices and coefficients, so nothing is reshaped. `n_vars` is the model's variable
/// count, against which every index is checked.
#[pyfunction]
#[pyo3(signature = (cmodel, vars, coefs, sense, rhs, n_vars, names=None))]
#[allow(clippy::too_many_arguments)]
pub fn add_constrs(
    cmodel: &Bound<'_, PyAny>,
    vars: PyChunkedArray,
    coefs: PyChunkedArray,
    sense: PyChunkedArray,
    rhs: PyChunkedArray,
    n_vars: usize,
    names: Option<PyChunkedArray>,
) -> PyResult<()> {
    let (api, model) = (api()?, model_ptr(cmodel)?);
    let vars = columns::Lists::new("vars", &vars)?;
    let coefs = columns::Lists::new("coefs", &coefs)?;
    if vars.cbeg() != coefs.cbeg() || vars.nnz() != coefs.nnz() {
        return Err(PyValueError::new_err(
            "vars and coefs must hold lists of the same lengths",
        ));
    }
    let n = vars.rows();
    let n_constrs = c_int_len("constraints", n)?;
    let cind = vars.indices()?;
    let cval = coefs.f64s()?;
    let cbeg = vars.cbeg();
    let width = match validate(cbeg, &cind, n_vars) {
        Ok(width) => width,
        Err(Invalid::Offsets) => {
            return Err(PyValueError::new_err(
                "list offsets must be non-decreasing and within the items",
            ));
        }
        Err(Invalid::Index(bad)) => {
            return Err(gurobi::error(
                cmodel.py(),
                gurobi::INDEX_OUT_OF_RANGE,
                &format!("Variable index {bad} out of range: the model has {n_vars} variables"),
            ));
        }
    };
    let sense = columns::chars("sense", &sense, Some(n))?;
    let rhs = columns::f64s("rhs", &rhs, Some(n))?;
    let names = names
        .as_ref()
        .map(|c| columns::cstrings("names", c, Some(n)))
        .transpose()?;
    let names = name_ptrs(&names);
    let (cbeg, cind, cval) = match merge_repeats(cbeg, &cind, &cval, width) {
        Some((beg, ind, val)) => (Cow::Owned(beg), Cow::Owned(ind), Cow::Owned(val)),
        None => (Cow::Borrowed(cbeg), cind, cval),
    };
    let rc = unsafe {
        (api.xaddconstrs)(
            model,
            n_constrs,
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

/// Why a CSR cannot go to Gurobi.
#[derive(Debug, PartialEq)]
enum Invalid {
    /// A row start decreases or lies beyond the items.
    Offsets,
    /// A variable index is negative or not below the model's variable count.
    Index(c_int),
}

/// Check what Gurobi would trust blindly or reject without saying why: the row starts `cbeg`,
/// which it reads without bounds checks, and the variable indices `cind` against the model's
/// `n_vars` variables. Returns the width `merge_repeats` needs, one past the largest index.
fn validate(cbeg: &[usize], cind: &[c_int], n_vars: usize) -> Result<usize, Invalid> {
    let mut prev = 0;
    for &beg in cbeg {
        if beg < prev || beg > cind.len() {
            return Err(Invalid::Offsets);
        }
        prev = beg;
    }
    let (low, high) = cind
        .iter()
        .fold((0, -1), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    if low < 0 {
        return Err(Invalid::Index(low));
    }
    let width = usize::try_from(high + 1).unwrap_or_default();
    if width > n_vars {
        return Err(Invalid::Index(high));
    }
    Ok(width)
}

/// Gurobi rejects a variable that appears more than once in a row. Merge repeats by summing their
/// coefficients left to right, in the order of the terms; `None` if there are none. Gurobi itself
/// then drops coefficients that are zero or tiny. Every index in `cind` is in `0..width`, and
/// there are at most `c_int::MAX` rows.
fn merge_repeats(cbeg: &[usize], cind: &[c_int], cval: &[f64], width: usize) -> Option<Csr> {
    let rows = || {
        let ends = cbeg
            .iter()
            .skip(1)
            .copied()
            .chain(std::iter::once(cind.len()));
        cbeg.iter().copied().zip(ends)
    };
    // A few short rows against a wide model, the shape of an incremental build, are scanned in
    // place: a table as wide as the model would cost more to allocate than the rows to merge.
    let longest = rows().map(|(start, end)| end - start).max().unwrap_or(0);
    if longest <= SCAN_ROW_LEN && cind.len() < width / 16 {
        return merge_repeats_by_scan(rows, cind, cval);
    }
    // `seen[v]` is the last row that used column `v`, offset by one so zero means none. The
    // table is as wide as the model, so it is kept to four bytes per column.
    let mut seen = vec![0u32; width];
    let repeats = rows().enumerate().any(|(row, (start, end))| {
        let row = row as u32 + 1;
        cind[start..end]
            .iter()
            .any(|&v| std::mem::replace(&mut seen[v as usize], row) == row)
    });
    if !repeats {
        return None;
    }
    seen.fill(0);
    let mut slot = vec![0usize; width];
    let (mut beg, mut ind, mut val) = (Vec::with_capacity(cbeg.len()), Vec::new(), Vec::new());
    for (row, (start, end)) in rows().enumerate() {
        let row = row as u32 + 1;
        beg.push(ind.len());
        for (&v, &coef) in cind[start..end].iter().zip(&cval[start..end]) {
            let column = v as usize;
            if seen[column] == row {
                val[slot[column]] += coef;
            } else {
                seen[column] = row;
                slot[column] = ind.len();
                ind.push(v);
                val.push(coef);
            }
        }
    }
    Some((beg, ind, val))
}

/// Rows up to this long are checked for repeats by a linear scan of the row itself.
const SCAN_ROW_LEN: usize = 64;

/// `merge_repeats` for short rows: each term is compared with the ones before it in its row.
fn merge_repeats_by_scan<I>(rows: impl Fn() -> I, cind: &[c_int], cval: &[f64]) -> Option<Csr>
where
    I: Iterator<Item = (usize, usize)>,
{
    let repeats = rows().any(|(start, end)| {
        let row = &cind[start..end];
        (1..row.len()).any(|i| row[..i].contains(&row[i]))
    });
    if !repeats {
        return None;
    }
    let (mut beg, mut ind, mut val) = (Vec::new(), Vec::new(), Vec::new());
    for (start, end) in rows() {
        beg.push(ind.len());
        let first = ind.len();
        for (&v, &coef) in cind[start..end].iter().zip(&cval[start..end]) {
            match ind[first..].iter().position(|&w| w == v) {
                Some(at) => val[first + at] += coef,
                None => {
                    ind.push(v);
                    val.push(coef);
                }
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
        Ok(to_array_pycapsules(
            py,
            self.field.clone(),
            &self.array,
            requested_schema,
        )?)
    }
}

/// Read attribute `attr` for the elements at `ind`. Gurobi writes straight into the buffer that
/// backs the returned Float64 array, or Int32 array if `int` is set.
#[pyfunction]
#[pyo3(signature = (cmodel, attr, ind, int=false))]
pub fn get_attr(
    cmodel: &Bound<'_, PyAny>,
    attr: &str,
    ind: PyChunkedArray,
    int: bool,
) -> PyResult<ArrowArray> {
    let (api, model, py) = (api()?, model_ptr(cmodel)?, cmodel.py());
    let ind = columns::indices("ind", &ind, None)?;
    let len = c_int_len("elements", ind.len())?;
    let name = CString::new(attr)?;
    let (rc, array): (c_int, ArrayRef) = if int {
        let mut out = vec![0; ind.len()];
        let rc = unsafe {
            (api.getintattrlist)(model, name.as_ptr(), len, ind.as_ptr(), out.as_mut_ptr())
        };
        (rc, Arc::new(Int32Array::from(out)))
    } else {
        let mut out = vec![0.0; ind.len()];
        let rc = unsafe {
            (api.getdblattrlist)(model, name.as_ptr(), len, ind.as_ptr(), out.as_mut_ptr())
        };
        (rc, Arc::new(Float64Array::from(out)))
    };
    api.check(py, model, rc)?;
    let field = Arc::new(Field::new(attr, array.data_type().clone(), false));
    Ok(ArrowArray { field, array })
}

/// Set attribute `attr` for the elements at `ind`. Float64 values use the double API, Int32
/// values the int API.
#[pyfunction]
pub fn set_attr(
    cmodel: &Bound<'_, PyAny>,
    attr: &str,
    ind: PyChunkedArray,
    values: PyChunkedArray,
) -> PyResult<()> {
    let (api, model, py) = (api()?, model_ptr(cmodel)?, cmodel.py());
    let ind = columns::indices("ind", &ind, None)?;
    let len = c_int_len("elements", ind.len())?;
    let name = CString::new(attr)?;
    let rc = match values.field().data_type() {
        arrow_schema::DataType::Int32 => {
            let values = columns::indices("values", &values, Some(ind.len()))?;
            unsafe {
                (api.setintattrlist)(model, name.as_ptr(), len, ind.as_ptr(), values.as_ptr())
            }
        }
        _ => {
            let values = columns::f64s("values", &values, Some(ind.len()))?;
            unsafe {
                (api.setdblattrlist)(model, name.as_ptr(), len, ind.as_ptr(), values.as_ptr())
            }
        }
    };
    api.check(py, model, rc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_and_measures() {
        assert_eq!(validate(&[0, 2, 2], &[1, 0, 4], 5), Ok(5));
        assert_eq!(validate(&[0, 0], &[], 0), Ok(0));
        assert_eq!(validate(&[], &[], 0), Ok(0));
        // A last row that ends exactly at the items is valid, and empty.
        assert_eq!(validate(&[0, 3], &[0, 1, 2], 3), Ok(3));
    }

    #[test]
    fn validate_rejects_offsets() {
        assert_eq!(validate(&[0, 2, 1], &[0, 1, 2], 3), Err(Invalid::Offsets));
        assert_eq!(validate(&[0, 4], &[0, 1, 2], 3), Err(Invalid::Offsets));
    }

    #[test]
    fn validate_rejects_indices() {
        assert_eq!(validate(&[0], &[0, 3], 3), Err(Invalid::Index(3)));
        assert_eq!(validate(&[0], &[2, -1, 9], 3), Err(Invalid::Index(-1)));
        assert_eq!(validate(&[0], &[0], 0), Err(Invalid::Index(0)));
        assert_eq!(
            validate(&[0], &[c_int::MIN], 3),
            Err(Invalid::Index(c_int::MIN))
        );
    }

    /// Each row with its repeats summed, in order of first appearance.
    fn reference(cbeg: &[usize], cind: &[c_int], cval: &[f64]) -> Csr {
        let (mut beg, mut ind, mut val) = (Vec::new(), Vec::new(), Vec::new());
        for (row, &start) in cbeg.iter().enumerate() {
            let end = cbeg.get(row + 1).copied().unwrap_or(cind.len());
            beg.push(ind.len());
            let first = ind.len();
            for (&v, &c) in cind[start..end].iter().zip(&cval[start..end]) {
                match ind[first..].iter().position(|&w| w == v) {
                    Some(at) => val[first + at] += c,
                    None => {
                        ind.push(v);
                        val.push(c);
                    }
                }
            }
        }
        (beg, ind, val)
    }

    #[test]
    fn merge_none_without_repeats() {
        // The same variable in different rows is not a repeat.
        assert_eq!(merge_repeats(&[0, 2, 3], &[0, 1, 0, 1], &[1.0; 4], 2), None);
        assert_eq!(merge_repeats(&[], &[], &[], 0), None);
        assert_eq!(merge_repeats(&[0, 0], &[], &[], 0), None);
    }

    #[test]
    fn merge_sums_left_to_right() {
        let (a, b, c) = (
            -999999999.0 * 3.0,
            -183862175.19652724 * 3.0,
            627515632.0 * 3.0,
        );
        assert_ne!((a + b) + c, a + (b + c));
        let merged = merge_repeats(&[0], &[1, 0, 1, 1], &[a, 7.0, b, c], 2);
        assert_eq!(merged, Some((vec![0], vec![1, 0], vec![(a + b) + c, 7.0])));
    }

    #[test]
    fn merge_scans_short_rows_of_wide_models() {
        // Few terms against a wide model: no table of `width` entries is needed.
        let width = 1 << 20;
        let big = (width - 1) as c_int;
        assert_eq!(merge_repeats(&[0, 2], &[big, 0, 5], &[1.0; 3], width), None);
        let merged = merge_repeats(
            &[0, 0, 3],
            &[big, 0, big, 7, 7],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            width,
        );
        assert_eq!(
            merged,
            Some((vec![0, 0, 2], vec![big, 0, 7], vec![4.0, 2.0, 9.0]))
        );
        // A row longer than the scan limit falls back to the table even in a wide model.
        let long: Vec<c_int> = (0..=SCAN_ROW_LEN as c_int).chain([0]).collect();
        let merged = merge_repeats(&[0], &long, &vec![1.0; long.len()], width).unwrap();
        assert_eq!(merged.1.len(), SCAN_ROW_LEN + 1);
        assert_eq!(merged.2[0], 2.0);
    }

    #[test]
    fn merge_keeps_empty_rows() {
        let merged = merge_repeats(&[0, 0, 2, 2], &[3, 3, 1], &[1.0, 2.0, 5.0], 4);
        assert_eq!(merged, Some((vec![0, 0, 1, 1], vec![3, 1], vec![3.0, 5.0])));
    }

    #[test]
    fn merge_matches_reference() {
        // A fixed-seed generator, so failures reproduce without a property-testing dependency.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = |bound: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % bound
        };
        for i in 0..4000 {
            // Wide models with few terms take the scan path, narrow ones the table.
            let width = 1 + next(5) as usize + if i % 2 == 0 { 0 } else { 1000 };
            let (rows, nnz) = (next(6) as usize, next(16) as usize);
            let span = width.min(5) as u64; // keep repeats likely in wide models too
            let cind: Vec<c_int> = (0..nnz).map(|_| next(span) as c_int).collect();
            let cval: Vec<f64> = (0..nnz).map(|_| next(9) as f64 - 4.0).collect();
            let mut cbeg: Vec<usize> = (0..rows).map(|_| next(nnz as u64 + 1) as usize).collect();
            cbeg.sort_unstable();
            if let Some(first) = cbeg.first_mut() {
                *first = 0;
            } else if nnz > 0 {
                continue; // No rows cannot hold terms.
            }
            let want = reference(&cbeg, &cind, &cval);
            match merge_repeats(&cbeg, &cind, &cval, width) {
                Some(got) => assert_eq!(got, want, "cbeg {cbeg:?} cind {cind:?}"),
                None => assert_eq!((cbeg.clone(), cind.clone(), cval.clone()), want),
            }
        }
    }
}
