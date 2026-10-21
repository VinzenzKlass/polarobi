//! Bulk calls into a gurobipy-owned `GRBmodel*`. Given gurobipy's handle list and class, each call
//! also registers the handles of what it adds, created while Gurobi works; the Python layer keeps
//! the rest of gurobipy's bookkeeping.

use std::borrow::Cow;
use std::ffi::{CString, c_char, c_int};
use std::ptr;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int32Array};
use arrow_schema::{Field, FieldRef};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList, PyTuple};
use pyo3_arrow::PyChunkedArray;
use pyo3_arrow::ffi::to_array_pycapsules;

use crate::gurobi::{GrbModel, api};
use crate::handles::{GurobiCall, with_handles};
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
fn name_ptrs(names: &Option<columns::CStrings>) -> Option<&[*const c_char]> {
    names.as_ref().map(columns::CStrings::ptrs)
}

#[pyfunction]
#[pyo3(signature = (cmodel, n, obj=None, lb=None, ub=None, vtype=None, names=None, pool=None, cls=None))]
#[allow(clippy::too_many_arguments)]
pub fn add_vars(
    cmodel: &Bound<'_, PyAny>,
    n: usize,
    obj: Option<PyChunkedArray>,
    lb: Option<PyChunkedArray>,
    ub: Option<PyChunkedArray>,
    vtype: Option<PyChunkedArray>,
    names: Option<PyChunkedArray>,
    pool: Option<&Bound<'_, PyList>>,
    cls: Option<&Bound<'_, PyAny>>,
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
    let n_vars = c_int_len("variables", n)?;
    let call = || unsafe {
        (api.addvars)(
            model,
            n_vars,
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            opt_ptr(&obj),
            opt_ptr(&lb),
            opt_ptr(&ub),
            opt_ptr(&vtype),
            names.map_or(ptr::null(), <[_]>::as_ptr),
        )
    };
    // SAFETY: the call reads only `model` and the columns above through raw pointers and borrows;
    // they outlive it and nothing else touches them until it returns. It touches no Python.
    let call = unsafe { GurobiCall::new(call) };
    let rc = with_handles(cmodel, pool.zip(cls), n, call)?;
    api.check(cmodel.py(), model, rc)
}

/// Add one constraint per row of the List columns `vars` and `coefs`, which hold its terms. Their
/// Arrow offsets are the row starts of the CSR that `GRBXaddconstrs` takes, and their items the
/// variable indices and coefficients, so nothing is reshaped. `n_vars` is the model's variable
/// count, against which every index is checked.
#[pyfunction]
#[pyo3(signature = (cmodel, vars, coefs, sense, rhs, n_vars, names=None, pool=None, cls=None))]
#[allow(clippy::too_many_arguments)]
pub fn add_constrs(
    cmodel: &Bound<'_, PyAny>,
    vars: PyChunkedArray,
    coefs: PyChunkedArray,
    sense: PyChunkedArray,
    rhs: PyChunkedArray,
    n_vars: usize,
    names: Option<PyChunkedArray>,
    pool: Option<&Bound<'_, PyList>>,
    cls: Option<&Bound<'_, PyAny>>,
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
    // Merging repeats is pure Rust, so it runs with the Gurobi call, alongside handle creation.
    let call = || {
        let (cbeg, cind, cval) = match merge_repeats(cbeg, &cind, &cval, width) {
            Some((beg, ind, val)) => (Cow::Owned(beg), Cow::Owned(ind), Cow::Owned(val)),
            None => (
                Cow::Borrowed(cbeg),
                Cow::Borrowed(&*cind),
                Cow::Borrowed(&*cval),
            ),
        };
        unsafe {
            (api.xaddconstrs)(
                model,
                n_constrs,
                cind.len(),
                cbeg.as_ptr(),
                cind.as_ptr(),
                cval.as_ptr(),
                sense.as_ptr(),
                rhs.as_ptr(),
                names.map_or(ptr::null(), <[_]>::as_ptr),
            )
        }
    };
    // SAFETY: as in `add_vars`, the call only borrows `model` and the columns above, which outlive
    // it and are not touched elsewhere until it returns, and it touches no Python.
    let call = unsafe { GurobiCall::new(call) };
    let rc = with_handles(cmodel, pool.zip(cls), n, call)?;
    api.check(cmodel.py(), model, rc)
}

pub type Csr = (Vec<usize>, Vec<c_int>, Vec<f64>);

/// Why a CSR cannot go to Gurobi.
#[derive(Debug, PartialEq)]
pub enum Invalid {
    /// A row start decreases or lies beyond the items.
    Offsets,
    /// A variable index is negative or not below the model's variable count.
    Index(c_int),
}

/// Check what Gurobi would trust blindly or reject without saying why: the row starts `cbeg`,
/// which it reads without bounds checks, and the variable indices `cind` against the model's
/// `n_vars` variables. Returns the width `merge_repeats` needs, one past the largest index.
pub fn validate(cbeg: &[usize], cind: &[c_int], n_vars: usize) -> Result<usize, Invalid> {
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
pub fn merge_repeats(cbeg: &[usize], cind: &[c_int], cval: &[f64], width: usize) -> Option<Csr> {
    let rows = || row_bounds(cbeg, cind.len());
    // A few short rows against a wide model, the shape of an incremental build, are scanned in
    // place: a table as wide as the model would cost more to allocate than the rows to merge.
    let longest = rows().map(|(start, end)| end - start).max().unwrap_or(0);
    if longest <= SCAN_ROW_LEN && cind.len() < width / 16 {
        return merge_repeats_by_scan(rows, cind, cval);
    }
    // Slots are positions in the output, which has at most `cind.len()` terms.
    if u32::try_from(cind.len()).is_ok() {
        merge_repeats_by_table::<u32>(cbeg, cind, cval, width)
    } else {
        merge_repeats_by_table::<usize>(cbeg, cind, cval, width)
    }
}

/// A position in the output of `merge_repeats_by_table`, as narrow as the output allows.
trait Slot: Copy + Default {
    /// `at` fits, the caller has checked.
    fn from_usize(at: usize) -> Self;
    fn to_usize(self) -> usize;
}

impl Slot for u32 {
    fn from_usize(at: usize) -> Self {
        at as u32
    }
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl Slot for usize {
    fn from_usize(at: usize) -> Self {
        at
    }
    fn to_usize(self) -> usize {
        self
    }
}

/// `merge_repeats` with tables as wide as the model. `S` must hold every position below
/// `cind.len()`.
fn merge_repeats_by_table<S: Slot>(
    cbeg: &[usize],
    cind: &[c_int],
    cval: &[f64],
    width: usize,
) -> Option<Csr> {
    let rows = || row_bounds(cbeg, cind.len());
    // `seen[v]` is the last row that used column `v`, offset by one so zero means none. The
    // table is as wide as the model, so it is kept to four bytes per column.
    let mut seen = vec![0u32; width];
    let first = rows().enumerate().position(|(row, (start, end))| {
        let row = row as u32 + 1;
        cind[start..end]
            .iter()
            .any(|&v| std::mem::replace(&mut seen[v as usize], row) == row)
    })?;
    // The rows before `first` have nothing to merge and are copied as they are. The table is
    // still valid for the rows after `first`, which the merge has not seen yet; only the entries
    // written for `first` itself are stale, since detection stopped partway through it.
    let (first_start, first_end) = rows().nth(first).unwrap();
    for &v in &cind[first_start..first_end] {
        seen[v as usize] = 0;
    }
    let mut slot = vec![S::default(); width];
    let mut beg = Vec::with_capacity(cbeg.len());
    beg.extend_from_slice(&cbeg[..first]);
    let mut ind = Vec::with_capacity(cind.len());
    ind.extend_from_slice(&cind[..first_start]);
    let mut val = Vec::with_capacity(cval.len());
    val.extend_from_slice(&cval[..first_start]);
    for (row, (start, end)) in rows().enumerate().skip(first) {
        let row = row as u32 + 1;
        beg.push(ind.len());
        for (&v, &coef) in cind[start..end].iter().zip(&cval[start..end]) {
            let column = v as usize;
            if seen[column] == row {
                val[slot[column].to_usize()] += coef;
            } else {
                seen[column] = row;
                slot[column] = S::from_usize(ind.len());
                ind.push(v);
                val.push(coef);
            }
        }
    }
    Some((beg, ind, val))
}

/// The `(start, end)` of each row, for rows starting at `cbeg` in `nnz` terms.
fn row_bounds(cbeg: &[usize], nnz: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
    let ends = cbeg.iter().skip(1).copied().chain(std::iter::once(nnz));
    cbeg.iter().copied().zip(ends)
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
    fn merge_starts_at_first_repeating_row() {
        // Repeat in the first row.
        let merged = merge_repeats(&[0, 2], &[1, 1, 0, 2], &[1.0, 2.0, 3.0, 4.0], 3);
        assert_eq!(
            merged,
            Some((vec![0, 1], vec![1, 0, 2], vec![3.0, 3.0, 4.0]))
        );
        // Repeat in the last row; earlier rows are copied unchanged.
        let merged = merge_repeats(&[0, 2, 3], &[0, 1, 2, 1, 1], &[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(
            merged,
            Some((vec![0, 2, 3], vec![0, 1, 2, 1], vec![1.0, 2.0, 3.0, 9.0]))
        );
        // Repeat in a row that follows empty rows.
        let merged = merge_repeats(&[0, 1, 1, 1], &[2, 0, 0], &[1.0, 2.0, 3.0], 3);
        assert_eq!(merged, Some((vec![0, 1, 1, 1], vec![2, 0], vec![1.0, 5.0])));
        // A variable seen in an earlier row repeats inside the first repeating row: a table
        // reset that misses or over-reaches would merge it wrongly.
        let merged = merge_repeats(&[0, 2], &[0, 1, 0, 2, 0], &[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(
            merged,
            Some((vec![0, 2], vec![0, 1, 0, 2], vec![1.0, 2.0, 8.0, 4.0]))
        );
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
            let unchanged = || (cbeg.clone(), cind.clone(), cval.clone());
            match merge_repeats(&cbeg, &cind, &cval, width) {
                Some(got) => assert_eq!(got, want, "cbeg {cbeg:?} cind {cind:?}"),
                None => assert_eq!(unchanged(), want),
            }
            // The `usize` slots of outputs beyond `u32::MAX` terms, too long to build here.
            match merge_repeats_by_table::<usize>(&cbeg, &cind, &cval, width) {
                Some(got) => assert_eq!(got, want, "cbeg {cbeg:?} cind {cind:?}"),
                None => assert_eq!(unchanged(), want),
            }
        }
    }
}
