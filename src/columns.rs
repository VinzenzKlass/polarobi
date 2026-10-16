//! Zero-copy views of Arrow columns as the flat arrays the Gurobi C API expects.
//!
//! A single-chunk column is borrowed in place. A multi-chunk column is concatenated once, since
//! Gurobi needs one contiguous array. Nulls are rejected: Gurobi has no notion of a missing value.

use std::borrow::Cow;
use std::ffi::{CString, c_char, c_int};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, UInt8Type, UInt32Type};
use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, OffsetSizeTrait};
use arrow_schema::DataType;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyChunkedArray;

/// The chunks, verified to hold no nulls and, if given, `len` values in total.
fn checked<'a>(name: &str, chunks: &'a [ArrayRef], len: Option<usize>) -> PyResult<&'a [ArrayRef]> {
    if chunks.iter().any(|c| c.null_count() > 0) {
        return Err(PyValueError::new_err(format!(
            "column '{name}' contains nulls"
        )));
    }
    let n: usize = chunks.iter().map(|c| c.len()).sum();
    if let Some(len) = len.filter(|&len| len != n) {
        return Err(PyValueError::new_err(format!(
            "column '{name}' has length {n}, expected {len}"
        )));
    }
    Ok(chunks)
}

/// The values of primitive chunks as `U`, a type with the layout of `T::Native` for which
/// `convert` is the identity on bits, so that a single chunk is borrowed in place. Several chunks
/// are concatenated, converting each value.
fn values<'a, T, U>(chunks: &'a [ArrayRef], convert: impl Fn(T::Native) -> U) -> Cow<'a, [U]>
where
    T: ArrowPrimitiveType,
    U: Copy,
{
    const {
        assert!(size_of::<T::Native>() == size_of::<U>());
        assert!(align_of::<T::Native>() == align_of::<U>());
    }
    match chunks {
        [one] => {
            let v = one.as_primitive::<T>().values();
            // SAFETY: `U` has the layout of `T::Native`, and the callers pair plain numbers for
            // which every bit pattern is valid (`u32`/`i32`, `u8`/`i8`, `f64`/`f64`).
            Cow::Borrowed(unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len()) })
        }
        _ => {
            let mut out = Vec::with_capacity(chunks.iter().map(|c| c.len()).sum());
            for chunk in chunks {
                out.extend(
                    chunk
                        .as_primitive::<T>()
                        .values()
                        .iter()
                        .map(|&x| convert(x)),
                );
            }
            Cow::Owned(out)
        }
    }
}

fn type_error(name: &str, dtype: &DataType, expected: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "column '{name}' has type {dtype}, expected {expected}"
    ))
}

fn dtype(col: &PyChunkedArray) -> &DataType {
    col.field().data_type()
}

pub fn f64s<'a>(
    name: &str,
    col: &'a PyChunkedArray,
    len: Option<usize>,
) -> PyResult<Cow<'a, [f64]>> {
    f64s_of(name, dtype(col), col.chunks(), len)
}

fn f64s_of<'a>(
    name: &str,
    dtype: &DataType,
    chunks: &'a [ArrayRef],
    len: Option<usize>,
) -> PyResult<Cow<'a, [f64]>> {
    let chunks = checked(name, chunks, len)?;
    match dtype {
        DataType::Float64 => Ok(values::<Float64Type, f64>(chunks, |x| x)),
        dt => Err(type_error(name, dt, "Float64")),
    }
}

/// Gurobi indices are C ints. UInt32 (polars' row index type) is reinterpreted in place once it
/// is known to fit.
pub fn indices<'a>(
    name: &str,
    col: &'a PyChunkedArray,
    len: Option<usize>,
) -> PyResult<Cow<'a, [c_int]>> {
    indices_of(name, dtype(col), col.chunks(), len)
}

fn indices_of<'a>(
    name: &str,
    dtype: &DataType,
    chunks: &'a [ArrayRef],
    len: Option<usize>,
) -> PyResult<Cow<'a, [c_int]>> {
    let chunks = checked(name, chunks, len)?;
    match dtype {
        DataType::Int32 => Ok(values::<Int32Type, c_int>(chunks, |x| x)),
        DataType::UInt32 => {
            let fits = |c: &ArrayRef| {
                c.as_primitive::<UInt32Type>()
                    .values()
                    .iter()
                    .all(|&v| v <= c_int::MAX as u32)
            };
            if !chunks.iter().all(fits) {
                return Err(PyValueError::new_err(format!(
                    "column '{name}' exceeds the Gurobi index range"
                )));
            }
            Ok(values::<UInt32Type, c_int>(chunks, |x| x as c_int))
        }
        dt => Err(type_error(name, dt, "Int32 or UInt32")),
    }
}

/// Single-byte codes such as constraint senses or variable types, as C chars.
pub fn chars<'a>(
    name: &str,
    col: &'a PyChunkedArray,
    len: Option<usize>,
) -> PyResult<Cow<'a, [c_char]>> {
    let chunks = checked(name, col.chunks(), len)?;
    match dtype(col) {
        DataType::UInt8 => Ok(values::<UInt8Type, c_char>(chunks, |x| x as c_char)),
        dt => Err(type_error(name, dt, "UInt8")),
    }
}

/// Names must be NUL-terminated for C, so unlike the numeric columns these are always copied.
pub fn cstrings(name: &str, col: &PyChunkedArray, len: Option<usize>) -> PyResult<Vec<CString>> {
    let chunks = checked(name, col.chunks(), len)?;
    let mut out = Vec::with_capacity(len.unwrap_or_default());
    let mut push = |s: Option<&str>| -> PyResult<()> {
        let s =
            s.ok_or_else(|| PyValueError::new_err(format!("column '{name}' contains nulls")))?;
        let s = CString::new(s)
            .map_err(|_| PyValueError::new_err(format!("column '{name}' contains a NUL byte")))?;
        out.push(s);
        Ok(())
    };
    for chunk in chunks {
        match chunk.data_type() {
            DataType::Utf8View => chunk.as_string_view().iter().try_for_each(&mut push)?,
            DataType::Utf8 => chunk.as_string::<i32>().iter().try_for_each(&mut push)?,
            DataType::LargeUtf8 => chunk.as_string::<i64>().iter().try_for_each(&mut push)?,
            dt => return Err(type_error(name, dt, "String")),
        }
    }
    Ok(out)
}

/// A List column read as compressed sparse rows: row `i` holds the items at
/// `cbeg[i]..cbeg[i + 1]` of the flat item array, which is how Arrow stores lists already. A
/// single chunk whose offsets start at item 0 is borrowed as it is; otherwise the offsets are
/// rebased onto the concatenated items.
pub struct Lists<'a> {
    name: &'a str,
    item: &'a DataType,
    cbeg: Cow<'a, [usize]>,
    nnz: usize,
    items: Vec<ArrayRef>,
}

impl<'a> Lists<'a> {
    pub fn new(name: &'a str, col: &'a PyChunkedArray) -> PyResult<Self> {
        let chunks = checked(name, col.chunks(), None)?;
        match dtype(col) {
            DataType::LargeList(field) => Self::read::<i64>(name, field.data_type(), chunks),
            DataType::List(field) => Self::read::<i32>(name, field.data_type(), chunks),
            dt => Err(type_error(name, dt, "List")),
        }
    }

    fn read<O: OffsetSizeTrait>(
        name: &'a str,
        item: &'a DataType,
        chunks: &'a [ArrayRef],
    ) -> PyResult<Self> {
        const { assert!(size_of::<usize>() == size_of::<i64>()) };
        let bad = || PyValueError::new_err(format!("column '{name}' has invalid list offsets"));
        if let [one] = chunks
            && O::IS_LARGE
        {
            let list = one.as_list::<O>();
            let offsets = list.value_offsets();
            let (first, last) = (offsets[0].as_usize(), offsets[offsets.len() - 1].as_usize());
            if first == 0 {
                let rows = offsets.len() - 1;
                // SAFETY: the offsets are `i64` (asserted above) and non-negative by Arrow's
                // invariant, so they reinterpret as `usize`.
                let cbeg = unsafe { std::slice::from_raw_parts(offsets.as_ptr().cast(), rows) };
                return Ok(Self {
                    name,
                    item,
                    cbeg: Cow::Borrowed(cbeg),
                    nnz: last,
                    items: vec![list.values().slice(0, last)],
                });
            }
        }
        let rows = chunks.iter().map(|c| c.len()).sum();
        let (mut cbeg, mut nnz, mut items) = (Vec::with_capacity(rows), 0, Vec::new());
        for chunk in chunks {
            let list = chunk.as_list::<O>();
            let offsets = list.value_offsets();
            let (first, last) = (offsets[0].as_usize(), offsets[offsets.len() - 1].as_usize());
            for &offset in &offsets[..offsets.len() - 1] {
                cbeg.push(nnz + offset.as_usize().checked_sub(first).ok_or_else(bad)?);
            }
            let len = last.checked_sub(first).ok_or_else(bad)?;
            items.push(list.values().slice(first, len));
            nnz += len;
        }
        Ok(Self {
            name,
            item,
            cbeg: Cow::Owned(cbeg),
            nnz,
            items,
        })
    }

    /// The start of every row in the items; `nnz()` ends the last.
    pub fn cbeg(&self) -> &[usize] {
        &self.cbeg
    }

    pub fn rows(&self) -> usize {
        self.cbeg.len()
    }

    pub fn nnz(&self) -> usize {
        self.nnz
    }

    pub fn indices(&self) -> PyResult<Cow<'_, [c_int]>> {
        indices_of(self.name, self.item, &self.items, Some(self.nnz))
    }

    pub fn f64s(&self) -> PyResult<Cow<'_, [f64]>> {
        f64s_of(self.name, self.item, &self.items, Some(self.nnz))
    }
}
