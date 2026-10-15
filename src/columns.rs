//! Zero-copy views of Arrow columns as the flat arrays the Gurobi C API expects.
//!
//! A single-chunk column is borrowed in place. A multi-chunk column is concatenated once, since
//! Gurobi needs one contiguous array. Nulls are rejected: Gurobi has no notion of a missing value.

use std::borrow::Cow;
use std::ffi::{CString, c_char, c_int};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, UInt8Type, UInt32Type, UInt64Type};
use arrow_array::{Array, ArrayRef, ArrowPrimitiveType};
use arrow_schema::DataType;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyChunkedArray;

fn checked<'a>(name: &str, col: &'a PyChunkedArray, len: Option<usize>) -> PyResult<&'a [ArrayRef]> {
    let chunks = col.chunks();
    if chunks.iter().any(|c| c.null_count() > 0) {
        return Err(PyValueError::new_err(format!("column '{name}' contains nulls")));
    }
    let n: usize = chunks.iter().map(|c| c.len()).sum();
    if let Some(len) = len.filter(|&len| len != n) {
        return Err(PyValueError::new_err(format!("column '{name}' has length {n}, expected {len}")));
    }
    Ok(chunks)
}

fn primitive<'a, T: ArrowPrimitiveType>(chunks: &'a [ArrayRef]) -> Cow<'a, [T::Native]> {
    match chunks {
        [one] => Cow::Borrowed(one.as_primitive::<T>().values()),
        _ => Cow::Owned(chunks.iter().flat_map(|c| c.as_primitive::<T>().values().iter().copied()).collect()),
    }
}

fn type_error(name: &str, dtype: &DataType, expected: &str) -> PyErr {
    PyTypeError::new_err(format!("column '{name}' has type {dtype}, expected {expected}"))
}

fn dtype(col: &PyChunkedArray) -> &DataType {
    col.field().data_type()
}

pub fn f64s<'a>(name: &str, col: &'a PyChunkedArray, len: Option<usize>) -> PyResult<Cow<'a, [f64]>> {
    let chunks = checked(name, col, len)?;
    match dtype(col) {
        DataType::Float64 => Ok(primitive::<Float64Type>(chunks)),
        dt => Err(type_error(name, dt, "Float64")),
    }
}

/// Gurobi indices are C ints. UInt32 (polars' row index type) is reinterpreted in place once it
/// is known to fit.
pub fn indices<'a>(name: &str, col: &'a PyChunkedArray, len: Option<usize>) -> PyResult<Cow<'a, [c_int]>> {
    let chunks = checked(name, col, len)?;
    match dtype(col) {
        DataType::Int32 => Ok(primitive::<Int32Type>(chunks)),
        DataType::UInt32 => {
            let values = primitive::<UInt32Type>(chunks);
            if values.iter().any(|&v| v > c_int::MAX as u32) {
                return Err(PyValueError::new_err(format!("column '{name}' exceeds the Gurobi index range")));
            }
            Ok(match values {
                Cow::Borrowed(v) => Cow::Borrowed(unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len()) }),
                Cow::Owned(v) => Cow::Owned(v.into_iter().map(|x| x as c_int).collect()),
            })
        }
        dt => Err(type_error(name, dt, "Int32 or UInt32")),
    }
}

/// Row offsets for `GRBXaddconstrs`, which takes `size_t`.
pub fn offsets<'a>(name: &str, col: &'a PyChunkedArray) -> PyResult<Cow<'a, [usize]>> {
    const { assert!(size_of::<usize>() == size_of::<u64>()) };
    let chunks = checked(name, col, None)?;
    match dtype(col) {
        DataType::UInt64 => Ok(match primitive::<UInt64Type>(chunks) {
            Cow::Borrowed(v) => Cow::Borrowed(unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len()) }),
            Cow::Owned(v) => Cow::Owned(v.into_iter().map(|x| x as usize).collect()),
        }),
        dt => Err(type_error(name, dt, "UInt64")),
    }
}

/// Single-byte codes such as constraint senses or variable types, as C chars.
pub fn chars<'a>(name: &str, col: &'a PyChunkedArray, len: Option<usize>) -> PyResult<Cow<'a, [c_char]>> {
    let chunks = checked(name, col, len)?;
    match dtype(col) {
        DataType::UInt8 => Ok(match primitive::<UInt8Type>(chunks) {
            Cow::Borrowed(v) => Cow::Borrowed(unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len()) }),
            Cow::Owned(v) => Cow::Owned(v.into_iter().map(|x| x as c_char).collect()),
        }),
        dt => Err(type_error(name, dt, "UInt8")),
    }
}

/// Names must be NUL-terminated for C, so unlike the numeric columns these are always copied.
pub fn cstrings(name: &str, col: &PyChunkedArray, len: Option<usize>) -> PyResult<Vec<CString>> {
    let chunks = checked(name, col, len)?;
    let mut out = Vec::with_capacity(len.unwrap_or_default());
    let mut push = |s: Option<&str>| -> PyResult<()> {
        let s = s.ok_or_else(|| PyValueError::new_err(format!("column '{name}' contains nulls")))?;
        let s = CString::new(s).map_err(|_| PyValueError::new_err(format!("column '{name}' contains a NUL byte")))?;
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
