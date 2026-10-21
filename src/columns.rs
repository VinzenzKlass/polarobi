//! Zero-copy views of Arrow columns as the flat arrays the Gurobi C API expects.
//!
//! A single-chunk column is borrowed in place. A multi-chunk column is concatenated once, since
//! Gurobi needs one contiguous array. Nulls are rejected: Gurobi has no notion of a missing value.

use std::borrow::Cow;
use std::ffi::{c_char, c_int};

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
fn values<T, U>(chunks: &'_ [ArrayRef], convert: impl Fn(T::Native) -> U) -> Cow<'_, [U]>
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

pub fn f64s_of<'a>(
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

pub fn indices_of<'a>(
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
    chars_of(name, dtype(col), col.chunks(), len)
}

fn chars_of<'a>(
    name: &str,
    dtype: &DataType,
    chunks: &'a [ArrayRef],
    len: Option<usize>,
) -> PyResult<Cow<'a, [c_char]>> {
    let chunks = checked(name, chunks, len)?;
    match dtype {
        DataType::UInt8 => Ok(values::<UInt8Type, c_char>(chunks, |x| x as c_char)),
        dt => Err(type_error(name, dt, "UInt8")),
    }
}

/// NUL-terminated copies of a String column, packed into one buffer. Names must be terminated
/// for C, so unlike the numeric columns these are always copied, but in one allocation rather
/// than one per name.
#[derive(Debug)]
pub struct CStrings {
    /// Only read through `ptrs` outside tests, but owns the memory they point into.
    #[allow(dead_code)]
    bytes: Vec<u8>,
    /// A pointer into `bytes` for each name. `bytes` never reallocates after packing, and moving
    /// `self` leaves its heap buffer in place, so these stay valid for as long as `self` is.
    ptrs: Vec<*const c_char>,
}

impl CStrings {
    /// A pointer to each name, in the form the `char **` arguments of the Gurobi API take. The
    /// pointers are valid for as long as `self` is.
    pub fn ptrs(&self) -> &[*const c_char] {
        &self.ptrs
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.ptrs.len()
    }

    #[cfg(test)]
    fn get(&self, i: usize) -> &std::ffi::CStr {
        // SAFETY: each pointer is to a NUL-terminated name inside `self.bytes`.
        unsafe { std::ffi::CStr::from_ptr(self.ptrs[i]) }
    }
}

pub fn cstrings(name: &str, col: &PyChunkedArray, len: Option<usize>) -> PyResult<CStrings> {
    cstrings_of(name, col.chunks(), len)
}

pub fn cstrings_of(name: &str, chunks: &[ArrayRef], len: Option<usize>) -> PyResult<CStrings> {
    let chunks = checked(name, chunks, len)?;
    let rows: usize = chunks.iter().map(|c| c.len()).sum();
    // Exactly the bytes of the names plus a terminator each. A sliced string-view chunk still
    // points at its parent's data buffers, so buffer sizes would over-reserve.
    // The pointers taken below rely on `bytes` never reallocating, so it is reserved at its exact
    // final size up front.
    let mut bytes: Vec<u8> = Vec::with_capacity(chunks.iter().map(text_len).sum::<usize>() + rows);
    let mut ptrs = Vec::with_capacity(rows);
    let mut push = |s: Option<&str>| -> PyResult<()> {
        let s =
            s.ok_or_else(|| PyValueError::new_err(format!("column '{name}' contains nulls")))?;
        if s.as_bytes().contains(&0) {
            return Err(PyValueError::new_err(format!(
                "column '{name}' contains a NUL byte"
            )));
        }
        ptrs.push(bytes.as_ptr_range().end.cast());
        bytes.extend_from_slice(s.as_bytes());
        bytes.push(0);
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
    debug_assert_eq!(
        bytes.len(),
        bytes.capacity(),
        "names buffer was not reserved exactly"
    );
    Ok(CStrings { bytes, ptrs })
}

/// The total length of the strings in a chunk, counting only the rows the chunk holds. Zero for
/// types that are not strings, which `cstrings_of` rejects afterwards.
fn text_len(chunk: &ArrayRef) -> usize {
    match chunk.data_type() {
        // The low 32 bits of a view are the string's length, inlined or not.
        DataType::Utf8View => chunk
            .as_string_view()
            .views()
            .iter()
            .map(|&v| v as u32 as usize)
            .sum(),
        DataType::Utf8 => {
            let offsets = chunk.as_string::<i32>().value_offsets();
            (offsets[offsets.len() - 1] - offsets[0]) as usize
        }
        DataType::LargeUtf8 => {
            let offsets = chunk.as_string::<i64>().value_offsets();
            (offsets[offsets.len() - 1] - offsets[0]) as usize
        }
        _ => 0,
    }
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
        Self::of(name, dtype(col), col.chunks())
    }

    fn of(name: &'a str, dtype: &'a DataType, chunks: &'a [ArrayRef]) -> PyResult<Self> {
        let chunks = checked(name, chunks, None)?;
        match dtype {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::Int64Type;
    use arrow_array::{
        Float64Array, Int32Array, Int64Array, LargeListArray, LargeStringArray, ListArray,
        StringArray, StringViewArray, UInt8Array, UInt32Array,
    };

    use super::*;

    /// The error as Python shows it, e.g. "ValueError: column 'x' contains nulls".
    fn message(err: PyErr) -> String {
        Python::initialize();
        err.to_string()
    }

    fn arrays(chunks: impl IntoIterator<Item = impl Array + 'static>) -> Vec<ArrayRef> {
        chunks
            .into_iter()
            .map(|c| Arc::new(c) as ArrayRef)
            .collect()
    }

    #[test]
    fn single_chunk_is_borrowed_in_place() {
        let chunks = arrays([Int32Array::from(vec![3, 1, 2])]);
        let got = indices_of("i", &DataType::Int32, &chunks, Some(3)).unwrap();
        assert!(matches!(got, Cow::Borrowed(_)));
        assert_eq!(&*got, &[3, 1, 2]);
        let chunks = arrays([UInt32Array::from(vec![0, c_int::MAX as u32])]);
        let got = indices_of("i", &DataType::UInt32, &chunks, None).unwrap();
        assert!(matches!(got, Cow::Borrowed(_)));
        assert_eq!(&*got, &[0, c_int::MAX]);
    }

    #[test]
    fn chunks_are_concatenated() {
        let chunks = arrays([
            UInt32Array::from(vec![1, 2]),
            UInt32Array::from(Vec::<u32>::new()),
            UInt32Array::from(vec![3]),
        ]);
        let got = indices_of("i", &DataType::UInt32, &chunks, Some(3)).unwrap();
        assert!(matches!(got, Cow::Owned(_)));
        assert_eq!(&*got, &[1, 2, 3]);
        let chunks = arrays([
            Float64Array::from(vec![-0.0]),
            Float64Array::from(vec![f64::INFINITY, 1.5]),
        ]);
        let got = f64s_of("f", &DataType::Float64, &chunks, Some(3)).unwrap();
        assert_eq!(got[0].to_bits(), (-0.0f64).to_bits());
        assert_eq!(&got[1..], &[f64::INFINITY, 1.5]);
        let (lt, eq) = (60u8, 61u8);
        let chunks = arrays([UInt8Array::from(vec![lt]), UInt8Array::from(vec![eq, 255])]);
        let got = chars_of("c", &DataType::UInt8, &chunks, Some(3)).unwrap();
        assert_eq!(&*got, &[lt as c_char, eq as c_char, 255u8 as c_char]);
        assert!(
            f64s_of("f", &DataType::Float64, &[], Some(0))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn indices_beyond_c_int_are_rejected() {
        // In a later chunk too, where the values are copied rather than borrowed.
        let chunks = arrays([
            UInt32Array::from(vec![0]),
            UInt32Array::from(vec![c_int::MAX as u32 + 1]),
        ]);
        let err = indices_of("ind", &DataType::UInt32, &chunks, None).unwrap_err();
        assert_eq!(
            message(err),
            "ValueError: column 'ind' exceeds the Gurobi index range"
        );
    }

    #[test]
    fn nulls_lengths_and_types_are_checked() {
        let chunks = arrays([Float64Array::from(vec![Some(1.0), None])]);
        let err = f64s_of("lb", &DataType::Float64, &chunks, None).unwrap_err();
        assert_eq!(message(err), "ValueError: column 'lb' contains nulls");
        let chunks = arrays([Float64Array::from(vec![1.0, 2.0])]);
        let err = f64s_of("lb", &DataType::Float64, &chunks, Some(3)).unwrap_err();
        assert_eq!(
            message(err),
            "ValueError: column 'lb' has length 2, expected 3"
        );
        let chunks = arrays([Int64Array::from(vec![1])]);
        let err = indices_of("ind", &DataType::Int64, &chunks, None).unwrap_err();
        assert_eq!(
            message(err),
            "TypeError: column 'ind' has type Int64, expected Int32 or UInt32"
        );
        let err = f64s_of("lb", &DataType::Int64, &chunks, None).unwrap_err();
        assert_eq!(
            message(err),
            "TypeError: column 'lb' has type Int64, expected Float64"
        );
        let err = chars_of("vtype", &DataType::Int64, &chunks, None).unwrap_err();
        assert_eq!(
            message(err),
            "TypeError: column 'vtype' has type Int64, expected UInt8"
        );
    }

    #[test]
    fn strings_of_every_layout() {
        let long = "a name longer than twelve bytes"; // not inlined in a string view
        let chunks: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec!["x[1]", ""])),
            Arc::new(LargeStringArray::from(vec!["\u{fc}"])),
            Arc::new(StringViewArray::from(vec![long])),
        ];
        let got = cstrings_of("names", &chunks, Some(4)).unwrap();
        let strs: Vec<&str> = (0..got.len())
            .map(|i| got.get(i).to_str().unwrap())
            .collect();
        assert_eq!(strs, ["x[1]", "", "\u{fc}", long]);
        let ptrs = got.ptrs();
        assert_eq!(ptrs.len(), 4);
        // Each pointer reads as its name, NUL-terminated.
        let via_ptr = unsafe { std::ffi::CStr::from_ptr(ptrs[3]) };
        assert_eq!(via_ptr.to_str().unwrap(), long);
        // The buffer was sized exactly: the names' bytes plus one terminator each.
        let want = "x[1]".len() + "\u{fc}".len() + long.len() + 4;
        assert_eq!(got.bytes.len(), want);
        assert_eq!(got.bytes.capacity(), want);
    }

    #[test]
    fn name_pointers_survive_a_move() {
        let chunks = arrays([StringArray::from(vec!["a", "bc", "def"])]);
        let got = cstrings_of("names", &chunks, Some(3)).unwrap();
        let before = got.ptrs().to_vec();
        let moved = Box::new(got);
        assert_eq!(moved.ptrs(), &before[..]);
        let strs: Vec<&str> = (0..moved.len())
            .map(|i| moved.get(i).to_str().unwrap())
            .collect();
        assert_eq!(strs, ["a", "bc", "def"]);
    }

    #[test]
    fn empty_names_get_their_own_terminators() {
        let chunks = arrays([StringArray::from(vec!["", "", ""])]);
        let got = cstrings_of("names", &chunks, Some(3)).unwrap();
        assert_eq!(got.bytes, [0, 0, 0]);
        let ptrs = got.ptrs();
        assert_eq!(ptrs.len(), 3);
        for (i, &p) in ptrs.iter().enumerate() {
            assert_eq!(p, got.bytes[i..].as_ptr().cast());
            assert!(got.get(i).is_empty());
        }
    }

    #[test]
    fn packed_names_are_sized_by_the_slice_not_the_parent() {
        let long = "a name longer than twelve bytes, held in a data buffer";
        let parent: Vec<&str> = std::iter::repeat_n(long, 1000).collect();
        let chunks = arrays([StringViewArray::from(parent.clone()).slice(10, 2)]);
        let got = cstrings_of("names", &chunks, Some(2)).unwrap();
        assert_eq!(got.bytes.capacity(), 2 * (long.len() + 1));
        let chunks = arrays([StringArray::from(parent).slice(10, 2)]);
        let got = cstrings_of("names", &chunks, Some(2)).unwrap();
        assert_eq!(got.bytes.capacity(), 2 * (long.len() + 1));
    }

    #[test]
    fn bad_strings_are_rejected() {
        let chunks = arrays([StringArray::from(vec!["a\0b"])]);
        let err = cstrings_of("names", &chunks, None).unwrap_err();
        assert_eq!(
            message(err),
            "ValueError: column 'names' contains a NUL byte"
        );
        let chunks = arrays([StringArray::from(vec![Some("a"), None])]);
        let err = cstrings_of("names", &chunks, None).unwrap_err();
        assert_eq!(message(err), "ValueError: column 'names' contains nulls");
        let chunks = arrays([Int32Array::from(vec![1])]);
        let err = cstrings_of("names", &chunks, None).unwrap_err();
        assert_eq!(
            message(err),
            "TypeError: column 'names' has type Int32, expected String"
        );
        let chunks = arrays([StringArray::from(vec!["a"])]);
        let err = cstrings_of("names", &chunks, Some(2)).unwrap_err();
        assert_eq!(
            message(err),
            "ValueError: column 'names' has length 1, expected 2"
        );
    }

    type Rows = Vec<Option<Vec<Option<u32>>>>;

    fn rows(rows: &[&[u32]]) -> Rows {
        rows.iter()
            .map(|r| Some(r.iter().copied().map(Some).collect()))
            .collect()
    }

    fn large(r: Rows) -> LargeListArray {
        LargeListArray::from_iter_primitive::<UInt32Type, _, _>(r)
    }

    fn small(r: Rows) -> ListArray {
        ListArray::from_iter_primitive::<UInt32Type, _, _>(r)
    }

    fn read(chunks: &[ArrayRef]) -> PyResult<Lists<'_>> {
        Lists::of("vars", chunks[0].data_type(), chunks)
    }

    /// The lists as the CSR they read as: row starts, item count and items.
    fn csr(lists: &Lists<'_>) -> (Vec<usize>, usize, Vec<c_int>) {
        let items = lists.indices().unwrap().into_owned();
        (lists.cbeg().to_vec(), lists.nnz(), items)
    }

    #[test]
    fn large_list_from_zero_is_borrowed() {
        let chunks = arrays([large(rows(&[&[0, 1], &[], &[2]]))]);
        let lists = read(&chunks).unwrap();
        assert!(matches!(lists.cbeg, Cow::Borrowed(_)));
        assert_eq!(lists.rows(), 3);
        assert_eq!(csr(&lists), (vec![0, 2, 2], 3, vec![0, 1, 2]));
    }

    #[test]
    fn sliced_lists_are_rebased() {
        let chunks = arrays([large(rows(&[&[9, 9], &[1], &[2, 3], &[9]])).slice(1, 2)]);
        let lists = read(&chunks).unwrap();
        assert!(matches!(lists.cbeg, Cow::Owned(_)));
        assert_eq!(csr(&lists), (vec![0, 1], 3, vec![1, 2, 3]));
    }

    #[test]
    fn small_offsets_are_widened() {
        let chunks = arrays([small(rows(&[&[4], &[5, 6]]))]);
        assert_eq!(csr(&read(&chunks).unwrap()), (vec![0, 1], 3, vec![4, 5, 6]));
        let chunks = arrays([small(rows(&[&[7], &[8, 9], &[]])).slice(1, 2)]);
        assert_eq!(csr(&read(&chunks).unwrap()), (vec![0, 2], 2, vec![8, 9]));
    }

    #[test]
    fn chunks_of_lists_are_joined() {
        let chunks = arrays([
            large(rows(&[&[0]])),
            large(rows(&[])),
            large(rows(&[&[9], &[1, 2], &[]])).slice(1, 2),
            large(rows(&[&[], &[3]])),
        ]);
        let lists = read(&chunks).unwrap();
        assert_eq!(lists.rows(), 5);
        assert_eq!(csr(&lists), (vec![0, 1, 3, 3, 3], 4, vec![0, 1, 2, 3]));
    }

    #[test]
    fn list_items_convert_like_columns() {
        let r: Vec<Option<Vec<Option<f64>>>> = vec![Some(vec![Some(1.5)]), Some(vec![Some(-2.0)])];
        let chunks = arrays([LargeListArray::from_iter_primitive::<Float64Type, _, _>(r)]);
        assert_eq!(&*read(&chunks).unwrap().f64s().unwrap(), &[1.5, -2.0]);
        let err = read(&chunks).unwrap().indices().unwrap_err();
        assert_eq!(
            message(err),
            "TypeError: column 'vars' has type Float64, expected Int32 or UInt32"
        );
        let r: Vec<Option<Vec<Option<i64>>>> = vec![Some(vec![Some(1)])];
        let chunks = arrays([LargeListArray::from_iter_primitive::<Int64Type, _, _>(r)]);
        let err = read(&chunks).unwrap().f64s().unwrap_err();
        assert_eq!(
            message(err),
            "TypeError: column 'vars' has type Int64, expected Float64"
        );
    }

    #[test]
    fn null_lists_and_items_are_rejected() {
        let chunks = arrays([large(vec![Some(vec![Some(0)]), None])]);
        let err = read(&chunks).err().unwrap();
        assert_eq!(message(err), "ValueError: column 'vars' contains nulls");
        let chunks = arrays([large(vec![Some(vec![Some(0), None])])]);
        let err = read(&chunks).unwrap().indices().unwrap_err();
        assert_eq!(message(err), "ValueError: column 'vars' contains nulls");
    }

    #[test]
    fn non_lists_are_rejected() {
        let chunks = arrays([UInt32Array::from(vec![0])]);
        let err = read(&chunks).err().unwrap();
        assert_eq!(
            message(err),
            "TypeError: column 'vars' has type UInt32, expected List"
        );
    }
}
