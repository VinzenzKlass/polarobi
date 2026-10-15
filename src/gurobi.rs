//! Function table into the Gurobi C library that gurobipy has already loaded.
//!
//! We never load a second copy of the library: two copies in one process would hold separate
//! global state and the `GRBmodel*` owned by gurobipy would be foreign to ours.

use std::ffi::{CStr, c_char, c_double, c_int, c_void};
use std::sync::OnceLock;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub type GrbModel = c_void;
type GrbEnv = c_void;

pub struct Api {
    _lib: libloading::Library,
    pub addvars: unsafe extern "C" fn(
        *mut GrbModel,
        c_int,
        c_int,
        *const c_int,
        *const c_int,
        *const c_double,
        *const c_double,
        *const c_double,
        *const c_double,
        *const c_char,
        *const *const c_char,
    ) -> c_int,
    pub xaddconstrs: unsafe extern "C" fn(
        *mut GrbModel,
        c_int,
        usize,
        *const usize,
        *const c_int,
        *const c_double,
        *const c_char,
        *const c_double,
        *const *const c_char,
    ) -> c_int,
    pub getdblattrlist:
        unsafe extern "C" fn(*mut GrbModel, *const c_char, c_int, *const c_int, *mut c_double) -> c_int,
    pub setdblattrlist: unsafe extern "C" fn(
        *mut GrbModel,
        *const c_char,
        c_int,
        *const c_int,
        *const c_double,
    ) -> c_int,
    pub getintattrlist:
        unsafe extern "C" fn(*mut GrbModel, *const c_char, c_int, *const c_int, *mut c_int) -> c_int,
    pub setintattrlist:
        unsafe extern "C" fn(*mut GrbModel, *const c_char, c_int, *const c_int, *const c_int) -> c_int,
    getenv: unsafe extern "C" fn(*mut GrbModel) -> *mut GrbEnv,
    geterrormsg: unsafe extern "C" fn(*mut GrbEnv) -> *const c_char,
    version: unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_int),
}

static API: OnceLock<Api> = OnceLock::new();

/// Bind to the already loaded Gurobi library at `path` and return its (major, minor, technical).
#[pyfunction]
pub fn load(path: &str) -> PyResult<(i32, i32, i32)> {
    if API.get().is_none() {
        let api = unsafe { Api::open(path) }.map_err(|e| {
            PyRuntimeError::new_err(format!("polarobi: cannot bind to {path}: {e}"))
        })?;
        let _ = API.set(api);
    }
    let api = api()?;
    let (mut major, mut minor, mut technical) = (0, 0, 0);
    unsafe { (api.version)(&mut major, &mut minor, &mut technical) };
    Ok((major, minor, technical))
}

pub fn api() -> PyResult<&'static Api> {
    API.get()
        .ok_or_else(|| PyRuntimeError::new_err("polarobi: Gurobi library is not bound"))
}

impl Api {
    unsafe fn open(path: &str) -> Result<Self, libloading::Error> {
        let lib = unsafe { open_already_loaded(path)? };
        macro_rules! sym {
            ($name:literal) => {
                *unsafe { lib.get(concat!($name, "\0").as_bytes())? }
            };
        }
        Ok(Self {
            addvars: sym!("GRBaddvars"),
            xaddconstrs: sym!("GRBXaddconstrs"),
            getdblattrlist: sym!("GRBgetdblattrlist"),
            setdblattrlist: sym!("GRBsetdblattrlist"),
            getintattrlist: sym!("GRBgetintattrlist"),
            setintattrlist: sym!("GRBsetintattrlist"),
            getenv: sym!("GRBgetenv"),
            geterrormsg: sym!("GRBgeterrormsg"),
            version: sym!("GRBversion"),
            _lib: lib,
        })
    }

    /// Map a Gurobi return code to `gurobipy.GurobiError`, carrying the env's last error message.
    pub fn check(&self, py: Python<'_>, model: *mut GrbModel, rc: c_int) -> PyResult<()> {
        if rc == 0 {
            return Ok(());
        }
        let msg = unsafe {
            let msg = (self.geterrormsg)((self.getenv)(model));
            if msg.is_null() { String::new() } else { CStr::from_ptr(msg).to_string_lossy().into_owned() }
        };
        let err = py.import("gurobipy")?.getattr("GurobiError")?.call1((rc, msg))?;
        Err(PyErr::from_value(err))
    }
}

#[cfg(windows)]
unsafe fn open_already_loaded(path: &str) -> Result<libloading::Library, libloading::Error> {
    libloading::os::windows::Library::open_already_loaded(path).map(Into::into)
}

#[cfg(unix)]
unsafe fn open_already_loaded(path: &str) -> Result<libloading::Library, libloading::Error> {
    let flags = libc::RTLD_NOW | libc::RTLD_LOCAL | libc::RTLD_NOLOAD;
    unsafe { libloading::os::unix::Library::open(Some(path), flags) }.map(Into::into)
}
