use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn wrap_err(err: ::lantern::Error) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{err:?}"))
}
