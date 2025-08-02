use std::path::Path;
use crate::ArrayDim;
use nrrd_rs::{read_nrrd_to, NRRD};
use nrrd_rs::header_defs::{NRRDType};
use num_traits::FromPrimitive;
pub use nrrd_rs::header_defs::Encoding;

/// read data from a nrrd, either attached (.nrrd) or detached (.nhdr)
pub fn read_nrrd<T>(file:impl AsRef<Path>) -> (Vec<T>, ArrayDim, NRRD)
where T:NRRDType + FromPrimitive
{
    let (data,nrrd) = read_nrrd_to(file);
    let dims = ArrayDim::from_shape(nrrd.shape());
    (data, dims, nrrd)
}

/// write a nrrd file from an array given a set of dimensions and an optional reference header.
/// The dimensions of the reference header must match the dimensions given.
pub fn write_nrrd<T>(file: impl AsRef<Path>, array:&[T], dims:ArrayDim, reference_header:Option<&NRRD>, attached:bool, encoding: Encoding)
where T:NRRDType
{
    assert_eq!(dims.numel(), array.len(), "data buffer and array dims must be consistent");
    if let Some(ref_header) = reference_header {
        assert_eq!(ref_header.shape(),dims.shape_ns(),"reference nhdr must have the same dimensionality as the array");
        nrrd_rs::write_nrrd(file, ref_header, array, attached, encoding);
    }else {
        let h = NRRD::new_from_dims::<T>(dims.shape_ns());
        nrrd_rs::write_nrrd(file, &h, array, attached, encoding);
    };
}