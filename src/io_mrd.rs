use std::path::Path;
use mrd_rs::MRD;
use num_complex::Complex32;
use crate::ArrayDim;

/// read data from a nrrd, either attached (.nrrd) or detached (.nhdr)
pub fn read_mrd(file:impl AsRef<Path>) -> (Vec<Complex32>, ArrayDim, MRD)
{
    let mrd = MRD::open(file);
    let dims = ArrayDim::from_shape(&mrd.dimensions());
    let data = mrd.complex_stream();
    (data, dims, mrd)
}