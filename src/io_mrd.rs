use std::path::Path;
use mrd_rs::MRD;
use num_complex::Complex32;
use crate::ArrayDim;

/// read data from an MRS MRD file. This also returns the file header
pub fn read_mrd(file:impl AsRef<Path>) -> (Vec<Complex32>, ArrayDim, MRD) {
    let mrd = MRD::open(file);
    let dims = ArrayDim::from_shape(&mrd.dimensions());
    let data = mrd.complex_stream();
    (data, dims, mrd)
}

/// read only the header for the MRD file
pub fn read_mrd_header(file: impl AsRef<Path>) -> MRD {
    MRD::open(file)
}

/// read partial MRD contents to a buffer with some offset from a file name
pub fn read_mrd_buffer_f(file: impl AsRef<Path>, offset:usize, buffer:&mut [Complex32]) -> std::io::Result<()> {
    let mrd = MRD::open(file);
    mrd.fill_buffer(buffer,offset)
}

/// read partial MRD contents to a buffer with some offset from an opened MRD
pub fn read_mrd_buffer(mrd:&MRD, offset:usize, buffer:&mut [Complex32]) -> std::io::Result<()> {
    mrd.fill_buffer(buffer,offset)
}