use std::path::Path;
use num_complex::Complex32;
use crate::ArrayDim;
use cfl;


pub fn read_cfl(cfl_file_base_name:impl AsRef<Path>) -> (Vec<Complex32>, ArrayDim)
{
    let cfl_dims = cfl::get_dims(&cfl_file_base_name).unwrap();
    let r = cfl::CflReader::new(cfl_file_base_name).unwrap();
    let dims = ArrayDim::from_shape(&cfl_dims);
    let mut data = vec![Complex32::ZERO; dims.numel()];
    r.read_slice(0,&mut data).unwrap();
    (data, dims)
}

pub fn write_cfl(cfl_file_base_name:impl AsRef<Path>, data: &[Complex32], dims: ArrayDim) {
    let mut w = cfl::CflWriter::new(cfl_file_base_name,dims.shape()).unwrap();
    w.write_slice(0, data).unwrap();
    w.flush().unwrap();
}

/// reads a contiguous slice from a cfl file. You must manually supply the starting offset and length of the
/// buffer to copy into
pub fn read_cfl_slice(cfl_file_base_name:impl AsRef<Path>,offset:usize, buff:&mut [Complex32]){
    let r = cfl::CflReader::new(&cfl_file_base_name).unwrap();
    r.read_slice(offset,buff).unwrap();
}