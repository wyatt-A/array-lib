use std::path::Path;
use bytemuck::Pod;
use nifti;
use nifti::{DataElement, IntoNdArray, NiftiHeader, NiftiObject, NiftiType, NiftiVolume};
use ndarray;
use ndarray::ShapeBuilder;
use crate::ArrayDim;


#[cfg(test)]
mod tests {
    use num_complex::Complex32;
    use crate::ArrayDim;
    use crate::io_nifti::{read_nifti_to_array, write_nifti_from_array};

    #[test]
    fn test_io_nifti() {
        let dims = ArrayDim::from_shape(&[10,5,4,3,3]);
        let x = dims.alloc(Complex32::ONE);
        write_nifti_from_array("test_complex",&x,dims);
        let (..,dims,data) = read_nifti_to_array::<Complex32>("test_complex.nii");
        std::fs::remove_file("test_complex.nii").unwrap();
        assert_eq!(x,data);
    }

}


pub fn read_nifti_to_array<T>(file:impl AsRef<Path>) -> (NiftiHeader,ArrayDim,Vec<T>)
where T:Sized + DataElement
{
    let nii = nifti::ReaderOptions::new().read_file(file.as_ref()).expect("failed to read nifti file");
    let nii_header = nii.header().clone();
    let volume = nii.into_volume();
    let dims:Vec<_> = volume.dim().iter().map(|&dim| dim as usize).collect();
    let data:Vec<T> = volume.into_nifti_typed_data().expect("failed to retrieve typed nifti data");
    (nii_header, ArrayDim::from_shape(&dims), data)
}

pub fn write_nifti_from_array<T>(file: impl AsRef<Path>, array:&[T], dims:ArrayDim)
where T:Sized + DataElement + Pod
{
    assert_eq!(dims.numel(), array.len(), "data buffer and array dims must be consistent");
    // collapse any dims above 3 into the 4th dim
    let dim4:usize = dims.shape()[3..].iter().product();
    let arr = if dim4 > 1 {
        ndarray::Array::from_shape_vec([dims.size(0),dims.size(1),dims.size(2),dim4].as_slice().f(), array.to_vec()).unwrap()
    }else {
        ndarray::Array::from_shape_vec([dims.size(0),dims.size(1),dims.size(2)].as_slice().f(), array.to_vec()).unwrap()
    };
    let writer = nifti::writer::WriterOptions::new(file.as_ref().with_extension("nii"));
    writer.write_nifti(&arr).expect("failed to write nifti");
}