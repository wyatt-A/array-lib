use std::path::Path;
use bytemuck::Pod;
use nifti;
use nifti::{DataElement, InMemNiftiVolume, IntoNdArray, NiftiHeader, NiftiObject, NiftiType, NiftiVolume};
use ndarray;
use ndarray::ShapeBuilder;
use num_complex::Complex;
use crate::ArrayDim;
use num_traits::{Num, NumCast, ToPrimitive, Zero};


#[cfg(test)]
mod tests {
    use num_complex::{Complex32, Complex64};
    use crate::ArrayDim;
    use crate::io_nifti::{read_nifti_complex, read_nifti, write_nifti};

    #[test]
    fn test_io_nifti() {
        let dims = ArrayDim::from_shape(&[10,5,4,3,3]);

        // real to real
        let x = dims.alloc(1f32);
        write_nifti("test",&x,dims);
        let (data,..) = read_nifti::<f32>("test.nii");
        std::fs::remove_file("test.nii").unwrap();
        assert_eq!(x,data);

        // real to real
        let x = dims.alloc(1f64);
        let exp = dims.alloc(1f32);
        write_nifti("test",&x,dims);
        let (data,..) = read_nifti::<f32>("test.nii");
        std::fs::remove_file("test.nii").unwrap();
        assert_eq!(exp,data);

        // complex to complex
        let x = dims.alloc(Complex32::ONE);
        write_nifti("test",&x,dims);
        let (data,..) = read_nifti_complex::<f32>("test.nii");
        std::fs::remove_file("test.nii").unwrap();
        assert_eq!(x,data);

        // complex to complex
        let x = dims.alloc(Complex64::ONE);
        write_nifti("test",&x,dims);
        let (data,..) = read_nifti_complex::<f64>("test.nii");
        std::fs::remove_file("test.nii").unwrap();
        assert_eq!(x,data);

        // real to complex
        let x = dims.alloc(1f32);
        let exp = dims.alloc(Complex32::ONE);
        write_nifti("test",&x,dims);
        let (data,..) = read_nifti_complex::<f32>("test.nii");
        std::fs::remove_file("test.nii").unwrap();
        assert_eq!(exp,data);

        // complex to real
        let x = dims.alloc(Complex32::new(1f32, 1f32));
        let exp = dims.alloc(1f32);
        write_nifti("test",&x,dims);
        let (data,..) = read_nifti::<f32>("test.nii");
        std::fs::remove_file("test.nii").unwrap();
        assert_eq!(exp,data);

    }

}

/// read data from a nifti file assumed to be storing real data. If the data is complex, then only
/// the real part is read. The returns the data as a vec, an array dimension helper type, and the
/// nifti header
pub fn read_nifti<T:ToPrimitive + NumCast + 'static + Pod>(file:impl AsRef<Path>) -> (Vec<T>, ArrayDim, NiftiHeader) {

    let nii = nifti::ReaderOptions::new().read_file(file.as_ref()).expect(&format!("failed to read nifti :{}",file.as_ref().display()));
    let nii_header = nii.header().clone();
    let volume = nii.into_volume();

    let dims:Vec<_> = volume.dim().iter().map(|&dim| dim as usize).collect();
    let dims = ArrayDim::from_shape(&dims);

    let data:Vec<T> = match volume.data_type() {
        NiftiType::Uint8 => cast_data::<u8, T>(volume),
        NiftiType::Int16 => cast_data::<i16, T>(volume),
        NiftiType::Int32 => cast_data::<i32, T>(volume),
        NiftiType::Float32 => cast_data::<f32, T>(volume),
        NiftiType::Float64 => cast_data::<f64, T>(volume),
        NiftiType::Int8 => cast_data::<i8, T>(volume),
        NiftiType::Uint16 => cast_data::<u16, T>(volume),
        NiftiType::Uint32 => cast_data::<u32, T>(volume),
        NiftiType::Int64 => cast_data::<i64, T>(volume),
        NiftiType::Uint64 => cast_data::<u64, T>(volume),
        NiftiType::Complex64 => {
            println!("WARNING: reading only real component from Complex32: {}",file.as_ref().display());
            extract_real(cast_complex_data::<f32, T>(volume))
        } ,
        NiftiType::Complex128 => {
            println!("WARNING: reading only real component from Complex64: {}",file.as_ref().display());
            extract_real(cast_complex_data::<f64, T>(volume))
        } ,
        NiftiType::Rgba32 => panic!("Rgba32 not supported for now."),
        NiftiType::Float128 => panic!("Float128 not supported."),
        NiftiType::Rgb24 => panic!("Rgb24 not supported for now."),
        NiftiType::Complex256 => panic!("Complex256 not supported."),
    };

    (data,dims,nii_header)

}

/// read data from a nifti file assumed to be storing complex data. If the data is real, then the imaginary
/// component is set to 0. The returns the data as a vec, an array dimension helper type, and the
/// nifti header
pub fn read_nifti_complex<T:ToPrimitive + Zero + NumCast + 'static + Pod>(file:impl AsRef<Path>) -> (Vec<Complex<T>>, ArrayDim, NiftiHeader) {

    let nii = nifti::ReaderOptions::new().read_file(file.as_ref()).expect(&format!("failed to read nifti :{}",file.as_ref().display()));
    let nii_header = nii.header().clone();
    let volume = nii.into_volume();

    let dims:Vec<_> = volume.dim().iter().map(|&dim| dim as usize).collect();
    let dims = ArrayDim::from_shape(dims.as_slice());

    let data:Vec<Complex<T>> = match volume.data_type() {
        NiftiType::Uint8 => convert_real(cast_data::<u8, T>(volume)),
        NiftiType::Int16 => convert_real(cast_data::<i16, T>(volume)),
        NiftiType::Int32 => convert_real(cast_data::<i32, T>(volume)),
        NiftiType::Float32 => convert_real(cast_data::<f32, T>(volume)),
        NiftiType::Float64 => convert_real(cast_data::<f64, T>(volume)),
        NiftiType::Int8 => convert_real(cast_data::<i8, T>(volume)),
        NiftiType::Uint16 => convert_real(cast_data::<u16, T>(volume)),
        NiftiType::Uint32 => convert_real(cast_data::<u32, T>(volume)),
        NiftiType::Int64 => convert_real(cast_data::<i64, T>(volume)),
        NiftiType::Uint64 => convert_real(cast_data::<u64, T>(volume)),
        NiftiType::Complex64 => cast_complex_data::<f32, T>(volume),
        NiftiType::Complex128 => cast_complex_data::<f64, T>(volume),
        NiftiType::Rgba32 => panic!("Rgba32 not supported for now."),
        NiftiType::Float128 => panic!("Float128 not supported."),
        NiftiType::Rgb24 => panic!("Rgb24 not supported for now."),
        NiftiType::Complex256 => panic!("Complex256 not supported."),
    };
    (data,dims,nii_header)
}

/// write a nifti file from a raw data array and a set of dimensions. If the number of dimensions
/// is greater than 4, the remaining dims will be flattened into the 4th dimension
pub fn write_nifti<T>(file: impl AsRef<Path>, array:&[T], dims:ArrayDim)
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

/// write a nifti file from a raw data array and a set of dimensions. If the number of dimensions
/// is greater than 4, the remaining dims will be flattened into the 4th dimension. The header will
/// be modified according to a reference header
pub fn write_nifti_with_header<T>(file: impl AsRef<Path>, array:&[T], dims:ArrayDim, ref_header:&NiftiHeader)
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
    let writer = nifti::writer::WriterOptions::new(file.as_ref().with_extension("nii")).reference_header(ref_header);
    writer.write_nifti(&arr).expect("failed to write nifti");
}

fn cast_data<N, T>(volume:InMemNiftiVolume)
                   -> Vec<T>
where
    N: ToPrimitive +  DataElement + 'static,
    T: NumCast + 'static,
{
    let typed = volume
        .into_nifti_typed_data::<N>()
        .expect("Failed to convert to typed volume");

    typed
        .into_iter()
        .map(|x| NumCast::from(x).expect("Failed to cast value"))
        .collect()
}

fn cast_complex_data<N, T>(volume: InMemNiftiVolume) -> Vec<Complex<T>>
where
    N: DataElement + ToPrimitive + Zero + 'static,
    T: NumCast + 'static + Copy + Pod,
{


    match volume.data_type() {
        NiftiType::Complex64 => (),
        NiftiType::Complex128 => (),
        NiftiType::Complex256 => (),
        _=> assert!(false,"volume is not complex"),
    }

    // 1. Interpret raw buffer as real-valued N data
    // let raw = volume
    //     .into_nifti_typed_data::<N>()
    //     .expect("Failed to convert volume to raw complex buffer");

    let raw = volume.into_raw_data();
    let raw = bytemuck::cast_slice::<u8, T>(&raw).to_vec();

    // 2. Chunk into real-imag pairs
    raw.chunks(2)
        .map(|chunk| {
            let re = chunk.get(0).copied().unwrap();
            let im = chunk.get(1).copied().unwrap();
            let re_t = NumCast::from(re).expect("Failed to cast real part");
            let im_t = NumCast::from(im).expect("Failed to cast imag part");
            Complex::new(re_t, im_t)
        })
        .collect()
}

fn convert_real<T:ToPrimitive + Zero>(x:Vec<T>) -> Vec<Complex<T>> {
    x.into_iter().map(|x| Complex::new(x,T::zero())).collect()
}

fn extract_real<T:Sized>(x:Vec<Complex<T>>) -> Vec<T> {
    x.into_iter().map(|x| x.re).collect()
}


