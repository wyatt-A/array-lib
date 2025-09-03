/*
    Minimal library for working with column-major array layouts
    The number of dimensions is static for efficient address calculations
    This is useful for batched matrix calculations and image/signal processing routines
 */
#[cfg(feature = "io-nifti")]
pub mod io_nifti;

#[cfg(feature = "io-nrrd")]
pub mod io_nrrd;

#[cfg(feature = "io-nrrd")]
pub use nrrd_rs;

const N_DIMS:usize = 16;

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test() {
        // 10x10 matrix
        let dims = ArrayDim::new()
            .with_dim(0,4)
            .with_dim(1,3);
        let src = (0..dims.numel()).collect::<Vec<usize>>();
        let mut dst = dims.alloc(0);
        for i in 0..dims.size(0) {
            for j in 0..dims.size(1) {
                let addr = dims.calc_addr(&[i,j,0,0,0,0,0,0,0,0,0,0,0,0,0,0]);
                dst[addr] = src[addr];
            }
        }
        assert_eq!(src,dst);
    }

    #[test]
    fn test2() {

        let nx = 50;
        let ny = 50;
        let nz = 50;
        let src_dims = ArrayDim::new()
            .with_dim(0,nx)
            .with_dim(1,ny);
        let src = (0..src_dims.numel()).collect::<Vec<usize>>();
        let dst_dims = ArrayDim::new()
            .with_dim(0,nx)
            .with_dim(1,ny)
            .with_dim(2,nz);
        let mut dst = dst_dims.alloc(0);
        for i in 0..src_dims.size(0) {
            for j in 0..src_dims.size(1) {
                let src_addr = src_dims.calc_addr(&[i,j]);
                for k in 0..dst_dims.size(2) {
                    let dst_addr = dst_dims.calc_addr(&[i,j,k]);
                    dst[dst_addr] = src[src_addr];
                }
            }
        }
        let c = dst_dims.shape()[0..2].iter().product();
        assert_eq!(src,dst[0..c]);
    }

    #[test]
    fn test3() {
        let dims = ArrayDim::from_shape(&[3,4]);
        let idx = dims.calc_idx(3);
        let addr = dims.calc_addr(&idx);
        assert_eq!(addr,3);
    }

    #[test]
    fn test_shape_ns() {
        let dims = ArrayDim::from_shape(&[3,4,5,1,6]);
        let dns = dims.shape_ns();
        assert_eq!(dns,&[3,4,5,1,6]);
        let dims = ArrayDim::from_shape(&[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]);
        let dns = dims.shape_ns();
        assert_eq!(dns,&[1]);
    }

    #[test]
    fn t_fftshift() {
        let dims = ArrayDim::from_shape(&[6,4,5]);
        let coord = [0,0,0];
        let mut out = [0,0,0];
        dims.fft_shift_coords(&coord,&mut out);
        let mut inv = [0,0,0];
        dims.ifft_shift_coords(&out,&mut inv);
        assert_eq!(inv,coord);
    }

}

#[derive(Clone,Copy,Debug)]
pub struct ArrayDim {
    shape: [usize; N_DIMS],
    strides: [usize; N_DIMS],
}

impl ArrayDim {

    pub fn new() -> ArrayDim {
        ArrayDim{
            shape: [1;N_DIMS],
            strides: [1;N_DIMS],
        }
    }

    pub fn from_shape(shape: &[usize]) -> ArrayDim {

        let mut dims = [1;N_DIMS];
        let mut strides = [1;N_DIMS];

        for (d,s) in dims.iter_mut().zip(shape.iter()) {
            *d = *s;
        }

        Self::calc_strides(shape, &mut strides);
        Self {
            shape: dims,
            strides,
        }

    }

    /// return the shape with all singleton dimensions intact
    pub fn shape(&self) -> &[usize; N_DIMS] {
        &self.shape
    }

    /// return the shape with trailing singleton dimensions removed
    pub fn shape_ns(&self) -> &[usize] {
        if let Some(i) = self.shape.iter().rev().position(|&dim| dim != 1) {
            let new_len = self.shape.len() - i;
            &self.shape[..new_len]
        } else {
            // All dims are 1, return scalar shape (or empty, up to convention)
            &[1]
        }
    }

    pub fn size(&self, dim:usize) -> usize {
        assert!(dim < N_DIMS);
        self.shape[dim]
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn with_dim(mut self,axis:usize,dim:usize) -> ArrayDim {
        assert!(axis < N_DIMS,"only axes of up to 16 are supported");
        self.shape[axis] = dim;
        self.update_strides();
        self
    }

    fn calc_strides(dims:&[usize],strides:&mut [usize]) {
        let mut stride = 1;
        for (dim,s) in dims.iter().zip(strides.iter_mut()) {
            *s = stride;
            stride *= dim;
        }
    }

    fn update_strides(&mut self) {
        Self::calc_strides(&self.shape,&mut self.strides);
    }

    #[inline]
    /// calculate the element address from the index (subscripts)
    pub fn calc_addr(&self,idx: &[usize]) -> usize {
        let mut offset = 0;
        for (i,stride) in idx.iter().zip(self.strides.iter()) {
            offset += i * stride;
        }
        offset
    }

    #[inline]
    /// calculate the element index (subscript) from the address
    pub fn calc_idx(&self,addr:usize) -> [usize;16] {
        let mut addr = addr;
        let total: usize = self.shape.iter().product();
        debug_assert!(addr < total, "offset {} exceeds total number of elements {}", addr, total);
        let mut idx = [0usize; N_DIMS];
        for k in 0..N_DIMS {
            idx[k] = addr % self.shape[k];
            addr /= self.shape[k];
        }
        idx
    }

    /// allocates a vector of values the size of dims
    pub fn alloc<T:Sized + Clone>(&self,value:T) -> Vec<T> {
        vec![value;self.numel()]
    }

    #[inline]
    /// perform a forward fft shift of the input coordinates
    pub fn fft_shift_coords(&self,input: &[usize], out: &mut [usize]) {
        debug_assert!(input.len() <= N_DIMS);
        debug_assert!(out.len() <= N_DIMS);
        for ((o, &i), &d) in out.iter_mut().zip(input).zip(self.shape.iter()) {
            *o = (i + d / 2) % d;          // forward shift
        }
    }

    #[inline]
    /// perform an inverse fft shift of the input coordinates
    pub fn ifft_shift_coords(&self, input: &[usize], out: &mut [usize]) {
        debug_assert!(input.len() <= N_DIMS);
        debug_assert!(out.len() <= N_DIMS);
        for ((o, &i), &d) in out.iter_mut().zip(input).zip(self.shape.iter()) {
            *o = (i + (d + 1) / 2) % d;    // inverse shift
        }
    }

}

impl From<[usize;16]> for ArrayDim {
    fn from(shape:[usize;N_DIMS]) -> ArrayDim {
        let mut arr_dim = ArrayDim::new();
        for (ax,&dim) in shape.iter().enumerate() {
            arr_dim = arr_dim.with_dim(ax,dim);
        }
        arr_dim
    }
}