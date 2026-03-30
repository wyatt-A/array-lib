use std::fmt::Display;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use num_complex::{Complex, Complex32};
use serde::{Deserialize, Serialize};
use crate::{io_bruker, io_cfl, ArrayDim};
use bruker_jcamp_rs::{parse_paravision_params, PvError};
use rayon::prelude::*;




#[test]
fn test_fill_buffer() {
    let acq_dir = r"/Users/Wyatt/scratch/btest/7";
    let buff_dims = ArrayDim::from_shape(&[600,45000]);
    let offset = 13 * buff_dims.numel();
    let mut buffer = buff_dims.alloc(Complex32::ZERO);
    read_bruker_buffer(acq_dir, offset, &mut buffer).unwrap();
}




// 1kB block size
const BLOCK_SIZE: usize = 1024;

#[derive(Debug, Serialize, Deserialize)]
pub enum BrukerDataError {
    FidNotFound(PathBuf),
    ACQPNotFound(PathBuf),
    UnexpectedEOF(PathBuf),
    InconsistentArraySize{expected: usize, actual: usize},
    PV(PvError),
}

impl Display for BrukerDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::write;
        write!(f, "{:?}", self)
    }
}

impl From<PvError> for BrukerDataError {
    fn from(err: PvError) -> Self {
        BrukerDataError::PV(err)
    }
}

/// read an entire Bruker fid file to an array of complex floats. This expects a fid and acqp file to be in
/// the acquisition directory
pub fn read_bruker_fid(acq_dir:impl AsRef<Path>) -> Result<(Vec<Complex32>, ArrayDim), BrukerDataError> {

    let fid_file = acq_dir.as_ref().join("fid");
    let acqp_file = acq_dir.as_ref().join("acqp");

    if !fid_file.is_file() {
        Err(BrukerDataError::FidNotFound(fid_file.clone()))?
    }

    if !acqp_file.is_file() {
        Err(BrukerDataError::ACQPNotFound(acqp_file.clone()))?
    }

    let [n_read,n_coils] = get_acq_pars(&acq_dir)?;

    let word_size = size_of::<i32>();

    // read i32 fid data from file
    let fid_data = {
        let mut f = File::open(&fid_file).unwrap();
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes).unwrap();
        let fid_data:&[i32] = bytemuck::cast_slice(bytes.as_slice());
        fid_data.to_vec()
    };

    // total bytes in the file
    let n_bytes = word_size * fid_data.len();

    // number of blocks per continuous fid readout
    let blocks_per_fid = (2 * n_coils * n_read * word_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // window size to iterate over
    let chunk_size_words = blocks_per_fid * BLOCK_SIZE / word_size;

    let n_fids_per_chunk = chunk_size_words / (2 * n_coils * n_read);
    let s_chunk_size = 2 * n_coils * n_read * n_fids_per_chunk;

    let floats:Vec<Complex32> = fid_data.par_chunks_exact(chunk_size_words).map(|chunk| {
        chunk[0..s_chunk_size].chunks_exact(2)
            .map(|pair| Complex32::new(pair[0] as f32, pair[1] as f32)).collect::<Vec<Complex32>>()
    }).flatten().collect();

    let n_readouts = floats.len() / (n_read * n_coils);
    let dims = ArrayDim::from_shape(&[n_coils, n_read, n_readouts]);

    if dims.numel() != floats.len() {
        return Err(BrukerDataError::InconsistentArraySize {expected: dims.numel(), actual: floats.len()})
    }

    Ok((floats, dims))

}

/// fills a buffer of complex values from some offset for a bruker fid file
pub fn read_bruker_buffer(acq_dir:impl AsRef<Path>, offset:usize, buffer:&mut [Complex32]) -> Result<(),PvError> {
    let mut tmp_buff = vec![0i32; buffer.len() * 2];
    let offset = offset * 2;
    fill_buffer_i32(acq_dir, offset, &mut tmp_buff)?;
    tmp_buff.chunks_exact(2).zip(buffer.iter_mut()).for_each(|(t, s)| {
        s.re = t[0] as f32;
        s.im = t[1] as f32;
    });
    Ok(())
}


/// fills a buffer of single i32 values from a bruker fid from some offset. Note that these are not
/// complex values, so the offset needs to be 2 * n_complex_points
fn fill_buffer_i32(acq_dir:impl AsRef<Path>, offset:usize, buffer:&mut [i32]) -> Result<(),PvError> {

    let word_size = size_of::<i32>();

    let [n_read, n_coils] = get_acq_pars(&acq_dir)?;

    // valid bytes per ray
    let ray_size = n_coils * n_read * word_size;

    // padded bytes per ray on disk
    let blocks_per_ray = (ray_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let chunk_size = blocks_per_ray * BLOCK_SIZE;

    // logical offset into the contiguous data stream
    let byte_offset = offset * word_size;

    // which ray contains the logical offset?
    let starting_chunk = byte_offset / ray_size;

    // byte offset within the valid part of that ray
    let first_chunk_offset = byte_offset % ray_size;

    // physical file offset
    let total_byte_offset = starting_chunk * chunk_size;

    let mut f = File::open(acq_dir.as_ref().join("fid"))?;
    let mut chunk_buff = vec![0u8; chunk_size];

    let mut n_elements_read = 0;
    let n_elements_to_read = buffer.len();

    f.seek(SeekFrom::Start(total_byte_offset as u64))?;

    while n_elements_read < n_elements_to_read {
        f.read_exact(&mut chunk_buff)?;

        let start_byte = if n_elements_read == 0 {
            first_chunk_offset
        } else {
            0
        };

        let available_bytes = ray_size - start_byte;
        let available_elements = available_bytes / word_size;
        let needed_elements = n_elements_to_read - n_elements_read;
        let n_copy = available_elements.min(needed_elements);

        let data: &[i32] = bytemuck::cast_slice(&chunk_buff[start_byte..start_byte + n_copy * word_size]);

        buffer[n_elements_read..n_elements_read + n_copy].copy_from_slice(data);

        n_elements_read += n_copy;
    }

    Ok(())
}

#[test]
fn test() {

    let mut b = vec![];
    read_bruker_buffer("/sd", 0, &mut b).unwrap()


}




/// returns the number of readout samples and number of coils to properly read the fid file
fn get_acq_pars(acq_dir:impl AsRef<Path>) -> Result<[usize;2],PvError> {
    let acqp_file = acq_dir.as_ref().join("acqp");
    let params = parse_paravision_params(acqp_file)?;
    let acq_size = params.acq_size()?;
    let n_coils = params.n_coils().unwrap_or(1);
    let n_read = acq_size[0];
    Ok([n_read,n_coils])
}