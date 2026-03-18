use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use num_complex::Complex32;
use serde::{Deserialize, Serialize};
use crate::ArrayDim;
use bruker_jcamp_rs::{parse_paravision_params, PvError};
use rayon::prelude::*;

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

impl From<PvError> for BrukerDataError {
    fn from(err: PvError) -> Self {
        BrukerDataError::PV(err)
    }
}

/// read a Bruker fid file to an array of complex floats. This expects a fid and acqp file to be in
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

    let params = parse_paravision_params(acqp_file)?;

    let acq_size = params.acq_size()?;
    let n_coils = params.n_coils()?;
    let n_read = acq_size[0];

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

