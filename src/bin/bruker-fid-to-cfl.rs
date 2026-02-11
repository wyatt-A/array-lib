use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use clap::Parser;
use bruker_jcamp_rs::{parse_paravision_params, PvError, PvValue};
use num_complex::Complex32;
use rayon::prelude::*;
use array_lib::ArrayDim;
use array_lib::io_cfl::write_cfl;

//* Bruker acqp definitions to infer fid file layout *//
/// number of echoes in a TR, usually within an inner loop of the ppg
const N_ECHOES: &str = "NECHOES";
const ACQ_SIZE: &str = "ACQ_size";

/// number of repeat scans often used for time-series acquisitions
const N_REPEATS: &str = "NR";

const RECEIVERS: &str = "ACQ_ReceiverSelect";

const WORD_SIZE:&str = "ACQ_word_size";

/// block size in bytes for the standard Bruker "KBlock" format
const BLOCK_SIZE: usize = 1024;

#[derive(Debug)]
enum FidToCflError {
    FieldNotFound(String),
    UnexpectedFormat(PvValue),
    IO(std::io::Error),
    PV(PvError),
    UnexpectedDataType(String),
}

#[derive(Parser)]
struct Args {
    /// path to Bruker fid file to parse
    fid_file: PathBuf,
    /// output cfl file
    cfl_file: PathBuf,
    /// path to Bruker acquisition parameters file
    acqp_file: PathBuf,

    /// oversampling factor for cases where the acq_size is reported as some factor of the readout size.
    /// This is usually 2 for radial scans
    #[clap(short, long)]
    f_oversample: Option<usize>,

    debug:bool,
}

fn main() -> Result<(), FidToCflError> {

    use FidToCflError::*;

    let args = Args::parse();

    let oversampling_factor = args.f_oversample.unwrap_or(1);

    let f = File::open(args.acqp_file).map_err(IO)?;
    let acqp = parse_paravision_params(
        BufReader::new(f),
    ).map_err(PV)?;

    let acq_size = acqp.params.get(ACQ_SIZE).ok_or_else(|| FieldNotFound(String::from(ACQ_SIZE)))?;
    let receivers = acqp.params.get(RECEIVERS).ok_or_else(|| FieldNotFound(String::from(RECEIVERS)))?;
    let n_echoes = acqp.params.get(N_ECHOES).ok_or_else(|| FieldNotFound(String::from(N_ECHOES)))?;
    let n_repeats = acqp.params.get(N_REPEATS).ok_or_else(|| FieldNotFound(String::from(N_REPEATS)))?;

    let acq_size = acq_size.to_vec_usize().ok_or_else(|| UnexpectedFormat(acq_size.clone()))?;
    let receivers = receivers.to_vec_bool().ok_or_else(|| UnexpectedFormat(receivers.clone()))?.iter().filter(|r|**r).count();
    let n_echoes = n_echoes.to_usize().ok_or_else(|| UnexpectedFormat(n_echoes.clone()))?;
    let n_repeats = n_repeats.to_usize().ok_or_else(|| UnexpectedFormat(n_repeats.clone()))?;



    let word_size = acqp.params.get(WORD_SIZE).ok_or_else(|| FieldNotFound(String::from(WORD_SIZE)))?.to_string();

    let bytes_per_sample = match word_size.as_str() {
        "_32_BIT" => {
            8 // 8 bytes per complex data point
        }
        _=> Err(UnexpectedDataType(word_size))?
    };

    // this is the data ordering usually streaming off the scanner. These data points should be contiguous in the fid file
    let chunk_size_samples = acq_size[0]/oversampling_factor * receivers * n_echoes;

    let total_samples = chunk_size_samples * acq_size[1..].iter().product::<usize>() * n_repeats;

    let n_chunks = total_samples / chunk_size_samples;

    let samples_per_block = BLOCK_SIZE / bytes_per_sample;



    // ceil division for blocks per chunk
    let blocks_per_chunk = (chunk_size_samples * bytes_per_sample + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let n_fid_samples = n_chunks * blocks_per_chunk * samples_per_block;
    let expected_fid_file_size_bytes = n_chunks * blocks_per_chunk * BLOCK_SIZE;

    if args.debug {
        println!("acq_size = {:?}",acq_size);
        println!("receivers = {:?}",receivers);
        println!("n_echoes = {:?}",n_echoes);
        println!("n_repeats = {:?}",n_repeats);
        println!("samples_per_block = {:?}",samples_per_block);
        println!("n_fid_samples = {:?}",n_fid_samples);
        println!("blocks_per_chunk = {:?}",blocks_per_chunk);
        println!("expected_fid_file_size_bytes = {:?}",expected_fid_file_size_bytes);
    }

    let mut f = File::open(args.fid_file).map_err(IO)?;

    let mut fid_bytes = vec![];
    f.read_to_end(&mut fid_bytes).map_err(IO)?;

    assert_eq!(
        fid_bytes.len(),
        expected_fid_file_size_bytes,
        "unexpected fid file size. Expected {}, got {} bytes",expected_fid_file_size_bytes,fid_bytes.len()
    );

    let mut fid_data = vec![Complex32::ZERO; total_samples];

    let bytes_per_chunk = chunk_size_samples * bytes_per_sample;

    fid_bytes.par_chunks_exact(blocks_per_chunk * BLOCK_SIZE).zip(fid_data.par_chunks_exact_mut(chunk_size_samples)).for_each(|(chunk_bytes,fid_data)| {
        let x = &chunk_bytes[0..bytes_per_chunk]; // only read the bytes we care about
        let y:&[i32] = bytemuck::cast_slice(x);
        y.chunks_exact(2).zip(fid_data.iter_mut()).for_each(|(i,f)| {
            *f = Complex32::new(
                i[0] as f32,
                i[1] as f32
            );
        });
    });

    let dim_x = acq_size[0]/oversampling_factor;
    let dim_y = acq_size[1];
    let dim_z = *acq_size.get(2).unwrap_or(&1usize);

    let dims = ArrayDim::from_shape(&[dim_x,receivers,n_echoes,dim_y,dim_z,n_repeats]);
    assert_eq!(dims.numel(),fid_data.len(),"incorrect dimensions");
    write_cfl(args.cfl_file,&fid_data,dims);

    Ok(())

}