use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use clap::Parser;
use num_complex::Complex32;
use rayon::prelude::*;
use array_lib::ArrayDim;
use array_lib::io_cfl::write_cfl;

#[derive(Parser, Debug)]
struct Args {
    traj_file: PathBuf,
    cfl_file: PathBuf,
    readout_size: usize,
}

#[derive(Debug)]
enum FidToCflError {
    IO(std::io::Error),
    UnexpectedDataType(String),
}

fn main() -> Result<(), FidToCflError> {

    let args = Args::parse();

    let mut traj_bytes:Vec<u8> = vec![];

    let mut f = File::open(args.traj_file).map_err(FidToCflError::IO)?;
    f.read_to_end(&mut traj_bytes).map_err(FidToCflError::IO)?;
    let traj:&[f64] = bytemuck::cast_slice(&traj_bytes);

    assert_eq!(traj.len()%(3*args.readout_size), 0, "number of traj samples must be divisible by 3");

    let points_per_channel = traj.len() / (3*args.readout_size);

    let mut cfl_data = vec![Complex32::ZERO; traj.len()];
    cfl_data.par_iter_mut().zip(traj.par_iter()).for_each(|(c,f)|{
       // write to real part
        *c = Complex32::new(*f as f32, 0.);
    });

    let cfl_dims = ArrayDim::from_shape(&[3, args.readout_size, points_per_channel]);

    write_cfl(args.cfl_file,&cfl_data,cfl_dims);

    Ok(())

}