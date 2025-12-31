use std::path::PathBuf;
use clap::Parser;
use array_lib::io_cfl::write_cfl;
use array_lib::io_mrd;

#[derive(Parser)]
struct Args {
    /// mrd file to read
    mrd_file:PathBuf,
    /// output cfl file
    cfl_file:PathBuf,
}

fn main() {
    let args = Args::parse();
    let (data,dims,_) = io_mrd::read_mrd(args.mrd_file);
    write_cfl(args.cfl_file,&data,dims);
}