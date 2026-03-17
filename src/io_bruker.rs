use std::fs::File;
use std::io::Read;
use std::path::Path;
use num_complex::Complex32;
use crate::ArrayDim;

// 1kB block size
const BLOCK_SIZE: usize = 1024;

#[test]
fn test_io_bruker() {

    read_bruker_fid("/Users/wyatt/1/fid", 63, 4 );

}


/// read a Bruker fid file based on known array dimensions. This function trims off extra bytes associated
/// with the 1kb-block format
pub fn read_bruker_fid(fid_file:impl AsRef<Path>, n_read:usize, n_coils:usize) -> (Vec<Complex32>, ArrayDim) {

    let mut f = File::open(&fid_file).unwrap();
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes).unwrap();

    //let fid_data:&[i32] = bytemuck::cast_slice(bytes.as_slice());

    let word_size = size_of::<i32>();

    // bytes per contiguous chunk
    let bytes_per_chunk = 2 * n_coils * n_read * word_size;

    // ceiling division
    let blocks_per_chunk = (bytes_per_chunk + BLOCK_SIZE - 1)/BLOCK_SIZE;

    let window_size = blocks_per_chunk * BLOCK_SIZE;

    let n_windows = bytes.len() /  window_size;
    assert!(bytes.len() %  window_size == 0, "window size does not fit into bytes evenly");

    let n_points = n_windows * n_coils * n_read;


    // allocate complex data buffer to read bytes into
    let mut array_data = vec![0f32; n_points * 2];

    bytes.chunks_exact(window_size).for_each(|window| {

        // pull bytes from window, cast to i32, then cast to f32
        window[0..]

    });

    // do final bytemuck cast to complex32 to return

    println!("blocks_per_chunk = {}", blocks_per_chunk);
    println!("n_windows = {}", n_windows);
    println!("n_points = {}", n_points);
    todo!()
}

