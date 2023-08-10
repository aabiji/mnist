use std::fs::File;
use std::io::{Read, BufReader};

fn read_file(path: &str) -> Vec<u8> {
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);
    let mut buffer: Vec<u8> = Vec::new();
    reader.read_to_end(&mut buffer).unwrap();
    return buffer;
}

fn read_big_endian(buf: &Vec<u8>, i: usize) -> (u32, usize) {
    let mut v: u32 = 0;
    v |= (buf[i+0] as u32) << 24;
    v |= (buf[i+1] as u32) << 16;
    v |= (buf[i+2] as u32) << 8;
    v |= buf[i+3] as u32;
    return (v, i + 4);
}

pub fn load_label(path: &str) -> Vec<u8> {
    let file = read_file(path);
    let (item_count, mut i) = read_big_endian(&file, 4);

    let mut labels: Vec<u8> = Vec::new();
    for _ in 0..item_count {
        labels.push(file[i as usize]);
        i += 1;
    }

    return labels;
}

pub fn load_dataset(path: &str) -> Vec<Vec<f64>> {
    let file = read_file(path);
    let (img_count, mut i) = read_big_endian(&file, 4);
    i += 8; // Skip row and column count which should be 28

    let mut imgs: Vec<Vec<f64>> = Vec::new();
    for _ in 0..img_count {
        let mut temp: Vec<f64> = Vec::new();
        for _ in 0..784 {
            temp.push(file[i as usize] as f64);
            i += 1;
        }
        imgs.push(temp);
    }

    return imgs;
}
