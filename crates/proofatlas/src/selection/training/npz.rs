//! Minimal NPZ (NumPy archive) writer.
//!
//! Writes .npy arrays inside a zip archive to produce files readable by
//! `numpy.load()`. Supports 1-D and 2-D arrays of f32, i32, i64, u8, i8.

use std::fs::File;
use std::io::{self, BufWriter, Write as IoWrite};
use std::path::Path;

use zip::write::{FileOptions, SimpleFileOptions};
use zip::ZipWriter;

/// Trait for types that can be written as NPY arrays.
pub trait NpyType: Copy {
    /// NumPy dtype string (little-endian).
    fn dtype_str() -> &'static str;
    /// Size in bytes of one element.
    fn byte_size() -> usize;
    /// Write raw bytes of a single element (little-endian).
    fn write_le(&self, buf: &mut Vec<u8>);
}

impl NpyType for f32 {
    fn dtype_str() -> &'static str { "<f4" }
    fn byte_size() -> usize { 4 }
    fn write_le(&self, buf: &mut Vec<u8>) { buf.extend_from_slice(&self.to_le_bytes()); }
}

impl NpyType for i32 {
    fn dtype_str() -> &'static str { "<i4" }
    fn byte_size() -> usize { 4 }
    fn write_le(&self, buf: &mut Vec<u8>) { buf.extend_from_slice(&self.to_le_bytes()); }
}

impl NpyType for i64 {
    fn dtype_str() -> &'static str { "<i8" }
    fn byte_size() -> usize { 8 }
    fn write_le(&self, buf: &mut Vec<u8>) { buf.extend_from_slice(&self.to_le_bytes()); }
}

impl NpyType for u8 {
    fn dtype_str() -> &'static str { "|u1" }
    fn byte_size() -> usize { 1 }
    fn write_le(&self, buf: &mut Vec<u8>) { buf.push(*self); }
}

impl NpyType for i8 {
    fn dtype_str() -> &'static str { "|i1" }
    fn byte_size() -> usize { 1 }
    fn write_le(&self, buf: &mut Vec<u8>) { buf.push(*self as u8); }
}

/// Writer for NPZ (zip of .npy) files.
pub struct NpzWriter {
    zip: ZipWriter<BufWriter<File>>,
}

impl NpzWriter {
    /// Create a new NPZ writer at the given path.
    pub fn new(path: &Path) -> io::Result<Self> {
        let file = File::create(path)?;
        let buf = BufWriter::new(file);
        let zip = ZipWriter::new(buf);
        Ok(NpzWriter { zip })
    }

    /// Write a 1-D array as `name.npy` inside the archive.
    pub fn write_array_1d<T: NpyType>(&mut self, name: &str, data: &[T]) -> io::Result<()> {
        let shape_str = format!("({},)", data.len());
        self.write_npy(name, T::dtype_str(), &shape_str, data)
    }

    /// Write a 2-D array (row-major) as `name.npy` inside the archive.
    pub fn write_array_2d<T: NpyType>(
        &mut self,
        name: &str,
        data: &[T],
        rows: usize,
        cols: usize,
    ) -> io::Result<()> {
        debug_assert_eq!(data.len(), rows * cols, "data length must equal rows * cols");
        let shape_str = format!("({}, {})", rows, cols);
        self.write_npy(name, T::dtype_str(), &shape_str, data)
    }

    /// Finalize and close the archive.
    pub fn finish(self) -> io::Result<()> {
        self.zip
            .finish()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(())
    }

    /// Internal: write a single .npy entry.
    fn write_npy<T: NpyType>(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &str,
        data: &[T],
    ) -> io::Result<()> {
        let entry_name = format!("{}.npy", name);
        let options: SimpleFileOptions = FileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);

        self.zip
            .start_file(&entry_name, options)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        // Build .npy header
        let header_dict = format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
            dtype, shape
        );
        // Header must be padded to align total (magic+version+header_len+header) to 64 bytes
        let prefix_len = 10; // magic(6) + version(2) + header_len(2)
        let unpadded = prefix_len + header_dict.len() + 1; // +1 for newline
        let padding = (64 - (unpadded % 64)) % 64;
        let header_len = header_dict.len() + padding + 1; // includes trailing newline

        // Magic number + version 1.0
        self.zip.write_all(b"\x93NUMPY\x01\x00")?;
        // Header length (little-endian u16)
        self.zip.write_all(&(header_len as u16).to_le_bytes())?;
        // Header dict + padding + newline
        self.zip.write_all(header_dict.as_bytes())?;
        for _ in 0..padding {
            self.zip.write_all(b" ")?;
        }
        self.zip.write_all(b"\n")?;

        // Raw data
        let mut buf = Vec::with_capacity(data.len() * T::byte_size());
        for item in data {
            item.write_le(&mut buf);
        }
        self.zip.write_all(&buf)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_write_and_read_npz() {
        let dir = std::env::temp_dir().join("proofatlas_npz_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test.npz");

        {
            let mut w = NpzWriter::new(&path).unwrap();
            w.write_array_1d("labels", &[1u8, 0, 1, 0, 1]).unwrap();
            w.write_array_1d("steps", &[0i32, 1, -1, 2, 3]).unwrap();
            w.write_array_2d("features", &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
                .unwrap();
            w.finish().unwrap();
        }

        // Verify the file is valid zip
        let file = File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 3);

        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(names.contains(&"labels.npy".to_string()));
        assert!(names.contains(&"steps.npy".to_string()));
        assert!(names.contains(&"features.npy".to_string()));

        let _ = fs::remove_file(&path);
    }
}
