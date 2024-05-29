use std::marker::PhantomData;
use usearch::{b1x8, new_index, Index, IndexOptions, MetricKind, ScalarKind};

use crate::features::OrbFeatureDescriptor;

trait BinaryVector {
    const SIZE: usize;

    fn to_bit_vec(&self) -> Vec<b1x8>;
}

impl BinaryVector for OrbFeatureDescriptor {
    const SIZE: usize = 256;

    fn to_bit_vec(&self) -> Vec<b1x8> {
        todo!()
    }
}

pub struct IndexDB<T> {
    index: Index,
    _phantom: PhantomData<T>,
}

impl<T> IndexDB<T>
where
    T: BinaryVector,
{
    pub fn new() -> IndexDB<T> {
        let options = IndexOptions {
            dimensions: 256,
            metric: MetricKind::Hamming,
            quantization: ScalarKind::B1,
            ..Default::default()
        };
        let index = new_index(&options).unwrap();
        Self {
            index,
            _phantom: PhantomData,
        }
    }

    pub fn add_vector(&self, key: u64, vector: &T) {
        let vector = vector.to_bit_vec();
        self.index.add(key, &vector).unwrap();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_index_db() {
        let db = super::IndexDB::new();
    }
}
