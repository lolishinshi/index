use usearch::{b1x8, new_index, Index, IndexOptions, MetricKind, ScalarKind};

pub struct IndexDB {
    index: Index,
}

impl IndexDB {
    pub fn new() -> IndexDB {
        let options = IndexOptions {
            dimensions: 256,
            metric: MetricKind::Hamming,
            quantization: ScalarKind::B1,
            ..Default::default()
        };
        let index = new_index(&options).unwrap();
        Self { index }
    }

    pub fn add_vector(&self, key: u64, vector: &[u8]) {
        let vector = vector.into_iter().map(|v| b1x8(*v)).collect::<Vec<_>>();
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
