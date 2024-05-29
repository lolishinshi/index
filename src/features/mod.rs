mod anms;
mod extractor;

use anyhow::Result;
use opencv::{
    core::{KeyPoint, Vector},
    prelude::*,
};

pub use anms::SccFilter;
pub use extractor::*;

pub trait FeatureExtractor {
    type Descriptor;

    fn detect(&mut self, image: &Mat, num_points: u32) -> Result<Vector<KeyPoint>>;

    fn detect_and_compute(
        &mut self,
        image: &Mat,
        num_points: u32,
    ) -> Result<(Vector<KeyPoint>, Vec<Self::Descriptor>)>;
}

pub trait KeyPointSelector {
    fn select(
        &self,
        keypoints: Vector<KeyPoint>,
        num_points: u32,
        width: usize,
        height: usize,
    ) -> Vector<KeyPoint>;
}
