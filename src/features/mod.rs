mod anms;
mod extractor;

use opencv::core::{KeyPoint, Vector};

pub use anms::SccFilter;
pub use extractor::FeatureExtractor;

pub trait KeyPointSelector {
    fn select(
        &self,
        keypoints: Vector<KeyPoint>,
        num_points: u32,
        width: usize,
        height: usize,
    ) -> Vector<KeyPoint>;
}
