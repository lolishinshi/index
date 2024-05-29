use super::KeyPointSelector;
use anyhow::Result;
use opencv::core::{KeyPoint, Vector};
use opencv::features2d::{GFTTDetector, SIFT};
use opencv::prelude::*;

pub struct FeatureExtractor<D, E, S> {
    detector: D,
    computer: E,
    selector: S,
}

impl<D, E, S> FeatureExtractor<D, E, S>
where
    D: Feature2DTrait,
    E: Feature2DTrait,
    S: KeyPointSelector,
{
    pub fn new(detector: D, computer: E, selector: S) -> Self {
        Self {
            detector,
            computer,
            selector,
        }
    }

    pub fn detect(&mut self, image: &Mat, num_points: u32) -> Result<Vector<KeyPoint>> {
        let mut keypoints = Vector::new();
        let mask = Mat::default();
        self.detector.detect(image, &mut keypoints, &mask)?;
        let size = image.size()?;
        let keypoints = self.selector.select(
            keypoints,
            num_points,
            size.width as usize,
            size.height as usize,
        );
        Ok(keypoints)
    }

    pub fn detect_and_compute(
        &mut self,
        image: &Mat,
        num_points: u32,
    ) -> Result<(Vector<KeyPoint>, Mat)> {
        let mut keypoints = self.detect(image, num_points)?;
        let mut descriptors = Mat::default();
        self.computer
            .compute(image, &mut keypoints, &mut descriptors)?;
        Ok((keypoints, descriptors))
    }
}
