use super::{FeatureExtractor, KeyPointSelector, SccFilter};
use anyhow::Result;
use opencv::core::{KeyPoint, Ptr, Vector};
use opencv::features2d::{FastFeatureDetector, ORB};
use opencv::prelude::*;

pub struct OrbFeatureDescriptor(Mat);

pub struct OrbFeatureExtractor<D, E, S> {
    detector: D,
    computer: E,
    selector: S,
}

impl OrbFeatureExtractor<Ptr<FastFeatureDetector>, Ptr<ORB>, SccFilter> {
    pub fn new() -> Self {
        let detector = FastFeatureDetector::create_def().unwrap();
        let computer = ORB::create_def().unwrap();
        let selector = SccFilter::default();
        Self {
            detector,
            computer,
            selector,
        }
    }
}

impl<D, S, E> FeatureExtractor for OrbFeatureExtractor<D, E, S>
where
    D: Feature2DTrait,
    E: Feature2DTrait,
    S: KeyPointSelector,
{
    type Descriptor = OrbFeatureDescriptor;

    fn detect(&mut self, image: &Mat, num_points: u32) -> Result<Vector<KeyPoint>> {
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

    fn detect_and_compute(
        &mut self,
        image: &Mat,
        num_points: u32,
    ) -> Result<(Vector<KeyPoint>, Vec<Self::Descriptor>)> {
        let mut keypoints = self.detect(image, num_points)?;
        let mut descriptors = Mat::default();
        self.computer
            .compute(image, &mut keypoints, &mut descriptors)?;
        Ok((keypoints, descriptors))
    }
}
