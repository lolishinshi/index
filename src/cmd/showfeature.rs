use anyhow::Result;
use opencv::features2d::{draw_keypoints_def, FastFeatureDetector, ORB};
use opencv::imgcodecs::{imread, imwrite_def, IMREAD_GRAYSCALE};
use opencv::prelude::*;

use crate::{
    features::{FeatureExtractor, SccFilter},
    ShowFeatureCommand,
};

pub fn cmd_show_feature(config: ShowFeatureCommand) -> Result<()> {
    let image = imread(&config.image, IMREAD_GRAYSCALE)?;
    let fast = FastFeatureDetector::create_def()?;
    let orb = ORB::create_def()?;
    let scc = SccFilter::default();
    let mut extractor = FeatureExtractor::new(fast, orb, scc);
    let keypoints = extractor.detect(&image, config.num_points)?;
    let mut output = Mat::default();
    draw_keypoints_def(&image, &keypoints, &mut output)?;
    imwrite_def(&config.output, &output)?;
    Ok(())
}
