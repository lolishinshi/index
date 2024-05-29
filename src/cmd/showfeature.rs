use anyhow::Result;
use opencv::features2d::draw_keypoints_def;
use opencv::imgcodecs::{imread, imwrite_def, IMREAD_GRAYSCALE};
use opencv::prelude::*;

use crate::{
    features::{FeatureExtractor, OrbFeatureExtractor},
    ShowFeatureCommand,
};

pub fn cmd_show_feature(config: ShowFeatureCommand) -> Result<()> {
    let image = imread(&config.image, IMREAD_GRAYSCALE)?;
    let mut extractor = OrbFeatureExtractor::new();
    let (keypoints, _) = extractor.detect_and_compute(&image, config.num_points)?;
    println!("Number of keypoints: {}", keypoints.len());
    let mut output = Mat::default();
    draw_keypoints_def(&image, &keypoints, &mut output)?;
    imwrite_def(&config.output, &output)?;
    Ok(())
}
