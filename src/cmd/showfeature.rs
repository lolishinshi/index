use anyhow::Result;
use opencv::{
    imgcodecs::{imread, IMREAD_GRAYSCALE},
};
use opencv::features2d::{draw_keypoints_def, GFTTDetector, SIFT};
use opencv::highgui::{imshow, wait_key};
use opencv::prelude::*;

use crate::{
    features::{FeatureExtractor, SccFilter},
    ShowFeatureCommand,
};

pub fn cmd_show_feature(config: ShowFeatureCommand) -> Result<()> {
    let image = imread(&config.image, IMREAD_GRAYSCALE)?;
    let gftt = GFTTDetector::create_def()?;
    let sift = SIFT::create_def()?;
    let scc = SccFilter::default();
    let mut extractor = FeatureExtractor::new(gftt, sift, scc);
    let keypoints = extractor.detect(&image, config.num_points)?;
    dbg!(keypoints.len());
    let mut output = Mat::default();
    draw_keypoints_def(&image, &keypoints, &mut output)?;
    imshow("output", &output)?;
    wait_key(0);
    Ok(())
}
