use anyhow::Result;
use opencv::{
    features2d::{Feature2DTraitConst, GFTTDetector, SIFT},
    imgcodecs::{imread, IMREAD_GRAYSCALE},
};

use crate::{
    features::{FeatureExtractor, SccFilter},
    ShowFeatureCommand,
};

pub fn cmd_show_feature(config: ShowFeatureCommand) -> Result<()> {
    let image = imread(&config.image, IMREAD_GRAYSCALE)?;
    let gftt = GFTTDetector::create_def()?;
    let sift = SIFT::create_def()?;
    let scc = SccFilter::default();
    let extractor = FeatureExtractor::new(gftt, sift, scc);
    todo!()
}
