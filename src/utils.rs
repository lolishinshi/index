use opencv::core::{KeyPoint, Vector};
use opencv::highgui::{imshow, wait_key};
use opencv::prelude::*;
use opencv::features2d::draw_keypoints_def;
use anyhow::Result;


pub fn show_keypoints(image: &Mat, keypoints: &Vector<KeyPoint>) -> Result<()> {
    let mut output = Mat::default();
    draw_keypoints_def(image, keypoints, &mut output)?;
    imshow("keypoints", &output)?;
    wait_key(0)?;
    Ok(())
}

