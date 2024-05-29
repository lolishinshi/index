//! Efficient adaptive non-maximal suppression algorithms for homogeneous spatial keypoint distribution
//!
//! ## Reference
//! - https://github.com/BAILOOL/ANMS-Codes

use opencv::core::{KeyPoint, Vector};
use opencv::prelude::*;

use super::KeyPointSelector;

pub struct SccFilter(pub f32);

impl Default for SccFilter {
    fn default() -> Self {
        Self(0.1)
    }
}

impl KeyPointSelector for SccFilter {
    fn select(
        &self,
        keypoints: Vector<KeyPoint>,
        num_points: u32,
        width: usize,
        height: usize,
    ) -> Vector<KeyPoint> {
        ssc(keypoints, num_points, self.0, width as u32, height as u32)
    }
}

// 注：tolerance 为选择的点的数量与理想数量的容差，默认 0.1
pub fn ssc(
    keypoints: Vector<KeyPoint>,
    num_ret_points: u32,
    tolerance: f32,
    cols: u32,
    rows: u32,
) -> Vector<KeyPoint> {
    let exp1 = (rows + cols + 2 * num_ret_points) as f32;
    let exp2 =
        (4 * cols + 4 * num_ret_points + 4 * rows * num_ret_points + rows * rows + cols * cols
            - 2 * rows * cols
            + 4 * rows * cols * num_ret_points) as f32;
    let exp3 = exp2.sqrt();
    let exp4 = (num_ret_points - 1) as f32;

    let sol1 = -((exp1 + exp3) / exp4).round(); // first solution
    let sol2 = -((exp1 - exp3) / exp4).round(); // second solution

    // binary search range initialization with positive solution
    let mut high = sol1.max(sol2);
    let mut low = (keypoints.len() as f32 / num_ret_points as f32)
        .sqrt()
        .floor();

    let mut prev_width = -1.0;
    let mut result = Vector::new();
    let k = num_ret_points as f32;
    let k_min = (k - (k * tolerance)).round();
    let k_max = (k + (k * tolerance)).round();

    loop {
        let width = low + (high - low) / 2.0;
        // needed to reassure the same radius is not repeated again
        if width == prev_width || low > high {
            // return the keypoints from the previous iteration
            return result;
        }

        let c = width / 2.0;
        let num_cell_cols = (cols as f32 / c).floor() as usize;
        let num_cell_rows = (rows as f32 / c).floor() as usize;
        let mut covered_vec = vec![vec![false; num_cell_cols + 1]; num_cell_rows + 1];
        result.clear();

        for keypoint in &keypoints {
            // get position of the cell current point is located at
            let row = (keypoint.pt().y / c).floor() as usize;
            let col = (keypoint.pt().x / c).floor() as usize;
            if !covered_vec[row][col] {
                result.push(keypoint);
                // get range which current radius is covering
                let row_min = row.saturating_sub((width / c).floor() as usize);
                let row_max = (row + (width / c).floor() as usize).min(num_cell_rows);
                let col_min = col.saturating_sub((width / c).floor() as usize);
                let col_max = (col + (width / c).floor() as usize).min(num_cell_cols);
                covered_vec
                    .iter_mut()
                    .take(row_max + 1)
                    .skip(row_min)
                    .for_each(|v| {
                        v.iter_mut().take(col_max + 1).skip(col_min).for_each(|v| {
                            *v = true;
                        });
                    });
            }
        }

        if k_min <= result.len() as f32 && result.len() as f32 <= k_max {
            // solution found
            return result;
        }
        if (result.len() as f32) < k_min {
            high = width - 1.0; // update binary search range
        } else {
            low = width + 1.0;
        }

        prev_width = width
    }
}
