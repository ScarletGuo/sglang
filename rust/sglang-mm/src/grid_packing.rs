//! Shared packed output for image families with an f32 patch grid and M-RoPE.

use crate::pipeline::{PositionOutput, TensorData};

pub struct GridPackedOutput {
    pub input_ids: Vec<i32>,
    /// Concatenated `pixel_values`, flattened in prompt order.
    pub features: Vec<f32>,
    /// Per-item `[t, h, w]` patch grid.
    pub grids: Vec<[u32; 3]>,
    pub hashes: Vec<u64>,
    pub offsets: Vec<(u32, u32)>,
    /// Flattened row-major `[3, input_len]` M-RoPE positions.
    pub mrope: Vec<i64>,
    pub mrope_delta: i64,
}

pub fn pack_grid_output(
    output: crate::driver::Output,
    family: &str,
) -> Result<GridPackedOutput, String> {
    let PositionOutput::MRope { positions, delta } = output.positions else {
        return Err(format!("{family} pack: expected M-RoPE positions"));
    };
    let mut features = Vec::new();
    let mut grids = Vec::with_capacity(output.items.len());
    let mut hashes = Vec::with_capacity(output.items.len());
    for item in output.items {
        let TensorData::F32(pixel_values) = item.feature.data else {
            return Err(format!("{family} pack: expected f32 feature"));
        };
        features.extend(pixel_values);
        let grid = item
            .aux
            .into_iter()
            .find_map(|(name, tensor)| match (name.as_str(), tensor.data) {
                ("image_grid_thw", TensorData::I64(v)) => Some(v),
                _ => None,
            })
            .ok_or_else(|| format!("{family} pack: missing image_grid_thw"))?;
        if grid.len() != 3 || grid.iter().any(|&v| v <= 0 || v > u32::MAX as i64) {
            return Err(format!("{family} pack: invalid image_grid_thw"));
        }
        grids.push([grid[0] as u32, grid[1] as u32, grid[2] as u32]);
        hashes.push(item.hash);
    }
    Ok(GridPackedOutput {
        input_ids: output.input_ids,
        features,
        grids,
        hashes,
        offsets: output.offsets,
        mrope: positions,
        mrope_delta: delta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_error_uses_family_label() {
        let output = crate::driver::Output {
            input_ids: vec![],
            items: vec![],
            offsets: vec![],
            positions: PositionOutput::Rope1D,
        };
        assert_eq!(
            pack_grid_output(output, "glm_vl").err().as_deref(),
            Some("glm_vl pack: expected M-RoPE positions")
        );
    }
}
