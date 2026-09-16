//! Packed scheduler-drain shape for grid-patch multimodal families.
//!
//! Extracted from `qwen_vl` so a second grid family reuses the same handoff
//! rather than inheriting qwen's module path.

use crate::pipeline::TensorData;

/// The scheduler-drain shape for grid-patch families, extracted from the
/// generic driver [`Output`](crate::driver::Output). Shared by
/// `sglang-server`'s MM worker and the parity binding so the mapping can't
/// drift. TODO(mm-families): replace with a generic named-tensor handoff once
/// a family needs a different shape.
pub struct GridPackedOutput {
    pub input_ids: Vec<i32>,
    /// All items' `pixel_values`, concatenated in prompt order; flattened
    /// `[Σ t·h·w, 3·temporal_patch_size·patch_size²]`.
    pub features: Vec<f32>,
    /// Per item `[t, h, w]` patch grid.
    pub grids: Vec<[u32; 3]>,
    pub hashes: Vec<u64>,
    /// Per item inclusive token range in `input_ids`.
    pub offsets: Vec<(u32, u32)>,
    /// Flattened row-major `[3, input_len]` M-RoPE positions.
    pub mrope: Vec<i64>,
    pub mrope_delta: i64,
}

/// `family` only labels the error messages, so each family keeps the exact
/// rejection text its clients already see.
pub fn pack_grid_output(
    output: crate::driver::Output,
    family: &'static str,
) -> Result<GridPackedOutput, String> {
    use crate::pipeline::PositionOutput;

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
