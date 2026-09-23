//! GLM-VL launch contract. Image processing is added in a later stage.

#[derive(Clone, Debug, serde::Deserialize)]
pub struct GlmVlSpec {
    pub image_token_id: i32,
    pub image_start_token_id: i32,
    pub image_end_token_id: i32,
    pub video_start_token_id: i32,
    pub video_end_token_id: i32,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub patch_expand_factor: usize,
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub resample: crate::qwen_vl::Resampler,
}
