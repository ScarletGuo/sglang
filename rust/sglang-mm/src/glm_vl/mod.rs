//! GLM-5.3-Flash image family. Images use an aligned canvas with right/bottom
//! padding; the content is resized independently of the canvas.

use crate::common::{grid, resize, token_layout};
use crate::pipeline::{
    DecodedMedia, Geometry, MmFamilyProcessor, PositionOutput, ProcessedItem, Tensor, TensorData,
    TokenLayout,
};
use crate::qwen_vl::Resampler;
use image::{DynamicImage, ImageDecoder, ImageReader};
use std::io::Cursor;

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
    pub resample: Resampler,
}

#[derive(Debug, PartialEq, Eq)]
struct GlmResize {
    canvas_h: usize,
    canvas_w: usize,
    content_h: usize,
    content_w: usize,
}

pub struct GlmVlProcessor {
    spec: GlmVlSpec,
    lut: [[f32; 256]; 3],
    factor: usize,
    min_pixels: u128,
    max_pixels: u128,
}

impl GlmVlProcessor {
    pub fn new(spec: GlmVlSpec) -> Result<Self, String> {
        if [
            spec.patch_size,
            spec.merge_size,
            spec.temporal_patch_size,
            spec.patch_expand_factor,
            spec.min_image_tokens,
            spec.max_image_tokens,
        ]
        .contains(&0)
            || spec.min_image_tokens > spec.max_image_tokens
        {
            return Err("glm_vl spec: sizes and token budgets must be positive and ordered".into());
        }
        if spec.image_std.iter().any(|&v| !v.is_finite() || v == 0.0)
            || spec.image_mean.iter().any(|&v| !v.is_finite())
        {
            return Err("glm_vl spec: mean/std must be finite and std nonzero".into());
        }
        if spec.resample == Resampler::Pil && spec.patch_expand_factor != 1 {
            return Err("glm_vl spec: PIL backend requires patch_expand_factor=1".into());
        }
        let factor = spec
            .patch_size
            .checked_mul(spec.merge_size)
            .and_then(|v| v.checked_mul(spec.patch_expand_factor))
            .ok_or("glm_vl spec: factor overflow")?;
        let base_factor = (spec.patch_size as u128) * (spec.merge_size as u128);
        let pixels_per_token = (spec.temporal_patch_size as u128)
            .checked_mul(base_factor)
            .and_then(|v| v.checked_mul(base_factor))
            .ok_or("glm_vl spec: pixel budget overflow")?;
        let min_pixels = (spec.min_image_tokens as u128)
            .checked_mul(pixels_per_token)
            .ok_or("glm_vl spec: minimum pixel budget overflow")?;
        let max_pixels = (spec.max_image_tokens as u128)
            .checked_mul(pixels_per_token)
            .ok_or("glm_vl spec: maximum pixel budget overflow")?;
        let minimum_canvas = (spec.temporal_patch_size as u128)
            .checked_mul(factor as u128)
            .and_then(|v| v.checked_mul(factor as u128))
            .ok_or("glm_vl spec: minimum canvas overflow")?;
        if max_pixels < minimum_canvas {
            return Err("glm_vl spec: max_image_tokens cannot fit the minimum canvas".into());
        }
        let lut = core::array::from_fn(|c| {
            grid::normalize_lut(spec.resample, spec.image_mean[c], spec.image_std[c])
        });
        Ok(Self {
            spec,
            lut,
            factor,
            min_pixels,
            max_pixels,
        })
    }

    fn align(&self, value: usize) -> Result<usize, String> {
        value
            .div_ceil(self.factor)
            .checked_mul(self.factor)
            .ok_or_else(|| "glm_vl: aligned dimension overflow".into())
    }

    fn resize_geometry(&self, height: usize, width: usize) -> Result<GlmResize, String> {
        if height == 0 || width == 0 {
            return Err("glm_vl: empty image".into());
        }
        let frames = self.spec.temporal_patch_size as u128;
        let mut ch = self.align(height)?;
        let mut cw = self.align(width)?;
        let budget =
            |h: usize, w: usize| frames.saturating_mul(h as u128).saturating_mul(w as u128);
        if budget(ch, cw) < self.min_pixels {
            let scale =
                (self.min_pixels as f64 / (frames as f64 * height as f64 * width as f64)).sqrt();
            let raised_h = ((height as f64 * scale).ceil() as usize).max(1);
            let raised_w = ((width as f64 * scale).ceil() as usize).max(1);
            ch = self.align(raised_h)?;
            cw = self.align(raised_w)?;
        }
        if budget(ch, cw) > self.max_pixels {
            // Match the vendor search: the upper bound is the original height.
            let (mut low, mut high) = (1usize, height);
            (ch, cw) = (self.factor, self.factor);
            while low <= high {
                let mid = low + (high - low) / 2;
                let content_w = (width as u128)
                    .saturating_mul(mid as u128)
                    .div_euclid(height as u128)
                    .max(1)
                    .min(usize::MAX as u128) as usize;
                let candidate_h = self.align(mid)?;
                let candidate_w = self.align(content_w)?;
                if budget(candidate_h, candidate_w) <= self.max_pixels {
                    (ch, cw) = (candidate_h, candidate_w);
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
        }
        let mut scale = (ch as f64 / height as f64).min(cw as f64 / width as f64);
        if budget(height, width) >= self.min_pixels {
            scale = scale.min(1.0);
        }
        let content_h = ((height as f64 * scale).floor() as usize).clamp(1, ch);
        let content_w = ((width as f64 * scale).floor() as usize).clamp(1, cw);
        Ok(GlmResize {
            canvas_h: ch,
            canvas_w: cw,
            content_h,
            content_w,
        })
    }

    fn preprocess_image(
        &self,
        rgb: &[u8],
        height: usize,
        width: usize,
    ) -> Result<ProcessedItem, String> {
        if rgb.len()
            != height
                .checked_mul(width)
                .and_then(|n| n.checked_mul(3))
                .ok_or("glm_vl: image dimensions overflow")?
        {
            return Err("glm_vl: invalid RGB buffer length".into());
        }
        let g = self.resize_geometry(height, width)?;
        let resized;
        let content = if (g.content_h, g.content_w) == (height, width) {
            rgb
        } else {
            resized = resize::resize_rgb(
                rgb,
                height,
                width,
                g.content_h,
                g.content_w,
                self.spec.resample.into(),
            );
            &resized
        };
        let mut padded = vec![
            0u8;
            g.canvas_h
                .checked_mul(g.canvas_w)
                .and_then(|n| n.checked_mul(3))
                .ok_or("glm_vl: canvas overflow")?
        ];
        for y in 0..g.content_h {
            let src = y * g.content_w * 3;
            let dst = y * g.canvas_w * 3;
            padded[dst..dst + g.content_w * 3]
                .copy_from_slice(&content[src..src + g.content_w * 3]);
        }
        let (gh, gw) = (
            g.canvas_h / self.spec.patch_size,
            g.canvas_w / self.spec.patch_size,
        );
        if gh == 0 || gw == 0 || gh % self.spec.merge_size != 0 || gw % self.spec.merge_size != 0 {
            return Err("glm_vl: invalid patch grid".into());
        }
        let pixels = grid::patchify(
            &padded,
            g.canvas_h,
            g.canvas_w,
            self.spec.patch_size,
            self.spec.merge_size,
            self.spec.temporal_patch_size,
            &self.lut,
        );
        let gh32 = u32::try_from(gh).map_err(|_| "glm_vl: grid height overflow")?;
        let gw32 = u32::try_from(gw).map_err(|_| "glm_vl: grid width overflow")?;
        Ok(ProcessedItem {
            feature: Tensor {
                shape: vec![gh * gw, pixels.len() / (gh * gw)],
                data: TensorData::F32(pixels),
            },
            aux: vec![(
                "image_grid_thw".into(),
                Tensor {
                    shape: vec![3],
                    data: TensorData::I64(vec![1, gh as i64, gw as i64]),
                },
            )],
            geometry: Geometry::Grid([1, gh32, gw32]),
        })
    }
}

impl MmFamilyProcessor for GlmVlProcessor {
    /// Apply EXIF orientation and composite transparency before converting to
    /// RGB, matching GLM's smart_to_rgb behavior.
    fn decode_image(&self, bytes: &[u8]) -> Result<DecodedMedia, String> {
        let reader = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| format!("glm_vl image decode: {e}"))?;
        let mut decoder = reader
            .into_decoder()
            .map_err(|e| format!("glm_vl image decode: {e}"))?;
        let orientation = decoder
            .orientation()
            .map_err(|e| format!("glm_vl image orientation: {e}"))?;
        let mut image =
            DynamicImage::from_decoder(decoder).map_err(|e| format!("glm_vl image decode: {e}"))?;
        if !matches!(
            image.color(),
            image::ColorType::L8
                | image::ColorType::La8
                | image::ColorType::Rgb8
                | image::ColorType::Rgba8
        ) {
            return Err(format!(
                "glm_vl image decode: unsupported color {:?}",
                image.color()
            ));
        }
        image.apply_orientation(orientation);
        let (w, h) = (image.width() as usize, image.height() as usize);
        let rgb = if image.color().has_alpha() {
            composite_alpha(image.to_rgba8().as_raw(), w, h)
        } else {
            image.to_rgb8().into_raw()
        };
        Ok(DecodedMedia::Image {
            rgb,
            height: h,
            width: w,
        })
    }

    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String> {
        let DecodedMedia::Image { rgb, height, width } = media;
        self.preprocess_image(rgb, *height, *width)
    }

    fn layout(&self, input_ids: &[i32], items: &[Geometry]) -> Result<TokenLayout, String> {
        if input_ids
            .iter()
            .any(|&id| id == self.spec.video_start_token_id || id == self.spec.video_end_token_id)
        {
            return Err("glm_vl: video wrapper token in image prompt".into());
        }
        let counts = items
            .iter()
            .map(|Geometry::Grid([t, h, w])| {
                let patches = (*t as usize)
                    .checked_mul(*h as usize)
                    .and_then(|v| v.checked_mul(*w as usize))
                    .ok_or("glm_vl: token count overflow")?;
                let merged = self
                    .spec
                    .merge_size
                    .checked_mul(self.spec.merge_size)
                    .ok_or("glm_vl: merge size overflow")?;
                if patches % merged != 0 {
                    return Err("glm_vl: patch grid is not divisible by merge size".into());
                }
                Ok(patches / merged)
            })
            .collect::<Result<Vec<_>, String>>()?;
        token_layout::layout_by_placeholder(input_ids, self.spec.image_token_id, &counts)
    }

    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[Geometry],
    ) -> Result<PositionOutput, String> {
        if offsets.len() != items.len() {
            return Err("glm_vl: offsets/grid count mismatch".into());
        }
        let entries = offsets
            .iter()
            .zip(items)
            .map(|(&(start, end), Geometry::Grid(grid))| grid::MropeItem {
                start,
                end,
                grid: *grid,
            })
            .collect::<Vec<_>>();
        let (positions, delta) = grid::mrope_image_only(input_len, &entries, self.spec.merge_size)?;
        Ok(PositionOutput::MRope { positions, delta })
    }
}

/// PIL's edge-based background selection and alpha-masked paste.
fn composite_alpha(rgba: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut sum = 0u64;
    let mut count = 0u64;
    let mut sample = |x: usize, y: usize| {
        let p = &rgba[(y * w + x) * 4..][..4];
        if p[3] > 128 {
            sum += u64::from(p[0]) + u64::from(p[1]) + u64::from(p[2]);
            count += 1;
        }
    };
    for x in (0..w).step_by((w / 20).max(1)) {
        sample(x, 0);
        sample(x, h - 1);
    }
    for y in (0..h).step_by((h / 20).max(1)) {
        sample(0, y);
        sample(w - 1, y);
    }
    let bg = if count == 0 {
        255u32
    } else if sum > count * 3 * 128 {
        32
    } else {
        240
    };
    rgba.chunks_exact(4)
        .flat_map(|p| {
            let alpha = u32::from(p[3]);
            [0, 1, 2].map(|c| ((u32::from(p[c]) * alpha + bg * (255 - alpha) + 127) / 255) as u8)
        })
        .collect()
}

// Expose the production family pipeline for HF parity checks. No image
// preprocessing algorithm is duplicated in these bindings.
#[cfg(feature = "python")]
mod python {
    use numpy::{IntoPyArray, PyArray1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::*;

    type PyProcessedImage<'py> = (Bound<'py, PyArray1<f32>>, (u32, u32, u32));
    type PyNativeOutput<'py> = (
        Vec<i32>,
        Bound<'py, PyArray1<f32>>,
        Vec<(u32, u32, u32)>,
        Vec<u64>,
        Vec<(u32, u32)>,
        Bound<'py, PyArray1<i64>>,
        i64,
    );

    /// Decode and preprocess one encoded image through the GLM server family.
    #[pyfunction]
    fn preprocess<'py>(
        py: Python<'py>,
        data: Vec<u8>,
        spec_json: String,
    ) -> PyResult<PyProcessedImage<'py>> {
        let result = py
            .detach(move || {
                let family = crate::registry::pipeline_from_spec(&spec_json)?;
                let media = family.decode_image(&data)?;
                family.process_item(&media)
            })
            .map_err(PyValueError::new_err)?;
        let Geometry::Grid([t, h, w]) = result.geometry;
        let TensorData::F32(values) = result.feature.data else {
            return Err(PyValueError::new_err("glm_vl: expected f32 feature"));
        };
        Ok((values.into_pyarray(py), (t, h, w)))
    }

    /// Run the real driver and packer for a tokenized image request.
    #[pyfunction]
    fn process_mm<'py>(
        py: Python<'py>,
        input_ids: Vec<i32>,
        images: Vec<Vec<u8>>,
        spec_json: String,
    ) -> PyResult<PyNativeOutput<'py>> {
        let packed = py
            .detach(move || {
                let family = crate::registry::pipeline_from_spec(&spec_json)?;
                let input = crate::driver::MmInput {
                    text: None,
                    input_ids: Some(input_ids),
                    images: images
                        .into_iter()
                        .map(crate::driver::ImageSource::Bytes)
                        .collect(),
                };
                let output = crate::driver::process(family.as_ref(), input, |_| {
                    Err("GLM parity API requires input_ids".into())
                })?;
                crate::grid_packing::pack_grid_output(output, "glm_vl")
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            packed.input_ids,
            packed.features.into_pyarray(py),
            packed
                .grids
                .into_iter()
                .map(|[t, h, w]| (t, h, w))
                .collect(),
            packed.hashes,
            packed.offsets,
            packed.mrope.into_pyarray(py),
            packed.mrope_delta,
        ))
    }

    pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let m = PyModule::new(parent.py(), "glm_vl")?;
        m.add_function(wrap_pyfunction!(preprocess, &m)?)?;
        m.add_function(wrap_pyfunction!(process_mm, &m)?)?;
        parent.add_submodule(&m)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::token_layout;
    use crate::driver::{self, ImageSource, MmInput};
    use crate::pipeline::{Segment, TokenPattern};

    fn spec() -> GlmVlSpec {
        GlmVlSpec {
            image_token_id: 10,
            image_start_token_id: 11,
            image_end_token_id: 12,
            video_start_token_id: 13,
            video_end_token_id: 14,
            patch_size: 14,
            merge_size: 2,
            temporal_patch_size: 2,
            patch_expand_factor: 1,
            min_image_tokens: 16,
            max_image_tokens: 8000,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            resample: Resampler::AtenU8,
        }
    }

    #[test]
    fn geometry_matches_reference_examples() {
        let proc = GlmVlProcessor::new(spec()).unwrap();
        for (h, w, ch, cw, rh, rw, tokens) in [
            (1365, 2048, 1372, 2072, 1365, 2048, 3626),
            (100, 100, 112, 112, 112, 112, 16),
            (20, 20, 112, 112, 112, 112, 16),
            (4000, 3000, 2884, 2156, 2874, 2156, 7931),
            (3000, 4000, 2156, 2884, 2156, 2874, 7931),
            (4096, 4096, 2492, 2492, 2492, 2492, 7921),
            (10, 2000, 28, 2016, 10, 2000, 72),
        ] {
            let got = proc.resize_geometry(h, w).unwrap();
            assert_eq!(
                got,
                GlmResize {
                    canvas_h: ch,
                    canvas_w: cw,
                    content_h: rh,
                    content_w: rw
                },
                "{h}x{w}"
            );
            assert_eq!((ch / 14) * (cw / 14) / 4, tokens);
        }
    }

    #[test]
    fn process_layout_positions_and_padding() {
        let proc = GlmVlProcessor::new(spec()).unwrap();
        let img = image::RgbImage::from_pixel(2000, 10, image::Rgb([255, 0, 0]));
        let mut buffer = Cursor::new(Vec::new());
        img.write_to(&mut buffer, image::ImageFormat::Png).unwrap();
        let result = driver::process(
            &proc,
            MmInput {
                text: None,
                input_ids: Some(vec![11, 10, 12]),
                images: vec![ImageSource::Bytes(buffer.into_inner())],
            },
            |_| unreachable!(),
        )
        .unwrap();
        assert_eq!(result.input_ids.len(), 74);
        assert_eq!(result.offsets, vec![(1, 72)]);
        assert_eq!(result.items[0].feature.shape, [288, 1176]);
        let TensorData::I64(grid) = &result.items[0].aux[0].1.data else {
            panic!("expected grid")
        };
        assert_eq!(grid, &[1, 2, 144]);
        let TensorData::F32(values) = &result.items[0].feature.data else {
            panic!("expected f32")
        };
        assert!(values.iter().any(|&v| v == -1.0)); // zero-filled pad normalizes via LUT[0]
        let PositionOutput::MRope { ref positions, .. } = result.positions else {
            panic!("expected mrope")
        };
        assert_eq!(positions.len(), 3 * result.input_ids.len());
        let packed = crate::grid_packing::pack_grid_output(result, "glm_vl").unwrap();
        assert_eq!(packed.grids, [[1, 2, 144]]);
        assert_eq!(packed.features.len(), 288 * 1176);
        assert_eq!(packed.offsets, [(1, 72)]);
        assert!(
            proc.layout(&[11, 13, 10, 12], &[Geometry::Grid([1, 2, 144])])
                .is_err()
        );
    }

    #[test]
    fn layout_repeats_each_image_in_prompt_order() {
        let proc = GlmVlProcessor::new(spec()).unwrap();
        let ids = [7, 11, 10, 12, 8, 11, 10, 12, 9];
        let grids = [Geometry::Grid([1, 4, 4]), Geometry::Grid([1, 2, 4])];
        let layout = proc.layout(&ids, &grids).unwrap();
        let patterns = layout
            .segments
            .iter()
            .filter_map(|segment| match segment {
                Segment::Media {
                    item,
                    pattern: TokenPattern::Repeat { id, n },
                } => Some((*item, *id, *n)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(patterns, [(0, 10, 4), (1, 10, 2)]);

        let expanded = token_layout::apply_layout(&ids, &layout, grids.len()).unwrap();
        assert_eq!(
            expanded.input_ids,
            [7, 11, 10, 10, 10, 10, 12, 8, 11, 10, 10, 12, 9]
        );
        assert_eq!(expanded.offsets, [(2, 5), (9, 10)]);
        assert!(proc.layout(&ids, &grids[..1]).is_err());
        for wrapper in [proc.spec.video_start_token_id, proc.spec.video_end_token_id] {
            let mut video_ids = ids;
            video_ids[0] = wrapper;
            assert!(proc.layout(&video_ids, &grids).is_err());
        }
    }

    #[test]
    fn transparent_edges_choose_background() {
        let rgba = [255, 0, 0, 0, 0, 0, 0, 255];
        assert_eq!(composite_alpha(&rgba, 2, 1), [240, 240, 240, 0, 0, 0]);
        let rgba = [240, 240, 240, 255, 255, 0, 0, 0];
        assert_eq!(composite_alpha(&rgba, 2, 1)[3..], [32, 32, 32]);
        assert_eq!(composite_alpha(&[0, 0, 0, 0], 1, 1), [255, 255, 255]);
    }

    #[test]
    fn jpeg_exif_orientation_is_applied_before_rgb() {
        let img = image::RgbImage::from_fn(3, 2, |x, y| image::Rgb([x as u8, y as u8, 0]));
        let mut jpeg = Cursor::new(Vec::new());
        img.write_to(&mut jpeg, image::ImageFormat::Jpeg).unwrap();
        let jpeg = jpeg.into_inner();
        let exif: [u8; 32] = [
            b'E', b'x', b'i', b'f', 0, 0, // Exif identifier
            b'I', b'I', 42, 0, 8, 0, 0, 0, // little-endian TIFF
            1, 0, // one IFD entry
            0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, // orientation=6
            0, 0, 0, 0, // no next IFD
        ];
        let len = (exif.len() + 2) as u16;
        let mut oriented = vec![0xff, 0xd8, 0xff, 0xe1];
        oriented.extend_from_slice(&len.to_be_bytes());
        oriented.extend_from_slice(&exif);
        oriented.extend_from_slice(&jpeg[2..]);
        let proc = GlmVlProcessor::new(spec()).unwrap();
        let DecodedMedia::Image { height, width, .. } = proc.decode_image(&oriented).unwrap();
        assert_eq!((height, width), (3, 2));
    }

    #[test]
    fn rgba_png_decode_composites_alpha() {
        let image = image::RgbaImage::from_raw(2, 1, vec![0, 0, 0, 0, 240, 240, 240, 255]).unwrap();
        let mut png = Cursor::new(Vec::new());
        image.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let proc = GlmVlProcessor::new(spec()).unwrap();
        let DecodedMedia::Image { rgb, height, width } =
            proc.decode_image(&png.into_inner()).unwrap();
        assert_eq!((height, width), (1, 2));
        assert_eq!(rgb, [32, 32, 32, 240, 240, 240]);
    }

    #[test]
    fn invalid_budget_rejected() {
        let mut s = spec();
        s.patch_expand_factor = 4;
        s.max_image_tokens = 1;
        assert!(GlmVlProcessor::new(s).is_err());
    }
}
