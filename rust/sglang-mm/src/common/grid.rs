//! Shared grid-image normalization, patch flattening, and image-only M-RoPE.

use super::par;
use crate::qwen_vl::Resampler;

const INV_RESCALE: f32 = 255.0;

pub fn normalize_lut(resample: Resampler, mean: f32, std: f32) -> [f32; 256] {
    match resample {
        Resampler::Pil => core::array::from_fn(|v| (v as f32 / INV_RESCALE - mean) / std),
        Resampler::AtenU8 => {
            let (mean, std) = (mean * INV_RESCALE, std * INV_RESCALE);
            core::array::from_fn(|v| (v as f32 - mean) / std)
        }
    }
}

/// HF flatten order: `(gh/m, gw/m, m, m)` patches, `(C, tps, ps, ps)` features.
pub fn patchify(
    rgb: &[u8],
    h: usize,
    w: usize,
    ps: usize,
    m: usize,
    tps: usize,
    lut: &[[f32; 256]; 3],
) -> Vec<f32> {
    let (gh, gw) = (h / ps, w / ps);
    let dim = 3 * tps * ps * ps;
    let block_row = gw * m * dim;
    let mut out = vec![0.0f32; gh * gw * dim];
    par::for_chunks_mut(&mut out, block_row, |i, chunk| {
        let mut p = 0;
        for j in 0..gw / m {
            for mh in 0..m {
                for mw in 0..m {
                    let y0 = (i * m + mh) * ps;
                    let x0 = (j * m + mw) * ps;
                    let patch = &mut chunk[p * dim..(p + 1) * dim];
                    for c in 0..3 {
                        let ch = &mut patch[c * tps * ps * ps..];
                        for py in 0..ps {
                            let src = ((y0 + py) * w + x0) * 3 + c;
                            for px in 0..ps {
                                ch[py * ps + px] = lut[c][rgb[src + px * 3] as usize];
                            }
                        }
                        let (t0, rest) = ch.split_at_mut(ps * ps);
                        for t in 0..tps - 1 {
                            rest[t * ps * ps..(t + 1) * ps * ps].copy_from_slice(t0);
                        }
                    }
                    p += 1;
                }
            }
        }
    });
    out
}

pub struct MropeItem {
    pub start: u32,
    pub end: u32,
    pub grid: [u32; 3],
}

pub fn mrope_image_only(
    input_len: usize,
    items: &[MropeItem],
    merge_size: usize,
) -> Result<(Vec<i64>, i64), String> {
    if merge_size == 0 {
        return Err("mrope: merge_size must be positive".into());
    }
    let len = input_len;
    let mut pos = vec![0i64; 3 * len];
    let fill_text = |st: usize, n: usize, base: i64, pos: &mut [i64]| {
        for k in 0..n {
            let v = base + k as i64;
            pos[st + k] = v;
            pos[len + st + k] = v;
            pos[2 * len + st + k] = v;
        }
    };
    let mut st = 0usize;
    let mut next_pos = 0i64;
    for item in items {
        let (start, end) = (item.start as usize, item.end as usize);
        if start < st || end >= len {
            return Err(format!(
                "mrope: item range ({start},{end}) out of order/bounds"
            ));
        }
        fill_text(st, start - st, next_pos, &mut pos);
        next_pos += (start - st) as i64;
        let t = item.grid[0] as usize;
        let gh = item.grid[1] as usize / merge_size;
        let gw = item.grid[2] as usize / merge_size;
        if t * gh * gw != end - start + 1 {
            return Err("mrope: token span does not match grid".into());
        }
        for ti in 0..t {
            for hi in 0..gh {
                for wi in 0..gw {
                    let idx = start + (ti * gh + hi) * gw + wi;
                    pos[idx] = next_pos + ti as i64;
                    pos[len + idx] = next_pos + hi as i64;
                    pos[2 * len + idx] = next_pos + wi as i64;
                }
            }
        }
        next_pos += (t.max(gh).max(gw)) as i64;
        st = end + 1;
    }
    if st < len {
        fill_text(st, len - st, next_pos, &mut pos);
    }
    let max = pos.iter().copied().max().unwrap_or(-1);
    Ok((pos, max + 1 - len as i64))
}
