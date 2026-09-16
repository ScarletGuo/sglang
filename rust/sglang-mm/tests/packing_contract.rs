//! Direct contracts for the scheduler-drain packing step.
//!
//! The Python parity binding reaches packing through the whole driver, so the
//! rejection branches below are unreachable from there: a driver `Output` that
//! made it this far always carries M-RoPE positions, f32 features and an
//! `image_grid_thw` aux tensor. These fixtures bypass the driver and build
//! `Output` directly, which is the only way to pin the error text and the order
//! the checks run in.
//!
//! Every expectation here is a fixed value, never a comparison against another
//! packing entry point.

use sglang_mm_core::driver::{Output, OutputItem};
use sglang_mm_core::grid_packing;
use sglang_mm_core::pipeline::{PositionOutput, Tensor, TensorData};
use sglang_mm_core::qwen_vl;

/// Names the old public packed-output type, so the retained Qwen entry point's
/// signature is checked by every call below.
fn rejection(result: Result<qwen_vl::QwenPackedOutput, String>) -> String {
    match result {
        Ok(_) => panic!("expected packing to reject this output"),
        Err(message) => message,
    }
}

fn tensor(shape: Vec<usize>, data: TensorData) -> Tensor {
    Tensor { shape, data }
}

fn grid_aux(grid: [i64; 3]) -> (String, Tensor) {
    (
        "image_grid_thw".to_string(),
        tensor(vec![3], TensorData::I64(grid.to_vec())),
    )
}

fn item(feature: Tensor, aux: Vec<(String, Tensor)>, hash: u64) -> OutputItem {
    OutputItem { feature, aux, hash }
}

/// An f32 feature of `len` patch rows' worth of recognizable values.
fn f32_feature(values: &[f32]) -> Tensor {
    tensor(vec![values.len()], TensorData::F32(values.to_vec()))
}

fn mrope(positions: Vec<i64>, delta: i64) -> PositionOutput {
    PositionOutput::MRope { positions, delta }
}

#[test]
fn rope1d_positions_are_rejected() {
    let output = Output {
        input_ids: vec![7, 900, 8],
        items: vec![item(f32_feature(&[1.0]), vec![grid_aux([1, 2, 2])], 11)],
        offsets: vec![(1, 1)],
        positions: PositionOutput::Rope1D,
    };
    assert_eq!(
        rejection(qwen_vl::pack_output(output)),
        "qwen_vl pack: expected M-RoPE positions"
    );
}

#[test]
fn non_f32_primary_feature_is_rejected() {
    let output = Output {
        input_ids: vec![7, 900, 8],
        items: vec![item(
            tensor(vec![2], TensorData::Bf16(vec![0x3f80, 0xbf80])),
            vec![grid_aux([1, 2, 2])],
            11,
        )],
        offsets: vec![(1, 1)],
        positions: mrope(vec![0, 1, 2, 0, 1, 2, 0, 1, 2], 0),
    };
    assert_eq!(
        rejection(qwen_vl::pack_output(output)),
        "qwen_vl pack: expected f32 feature"
    );
}

#[test]
fn missing_grid_aux_is_rejected() {
    let output = Output {
        input_ids: vec![7, 900, 8],
        items: vec![item(f32_feature(&[1.0, 2.0]), vec![], 11)],
        offsets: vec![(1, 1)],
        positions: mrope(vec![0, 1, 2, 0, 1, 2, 0, 1, 2], 0),
    };
    assert_eq!(
        rejection(qwen_vl::pack_output(output)),
        "qwen_vl pack: missing image_grid_thw"
    );
}

#[test]
fn wrong_dtype_grid_aux_is_rejected_as_missing() {
    let output = Output {
        input_ids: vec![7, 900, 8],
        items: vec![item(
            f32_feature(&[1.0, 2.0]),
            vec![(
                "image_grid_thw".to_string(),
                tensor(vec![3], TensorData::F32(vec![1.0, 2.0, 2.0])),
            )],
            11,
        )],
        offsets: vec![(1, 1)],
        positions: mrope(vec![0, 1, 2, 0, 1, 2, 0, 1, 2], 0),
    };
    assert_eq!(
        rejection(qwen_vl::pack_output(output)),
        "qwen_vl pack: missing image_grid_thw"
    );
}

/// Combined-invalid: every check below fails too, so the message pins which one
/// runs first.
#[test]
fn positions_are_checked_before_item_features() {
    let output = Output {
        input_ids: vec![7, 900, 8],
        items: vec![item(
            tensor(vec![1], TensorData::Bf16(vec![0x3f80])),
            vec![],
            11,
        )],
        offsets: vec![(1, 1)],
        positions: PositionOutput::Rope1D,
    };
    assert_eq!(
        rejection(qwen_vl::pack_output(output)),
        "qwen_vl pack: expected M-RoPE positions"
    );
}

/// Combined-invalid: the grid is missing as well, so this pins the feature
/// check as the earlier of the two per-item checks.
#[test]
fn item_features_are_checked_before_the_grid_lookup() {
    let output = Output {
        input_ids: vec![7, 900, 8],
        items: vec![item(
            tensor(vec![1], TensorData::Bf16(vec![0x3f80])),
            vec![],
            11,
        )],
        offsets: vec![(1, 1)],
        positions: mrope(vec![0, 1, 2, 0, 1, 2, 0, 1, 2], 0),
    };
    assert_eq!(
        rejection(qwen_vl::pack_output(output)),
        "qwen_vl pack: expected f32 feature"
    );
}

/// The success path, with two items whose features, grids and hashes are all
/// distinct so concatenation order is observable. `-0.0` and the signalling
/// values are compared by bit pattern, which `==` on f32 would not catch.
#[test]
fn two_items_pack_in_prompt_order() {
    let first = [-0.0f32, 0.0, 1.5, -2.25];
    let second = [3.5f32, -0.0];
    let positions = vec![
        0, 1, 1, 2, 3, 3, //
        0, 1, 2, 2, 3, 4, //
        0, 1, 1, 3, 3, 4,
    ];
    let output = Output {
        input_ids: vec![7, 900, 900, 8, 900, 9],
        items: vec![
            item(f32_feature(&first), vec![grid_aux([1, 4, 4])], 0xA1),
            item(f32_feature(&second), vec![grid_aux([1, 2, 2])], 0xB2),
        ],
        offsets: vec![(1, 2), (4, 4)],
        positions: mrope(positions.clone(), 3),
    };

    let packed: qwen_vl::QwenPackedOutput = match qwen_vl::pack_output(output) {
        Ok(packed) => packed,
        Err(message) => panic!("expected packing to succeed: {message}"),
    };

    assert_eq!(packed.input_ids, vec![7, 900, 900, 8, 900, 9]);
    let expected_bits: Vec<u32> = first
        .iter()
        .chain(second.iter())
        .map(|value| value.to_bits())
        .collect();
    let packed_bits: Vec<u32> = packed
        .features
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(packed_bits, expected_bits);
    assert_eq!(packed.grids, vec![[1, 4, 4], [1, 2, 2]]);
    assert_eq!(packed.hashes, vec![0xA1, 0xB2]);
    assert_eq!(packed.offsets, vec![(1, 2), (4, 4)]);
    assert_eq!(packed.mrope, positions);
    assert_eq!(packed.mrope_delta, 3);
}

/// The `family` label is the only thing that varies between grid families, so
/// each rejection is pinned again under a non-qwen label. A label that leaked
/// into anything but the message prefix would show up as a changed suffix.
#[test]
fn a_second_family_label_only_changes_the_message_prefix() {
    let rope1d = Output {
        input_ids: vec![7, 900, 8],
        items: vec![item(f32_feature(&[1.0]), vec![grid_aux([1, 2, 2])], 11)],
        offsets: vec![(1, 1)],
        positions: PositionOutput::Rope1D,
    };
    assert_eq!(
        rejection(grid_packing::pack_grid_output(rope1d, "other_vl")),
        "other_vl pack: expected M-RoPE positions"
    );

    let wrong_dtype = Output {
        input_ids: vec![7, 900, 8],
        items: vec![item(
            tensor(vec![2], TensorData::Bf16(vec![0x3f80, 0xbf80])),
            vec![grid_aux([1, 2, 2])],
            11,
        )],
        offsets: vec![(1, 1)],
        positions: mrope(vec![0, 1, 2, 0, 1, 2, 0, 1, 2], 0),
    };
    assert_eq!(
        rejection(grid_packing::pack_grid_output(wrong_dtype, "other_vl")),
        "other_vl pack: expected f32 feature"
    );

    let no_grid = Output {
        input_ids: vec![7, 900, 8],
        items: vec![item(f32_feature(&[1.0, 2.0]), vec![], 11)],
        offsets: vec![(1, 1)],
        positions: mrope(vec![0, 1, 2, 0, 1, 2, 0, 1, 2], 0),
    };
    assert_eq!(
        rejection(grid_packing::pack_grid_output(no_grid, "other_vl")),
        "other_vl pack: missing image_grid_thw"
    );
}

/// `pack_output` must stay a pure `"qwen_vl"`-labelled call into the moved
/// function: same packed values through both entry points.
#[test]
fn the_retained_qwen_entry_point_forwards_to_the_moved_function() {
    let values = [1.0_f32, 2.0, 3.0, 4.0];
    let positions = vec![0, 1, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2];
    let build = || Output {
        input_ids: vec![7, 900, 900, 8],
        items: vec![item(f32_feature(&values), vec![grid_aux([1, 2, 2])], 0xC3)],
        offsets: vec![(1, 2)],
        positions: mrope(positions.clone(), 2),
    };

    let through_wrapper = qwen_vl::pack_output(build()).expect("wrapper packs");
    let through_moved =
        grid_packing::pack_grid_output(build(), "qwen_vl").expect("moved function packs");

    assert_eq!(through_wrapper.input_ids, through_moved.input_ids);
    assert_eq!(through_wrapper.features, through_moved.features);
    assert_eq!(through_wrapper.grids, through_moved.grids);
    assert_eq!(through_wrapper.hashes, through_moved.hashes);
    assert_eq!(through_wrapper.offsets, through_moved.offsets);
    assert_eq!(through_wrapper.mrope, through_moved.mrope);
    assert_eq!(through_wrapper.mrope_delta, through_moved.mrope_delta);
}
