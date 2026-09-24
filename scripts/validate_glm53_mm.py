#!/usr/bin/env python3
"""Pinned HF vs production Rust GLM image-pipeline validation.

Build only the sglang-mm crate; no editable SGLang install is needed. Cargo
and the Rust 1.92 toolchain pinned by rust/rust-toolchain.toml are required:

    PYO3_PYTHON="$(command -v python)" RUSTUP_TOOLCHAIN=1.92 \
      cargo build --manifest-path rust/Cargo.toml \
      -p sglang-mm --features python,parallel --release --locked

Use --hf-only to collect references before the extension is available. The
script exits nonzero for mismatches and writes a machine-readable report.
"""

from __future__ import annotations

import argparse
from importlib.machinery import ExtensionFileLoader
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoConfig, AutoProcessor

MODEL = "zai-org/GLM-5.3-Flash"
REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
REQUIRED_IMAGE_CONFIG = {
    "patch_size": 14,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "patch_expand_factor": 1,
    "min_image_tokens": 16,
    "max_image_tokens": 8000,
    "resample": 3,
    "rescale_factor": 1 / 255,
    "do_convert_rgb": True,
    "do_resize": True,
    "do_rescale": True,
    "do_normalize": True,
    "size": {"longest_edge": 1},
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
}
BACKENDS = {
    "fast": (
        True,
        "transformers.models.glm5_next.image_processing_glm5_next",
        "Glm5NextImageProcessor",
        "aten_u8",
    ),
    "pil": (
        False,
        "transformers.models.glm5_next.image_processing_pil_glm5_next",
        "Glm5NextImageProcessorPil",
        "pil",
    ),
}


def encoded(image: Image.Image, fmt: str = "PNG", **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def fixtures() -> dict[str, bytes]:
    result = {"constant": encoded(Image.new("RGB", (1024, 768), (123, 45, 67)))}
    for name, (width, height) in {
        "gradient_landscape": (301, 199),
        "gradient_portrait": (85, 127),
        "tiny_odd": (19, 17),
        "large_gradient": (1024, 768),
    }.items():
        y, x = np.mgrid[:height, :width]
        rgb = np.stack(
            np.broadcast_arrays((x * 17 + y * 3) % 256, (y * 29 + x) % 256, (x * 7 + y * 11) % 256),
            axis=-1,
        ).astype(np.uint8)
        result[name] = encoded(Image.fromarray(rgb, "RGB"))
    rgba = Image.new("RGBA", (131, 79))
    rgba.putdata(
        [((x * 3) % 256, (y * 7) % 256, 91, (x + y) % 256)
         for y in range(79) for x in range(131)]
    )
    result["rgba"] = encoded(rgba)
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise when applying orientation.
    result["exif_rotated"] = encoded(
        Image.new("RGB", (91, 57), (23, 117, 208)), "JPEG", exif=exif
    )
    return result


def compare(actual: np.ndarray, expected: np.ndarray, atol: float) -> dict:
    info = {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape),
            "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype)}
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {**info, "pass": False, "reason": "shape or dtype mismatch"}
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    finite = np.isfinite(delta)
    bad = np.flatnonzero((delta > atol) | ~finite)
    return {
        **info,
        "pass": len(bad) == 0,
        "exact": bool(np.array_equal(actual, expected)),
        "max_abs_diff": float(np.max(delta)) if delta.size and finite.all() else None,
        "mean_abs_diff": float(np.mean(delta)) if delta.size and finite.all() else None,
        "mismatch_count": int(len(bad)),
        "first_mismatch_flat_index": int(bad[0]) if len(bad) else None,
    }


def manifest() -> dict:
    return {
        "model": MODEL, "revision": REVISION, "python": sys.version,
        "platform": platform.platform(),
        "versions": {name: version(name) for name in
                     ("transformers", "torch", "torchvision", "numpy", "Pillow", "huggingface_hub")},
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "hf_output_device": "recorded per case; this does not establish the compute device",
    }


def find_rust_library(override: Path | None) -> tuple[Path, list[Path]]:
    """Locate Cargo's release cdylib, including a custom target directory."""
    if override is not None:
        candidates = [override.expanduser().resolve()]
    else:
        repo = Path(__file__).resolve().parents[1]
        target_dirs = [repo / "rust" / "target"]
        if target := os.getenv("CARGO_TARGET_DIR"):
            target_dirs.insert(0, Path(target).expanduser().resolve())
        try:
            result = subprocess.run(
                ["cargo", "metadata", "--format-version", "1", "--no-deps",
                 "--manifest-path", str(repo / "rust" / "Cargo.toml")],
                capture_output=True, text=True, check=True, timeout=15,
            )
            target_dirs.insert(0, Path(json.loads(result.stdout)["target_directory"]))
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, KeyError):
            pass
        candidates = [directory / "release" / "libsglang_mm_core.so" for directory in target_dirs]
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return next((candidate for candidate in unique if candidate.is_file()), unique[0]), unique


def load_processor(use_fast: bool, local_files_only: bool):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        processor = AutoProcessor.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=False,
            use_fast=use_fast, local_files_only=local_files_only,
        )
    return processor, [str(item.message) for item in caught]


def make_spec(ip, hf_config, resample: str) -> str:
    image = ip.to_dict()
    names = ("image_token_id", "image_start_token_id", "image_end_token_id",
             "video_start_token_id", "video_end_token_id")
    spec = {name: int(getattr(hf_config, name)) for name in names}
    spec.update({name: image[name] for name in (
        "patch_size", "merge_size", "temporal_patch_size", "patch_expand_factor",
        "min_image_tokens", "max_image_tokens", "image_mean", "image_std")})
    spec.update(family="glm_vl", resample=resample)
    return json.dumps(spec)


def serving_pil_rgb(image: Image.Image) -> Image.Image:
    """Mirror SGLang's smart_to_rgb for the PIL image-load path.

    Keep this in sync with python/sglang/srt/utils/common.py. SGLang may
    instead decode JPEG to a GPU tensor, which this reference does not cover.
    """
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = image.convert("RGBA")
        width, height = image.size
        edge_pixels = []
        for x in range(0, width, max(1, width // 20)):
            for y in (0, height - 1):
                pixel = image.getpixel((x, y))
                if pixel[3] > 128:
                    edge_pixels.append(pixel[:3])
        for y in range(0, height, max(1, height // 20)):
            for x in (0, width - 1):
                pixel = image.getpixel((x, y))
                if pixel[3] > 128:
                    edge_pixels.append(pixel[:3])
        if edge_pixels:
            avg_brightness = sum(sum(pixel) for pixel in edge_pixels) / (
                len(edge_pixels) * 3
            )
            background_color = (32, 32, 32) if avg_brightness > 128 else (240, 240, 240)
        else:
            background_color = (255, 255, 255)
        background = Image.new("RGB", image.size, background_color)
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def hf_result(ip, data: bytes):
    with Image.open(io.BytesIO(data)) as image:
        prepared = serving_pil_rgb(image)
        result = ip(images=[prepared], return_tensors="pt")
    values = result["pixel_values"]
    grid = result["image_grid_thw"]
    return values.detach().cpu().numpy(), grid.detach().cpu().numpy(), {
        "pixel_values_device": str(values.device), "image_grid_thw_device": str(grid.device)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("glm53-validation") / REVISION)
    parser.add_argument("--golden-dir", type=Path, help="Compare constant case to existing .npy files")
    parser.add_argument("--hf-only", action="store_true", help="Collect HF references without Rust checks")
    parser.add_argument("--local-files-only", action="store_true", help="Use only the cached checkpoint")
    parser.add_argument("--rust-library", type=Path,
                        help="Built PyO3 shared library; otherwise locate Cargo's release output")
    parser.add_argument("--atol", type=float, default=0.0,
                        help="Absolute Rust parity tolerance; default requires exact values")
    args = parser.parse_args()
    if args.atol < 0 or not np.isfinite(args.atol):
        parser.error("--atol must be finite and nonnegative")
    if version("transformers") != "5.17.0":
        parser.error("this reference requires transformers==5.17.0")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"manifest": manifest(), "checks": {}, "failures": []}
    report["manifest"]["reference_input"] = (
        "SGLang smart_to_rgb PIL path, then pinned HF image processor; "
        "JPEG GPU decode is not covered"
    )
    rust = None
    if not args.hf_only:
        library, checked = find_rust_library(args.rust_library)
        if not library.is_file():
            paths = "\n".join(f"  {path}" for path in checked)
            raise SystemExit(
                "GLM Rust library not found. Checked:\n"
                f"{paths}\n"
                "Confirm the cargo build completed, or pass --rust-library /path/to/libsglang_mm_core.so"
            )
        report["manifest"]["rust_library"] = str(library)
        spec = importlib.util.spec_from_loader(
            "_multimodal", ExtensionFileLoader("_multimodal", str(library))
        )
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load GLM Rust library: {library}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["_multimodal"] = module
        spec.loader.exec_module(module)
        rust = module.glm_vl
    config = AutoConfig.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=False,
        local_files_only=args.local_files_only,
    )
    if getattr(config, "model_type", None) != "glm5_next":
        raise SystemExit("unexpected HF model_type")
    cases = fixtures()
    for name, raw in cases.items():
        suffix = ".jpg" if name == "exif_rotated" else ".png"
        (args.output / f"{name}{suffix}").write_bytes(raw)
    default_processor = AutoProcessor.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=False,
        local_files_only=args.local_files_only,
    )
    default_ip = default_processor.image_processor
    report["checks"]["default_selection"] = {
        "class": type(default_ip).__name__, "module": type(default_ip).__module__,
        "pass": (type(default_ip).__module__, type(default_ip).__name__) == BACKENDS["fast"][1:3],
    }
    if not report["checks"]["default_selection"]["pass"]:
        report["failures"].append("default AutoProcessor backend differs from fast profile")
    outputs = {}
    for label, (use_fast, module, class_name, resample) in BACKENDS.items():
        processor, caught = load_processor(use_fast, args.local_files_only)
        ip = processor.image_processor
        image_config = json.loads(json.dumps(ip.to_dict()))
        profile_match = all(image_config.get(key) == value for key, value in REQUIRED_IMAGE_CONFIG.items())
        identity_match = type(ip).__module__ == module and type(ip).__name__ == class_name
        report["checks"][label] = {
            "processor_class": type(processor).__name__,
            "image_processor_class": type(ip).__name__,
            "image_processor_module": type(ip).__module__,
            "image_config": image_config,
            "warnings": caught,
            "profile_match": profile_match,
            "identity_match": identity_match,
            "cases": {},
        }
        if not profile_match or not identity_match:
            report["failures"].append(f"{label}: processor identity/profile mismatch")
        spec_json = make_spec(ip, config, resample) if rust is not None else None
        outputs[label] = {}
        for name, raw in cases.items():
            try:
                expected, grid, device = hf_result(ip, raw)
            except Exception as exc:
                report["checks"][label]["cases"][name] = {"error": repr(exc)}
                report["failures"].append(f"{label}/{name}: HF preprocessing failed: {exc!r}")
                continue
            np.save(args.output / f"{name}_{label}_pixel_values.npy", expected)
            np.save(args.output / f"{name}_{label}_grid.npy", grid)
            entry = {"grid": grid.tolist(), "shape": list(expected.shape),
                     "dtype": str(expected.dtype), **device}
            entry["hf_contract_pass"] = bool(
                expected.ndim == 2 and expected.dtype == np.float32
                and grid.shape == (1, 3) and grid.dtype == np.int64
                and int(np.prod(grid[0])) == expected.shape[0]
                and np.isfinite(expected).all()
            )
            if not entry["hf_contract_pass"]:
                report["failures"].append(f"{label}/{name}: invalid HF tensor contract")
            outputs[label][name] = (expected, grid)
            if name == "constant" and args.golden_dir:
                for kind, value in (("pixel_values", expected), ("image_grid_thw", grid)):
                    path = args.golden_dir / f"{kind}_{label}.npy"
                    if not path.is_file():
                        report["failures"].append(f"missing golden: {path}")
                    else:
                        result = compare(value, np.load(path, allow_pickle=False), 0)
                        entry[f"saved_golden_{kind}"] = result
                        if not result["pass"]:
                            report["failures"].append(f"{label}/{name}: saved {kind} differs")
            if rust is not None:
                try:
                    flat, rust_grid = rust.preprocess(raw, spec_json)
                    actual = np.asarray(flat, dtype=np.float32)
                    grid_ok = list(rust_grid) == grid[0].tolist()
                    entry["rust_grid"] = list(rust_grid)
                    entry["rust_grid_pass"] = grid_ok
                    if not grid_ok:
                        report["failures"].append(f"{label}/{name}: Rust grid differs")
                    if actual.size == expected.size:
                        actual = actual.reshape(expected.shape)
                    result = compare(actual, expected, args.atol)
                    entry["rust_pixel_values"] = result
                    if not result["pass"]:
                        report["failures"].append(f"{label}/{name}: Rust pixel_values differs")
                except Exception as exc:
                    entry["rust_error"] = repr(exc)
                    report["failures"].append(f"{label}/{name}: Rust preprocessing failed: {exc!r}")
            report["checks"][label]["cases"][name] = entry
        if rust is not None:
            names = ("gradient_landscape", "gradient_portrait")
            image_id = int(getattr(config, "image_token_id"))
            if all(name in outputs[label] for name in names):
                try:
                    source_ids = [42, image_id, 43, image_id, 44]
                    ids, features, grids, hashes, offsets, positions, delta = rust.process_mm(
                        source_ids, [cases[name] for name in names], spec_json
                    )
                    expected_features = np.concatenate([outputs[label][name][0] for name in names], axis=0)
                    expected_grids = [outputs[label][name][1][0].tolist() for name in names]
                    counts = [int(np.prod(g)) // int(image_config["merge_size"]) ** 2 for g in expected_grids]
                    expected_ids = [42] + [image_id] * counts[0] + [43] + [image_id] * counts[1] + [44]
                    expected_offsets = [(1, counts[0]), (counts[0] + 2, counts[0] + counts[1] + 1)]
                    actual_features = np.asarray(features, dtype=np.float32)
                    if actual_features.size == expected_features.size:
                        actual_features = actual_features.reshape(expected_features.shape)
                    multi = {
                        "grids_pass": [list(g) for g in grids] == expected_grids,
                        "ids_pass": ids == expected_ids,
                        "offsets_pass": [tuple(x) for x in offsets] == expected_offsets,
                        "feature_comparison": compare(actual_features, expected_features, args.atol),
                        "hash_count_pass": len(hashes) == 2,
                        "mrope_shape_pass": np.asarray(positions).size == 3 * len(ids),
                        "mrope_delta": int(delta),
                    }
                    report["checks"][label]["multi_image"] = multi
                    if not all(value for key, value in multi.items() if key.endswith("_pass")) or not multi["feature_comparison"]["pass"]:
                        report["failures"].append(f"{label}: multi-image Rust pipeline differs")
                except Exception as exc:
                    report["checks"][label]["multi_image"] = {"error": repr(exc)}
                    report["failures"].append(f"{label}: multi-image Rust pipeline failed: {exc!r}")
            else:
                report["checks"][label]["multi_image"] = {"skipped": "HF single-image case failed"}
            negative_cases = {
                "bad_image": ([42, image_id], [b"not an image"]),
                "missing_placeholder": ([42, 43], [cases[names[0]]]),
                "video_wrapper": ([int(getattr(config, "video_start_token_id")), image_id], [cases[names[0]]]),
            }
            report["checks"][label]["rejections"] = {}
            for case_name, (tokens, images) in negative_cases.items():
                try:
                    rust.process_mm(tokens, images, spec_json)
                except ValueError as exc:
                    report["checks"][label]["rejections"][case_name] = str(exc)
                else:
                    report["failures"].append(f"{label}: {case_name} was accepted")
    for name in cases:
        if name in outputs["fast"] and name in outputs["pil"]:
            fast = outputs["fast"][name][0]
            pil = outputs["pil"][name][0]
            report["checks"][f"fast_vs_pil_{name}"] = compare(fast, pil, 0)
    report["rust_checks_executed"] = rust is not None
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")
    print(f"report: {args.output / 'report.json'}")
    print(f"Rust checks executed: {rust is not None}; failures: {len(report['failures'])}")
    for failure in report["failures"]:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
