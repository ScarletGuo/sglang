"""GLM's opt-in startup gate against the pinned HF image profile."""

import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor  # noqa: E402
from sglang.srt.rust_server.multimodal import (  # noqa: E402
    RustMmProcessor,
)
from sglang.srt.rust_server.server import RustServer  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VERIFIED_PROFILE = {
    "image_processor_type": "Glm5NextImageProcessor",
    "patch_size": 14,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "patch_expand_factor": 1,
    "min_image_tokens": 16,
    "max_image_tokens": 8000,
    "do_convert_rgb": True,
    "do_resize": True,
    "do_rescale": True,
    "do_normalize": True,
    "resample": 3,
    "rescale_factor": 1 / 255,
    "size": {"longest_edge": 1},
    "image_mean": (0.48145466, 0.4578275, 0.40821073),
    "image_std": (0.26862954, 0.26130258, 0.27577711),
}
VERIFIED_MODULES = {
    "aten_u8": "transformers.models.glm5_next.image_processing_glm5_next",
    "pil": "transformers.models.glm5_next.image_processing_pil_glm5_next",
}


def image_processor(resample="aten_u8", *, config_changes=None, attr_changes=None):
    config = copy.deepcopy(VERIFIED_PROFILE)
    config.update(config_changes or {})
    attrs = {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key != "image_processor_type"
    }
    attrs["backend"] = "pil" if resample == "pil" else "torchvision"
    attrs.update(attr_changes or {})
    attrs["__module__"] = VERIFIED_MODULES[resample]
    attrs["to_dict"] = lambda self: copy.deepcopy(config)
    name = (
        "Glm5NextImageProcessorPil"
        if resample == "pil"
        else "Glm5NextImageProcessor"
    )
    return type(name, (), attrs)()


def processor(image):
    cls = type(
        "Glm5NextProcessor",
        (),
        {"__module__": "transformers.models.glm5_next.processing_glm5_next"},
    )
    instance = cls()
    instance.image_processor = image
    return instance


class TestGlmSpecGate(CustomTestCase):
    def resolve(
        self,
        image,
        *,
        enabled=True,
        mm_config=None,
        hf_config=None,
        processor_override=None,
        transformers_version="5.17.0",
    ):
        host = RustMmProcessor.__new__(RustMmProcessor)
        host.model_config = SimpleNamespace(
            hf_config=hf_config
            or SimpleNamespace(
                model_type="glm5_next",
                image_token_id=10,
                image_start_token_id=11,
                image_end_token_id=12,
                video_start_token_id=13,
                video_end_token_id=14,
            )
        )
        host._processor = processor_override or processor(image)
        host._use_feature_shm = lambda: False
        with (
            patch.dict(
                os.environ,
                {"SGLANG_RUST_MM_GLM5_NEXT": "1" if enabled else "0"},
            ),
            patch(
                "sglang.srt.managers.multimodal_processor.get_mm_processor_cls",
                return_value=Glm4vImageProcessor,
            ),
            patch(
                "sglang.srt.rust_server.multimodal.get_mm",
                return_value=SimpleNamespace(mm_process_config=mm_config),
            ),
            patch(
                "sglang.srt.rust_server.multimodal.version",
                return_value=transformers_version,
            ),
        ):
            return host.resolve_spec()

    def test_opt_in_and_both_valid_backends(self):
        for backend, expected in (("aten_u8", "aten_u8"), ("pil", "pil")):
            with self.subTest(backend=backend):
                image = image_processor(backend)
                self.assertIsNone(self.resolve(image, enabled=False))
                spec = self.resolve(image)
                self.assertIsNotNone(spec)
                self.assertEqual(spec.family, "glm_vl")
                self.assertEqual(spec.resample, expected)
                self.assertEqual(spec.patch_expand_factor, 1)

    def test_rejects_other_transformers_versions(self):
        self.assertIsNone(
            self.resolve(image_processor(), transformers_version="5.16.0")
        )
        self.assertIsNone(
            self.resolve(image_processor(), transformers_version="5.18.0")
        )

    def test_rejects_unknown_identity_and_backend(self):
        image = image_processor()
        type(image).__module__ = "external.image_processing_glm5_next"
        self.assertIsNone(self.resolve(image))
        image = image_processor(attr_changes={"backend": "pil"})
        self.assertIsNone(self.resolve(image))
        image = image_processor()
        type(image).__name__ = "UnverifiedImageProcessor"
        self.assertIsNone(self.resolve(image))

    def test_rejects_outer_processor_override(self):
        image = image_processor()
        host = processor(image)
        type(host).__module__ = "external.processing_glm5_next"
        self.assertIsNone(self.resolve(image, processor_override=host))

    def test_rejects_missing_or_changed_profile_fields(self):
        for field, value in (
            ("do_convert_rgb", False),
            ("do_resize", False),
            ("do_rescale", False),
            ("do_normalize", False),
            ("resample", 2),
            ("size", {"longest_edge": 1024}),
            ("patch_expand_factor", 2),
            ("min_image_tokens", 8),
            ("max_image_tokens", 4000),
            ("image_mean", (float("nan"), 0.4578275, 0.40821073)),
        ):
            with self.subTest(field=field):
                self.assertIsNone(
                    self.resolve(image_processor(config_changes={field: value}))
                )
        for field in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb"):
            with self.subTest(missing=field):
                image = image_processor()
                original = image.to_dict()
                original.pop(field)
                type(image).to_dict = lambda self, data=original: data
                self.assertIsNone(self.resolve(image))
        self.assertIsNone(
            self.resolve(image_processor(config_changes={"unverified_knob": 1}))
        )
        self.assertIsNone(
            self.resolve(image_processor(attr_changes={"do_resize": False}))
        )

    def test_rejects_request_overrides_and_missing_tokens(self):
        self.assertIsNone(
            self.resolve(
                image_processor(), mm_config={"image": {"max_image_tokens": 10}}
            )
        )
        self.assertIsNone(
            self.resolve(image_processor(), mm_config={"video": {"fps": 1}})
        )
        hf = SimpleNamespace(model_type="glm5_next", image_token_id=10)
        self.assertIsNone(self.resolve(image_processor(), hf_config=hf))
        hf = SimpleNamespace(
            model_type="glm5_next",
            image_token_id=2**31,
            image_start_token_id=11,
            image_end_token_id=12,
            video_start_token_id=13,
            video_end_token_id=14,
        )
        self.assertIsNone(self.resolve(image_processor(), hf_config=hf))

    def test_startup_reports_opt_in_or_profile_error(self):
        scheduler = SimpleNamespace(
            server_args=None,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="glm5_next")
            ),
            processor=None,
        )
        with patch("sglang.srt.rust_server.server.RustMmProcessor") as host:
            host.return_value.resolve_spec.return_value = None
            for enabled, message in (
                ("0", "requires SGLANG_RUST_MM_GLM5_NEXT=1"),
                ("1", "unsupported GLM-5.3-Flash Rust MM profile"),
            ):
                with self.subTest(enabled=enabled):
                    with patch.dict(
                        os.environ, {"SGLANG_RUST_MM_GLM5_NEXT": enabled}
                    ):
                        with self.assertRaisesRegex(RuntimeError, message):
                            RustServer.__new__(RustServer)._start_multimodal(
                                scheduler
                            )


if __name__ == "__main__":
    unittest.main()
