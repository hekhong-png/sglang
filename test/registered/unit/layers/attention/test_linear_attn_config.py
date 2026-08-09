"""Backend selection in initialize_linear_attn_config.

The function projects the *published* linear-attn config into the module
table. The SM100 GDN auto-default travels as a published override
(`gdn_backend.sm100_flashinfer_default`), so the precedence between an
explicit flag, that default, and the shared base backend is pinned here --
and so is idempotency across re-initialization: the TBO dispatcher builds
three backend replicas, and every one of them must land on the value the
first one published rather than falling back to the base backend.
"""

import unittest

from sglang.srt.layers.attention.linear import utils as linear_utils
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    initialize_linear_attn_config,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLinearAttnConfig(CustomTestCase):
    def setUp(self):
        self.addCleanup(linear_utils._BACKENDS.update, linear_utils._BACKENDS.copy())

    def _publish(self, **fields):
        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)

    def _publish_gdn_default(self, value="flashinfer"):
        get_context().override(
            "gdn_backend.sm100_flashinfer_default",
            linear_attn_prefill_backend=value,
        )

    def _backends(self):
        return (
            linear_utils.get_linear_attn_prefill_backend(),
            linear_utils.get_linear_attn_decode_backend(),
        )

    def test_default_applies_when_the_flag_is_unset(self):
        self._publish(linear_attn_backend="triton")
        self._publish_gdn_default()
        initialize_linear_attn_config()
        prefill, _ = self._backends()
        self.assertEqual(prefill, LinearAttnKernelBackend.FLASHINFER)

    def test_explicit_flag_wins_over_the_default(self):
        self._publish(
            linear_attn_backend="triton",
            linear_attn_prefill_backend="cutedsl",
        )
        initialize_linear_attn_config()
        prefill, _ = self._backends()
        self.assertEqual(prefill, LinearAttnKernelBackend.CUTEDSL)

    def test_base_backend_applies_without_a_default(self):
        self._publish(linear_attn_backend="triton")
        initialize_linear_attn_config()
        prefill, decode = self._backends()
        self.assertEqual(prefill, LinearAttnKernelBackend.TRITON)
        self.assertEqual(decode, LinearAttnKernelBackend.TRITON)

    def test_reinitialization_keeps_the_published_default(self):
        # The TBO dispatcher invokes the backend creator three times; the
        # replicas see the leaf the first call published and must not fall
        # back to the base backend.
        self._publish(linear_attn_backend="triton")
        self._publish_gdn_default()
        for _ in range(3):
            initialize_linear_attn_config()
            prefill, _ = self._backends()
            self.assertEqual(prefill, LinearAttnKernelBackend.FLASHINFER)

    def test_a_recorded_default_shows_in_the_resolved_config(self):
        from sglang.srt.runtime_context import get_exec

        override = get_context().override_server_args(linear_attn_backend="triton")
        server_args = override.install()
        self.addCleanup(override.restore)

        self._publish_gdn_default()
        self.assertEqual(get_exec().mamba.linear_attn_prefill_backend, "flashinfer")
        self.assertEqual(
            get_context().resolved_server_args_dict()["linear_attn_prefill_backend"],
            "flashinfer",
        )
        self.assertIsNone(server_args.linear_attn_prefill_backend)

    def test_the_default_does_not_reach_the_decode_backend(self):
        self._publish(linear_attn_backend="triton")
        self._publish_gdn_default()
        initialize_linear_attn_config()
        _, decode = self._backends()
        self.assertEqual(decode, LinearAttnKernelBackend.TRITON)


if __name__ == "__main__":
    unittest.main()
