"""The step-12 debt on the supplied-instance surface may only shrink.

A callee that takes ``server_args`` keeps the supplied-instance contract: the
caller chose the object, so no global-read ratchet counts it. Step 12 changes
what that object *carries* — the instance stays at the user's raw input — so a
callee reading a field **resolution fills in** would start seeing the CLI default
instead of the effective value.

This pins that intersection. Each entry is one (file, field) pair where a
parameter named ``server_args`` is read for a field resolution writes; the plan
doc carries the proposed disposition per field
(``global_context/12-raw-input-config.md``, "the supplied-instance conversion
list"). New pairs fail: a new one is new step-12 work, and the moment to decide
where the value should come from is when the read is written, not during the
flip. Pairs that disappear also fail, with the entry to delete — the list is the
measurement, not a memory of one.

The written-field set is derived here rather than hardcoded: eight
representative configs are resolved and compared against the dataclass defaults,
the same matrix the context repo's audit tool uses.
"""

import ast
import dataclasses
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import sglang
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_PACKAGE_ROOT = Path(next(iter(sglang.__path__))) / "srt"

# The config the resolution pipeline owns; reading the in-flight record is their
# job, not a supplied-instance read.
_OWNERS = ("server_args.py", "runtime_context.py", "arg_groups/")

_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}

# One config resolves only its own decisions, so the written set is a union.
_MATRIX = (
    {},
    {
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    },
    {"dp_size": 2, "tp_size": 2, "enable_dp_attention": True},
    # DWDP resolves dp_size and enable_dp_attention *itself* -- the plain DP
    # entry above passes them in, and passed-in fields are excluded from the
    # written set, so without this entry the dp_size readers would never pin.
    {"tp_size": 2, "dwdp_size": 2},
    {"enable_hierarchical_cache": True, "hicache_ratio": 2.0},
    {"disaggregation_mode": "prefill"},
    {"tp_size": 2, "attn_cp_size": 2},
    {"enable_lora": True, "max_lora_rank": 16},
    {"kv_cache_dtype": "fp8_e4m3", "page_size": 64},
)

_PASSED = frozenset(
    {"model_path", "device", "random_seed", "tokenizer_path", "served_model_name"}
)

_EXPOSED = {
    ("configs/embedding_model_spec.py", "chunked_prefill_size"),
    ("configs/embedding_model_spec.py", "cuda_graph_config"),
    ("configs/model_config.py", "_speculative_draft_quantization_explicitly_set"),
    ("constrained/base_grammar_backend.py", "grammar_backend"),
    ("disaggregation/decode_kvcache_offload_manager.py", "page_size"),
    ("disaggregation/encode_receiver.py", "encoder_transfer_backend"),
    ("disaggregation/encode_server.py", "dp_size"),
    ("disaggregation/encode_server.py", "encoder_transfer_backend"),
    ("disaggregation/encode_server.py", "mm_process_config"),
    (
        "distributed/device_communicators/mooncake_transfer_engine.py",
        "encoder_transfer_backend",
    ),
    ("dllm/config.py", "max_running_requests"),
    ("entrypoints/engine.py", "dp_size"),
    ("entrypoints/engine.py", "enable_dp_attention"),
    ("entrypoints/engine.py", "ep_size"),
    ("entrypoints/http_server.py", "dp_size"),
    ("eplb/eplb_manager.py", "expert_distribution_recorder_buffer_size"),
    ("eplb/expert_distribution.py", "chunked_prefill_size"),
    ("kv_canary/capacities.py", "chunked_prefill_size"),
    ("kv_canary/capacities.py", "cuda_graph_config"),
    ("kv_canary/token_oracle/install.py", "sampling_backend"),
    ("layers/attention/flashattention_backend.py", "enable_dp_attention"),
    ("layers/deep_gemm_wrapper/compile_utils.py", "chunked_prefill_size"),
    ("layers/dp_attention.py", "dp_size"),
    ("layers/dp_attention.py", "enable_dp_attention"),
    ("layers/moe/kt_ep_wrapper.py", "chunked_prefill_size"),
    ("layers/moe/utils.py", "speculative_moe_runner_backend"),
    ("lora/lora_manager.py", "cuda_graph_config"),
    ("lora/lora_manager.py", "enable_dp_attention"),
    ("managers/data_parallel_controller.py", "dp_size"),
    ("managers/data_parallel_controller.py", "enable_dp_attention"),
    (
        "managers/data_parallel_controller.py",
        "enable_dp_attention_local_control_broadcast",
    ),
    ("managers/data_parallel_controller.py", "ep_size"),
    ("managers/data_parallel_controller.py", "load_balance_method"),
    ("managers/load_snapshot.py", "dp_size"),
    ("managers/load_snapshot.py", "enable_dp_attention"),
    ("managers/load_snapshot.py", "load_balance_method"),
    ("managers/prefill_delayer.py", "enable_dp_attention"),
    ("managers/rust_server.py", "mm_process_config"),
    ("managers/scheduler.py", "dp_size"),
    ("managers/scheduler.py", "enable_dp_attention"),
    ("managers/scheduler.py", "ep_size"),
    ("managers/scheduler.py", "page_size"),
    (
        "managers/scheduler_components/new_token_ratio_tracker.py",
        "schedule_conservativeness",
    ),
    ("managers/scheduler_components/recv_skipper.py", "enable_dp_attention"),
    ("managers/tokenizer_control_mixin.py", "dp_size"),
    ("managers/tokenizer_manager.py", "dp_size"),
    ("managers/tokenizer_manager.py", "enable_dp_attention"),
    ("managers/tokenizer_manager.py", "encoder_transfer_backend"),
    ("managers/tokenizer_manager.py", "mm_feature_transport"),
    ("mem_cache/allocation_sizing.py", "page_size"),
    ("mem_cache/kv_cache_builder.py", "chunked_prefill_size"),
    ("mem_cache/kv_cache_builder.py", "enable_dp_attention"),
    ("model_executor/model_runner.py", "page_size"),
    (
        "model_executor/model_runner_components/ngram_embedding_manager.py",
        "chunked_prefill_size",
    ),
    (
        "model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py",
        "cuda_graph_config",
    ),
    ("models/sarvam_moe.py", "attention_backend"),
    ("multimodal/processors/base_processor.py", "mm_feature_transport"),
    ("multimodal/processors/base_processor.py", "mm_process_config"),
    ("ray/data_parallel_controller.py", "dp_size"),
    ("ray/data_parallel_controller.py", "enable_dp_attention"),
    ("ray/engine.py", "dp_size"),
    ("ray/engine.py", "enable_dp_attention"),
    ("speculative/dflash_worker_v2.py", "page_size"),
    ("speculative/dspark_components/dspark_planner.py", "max_running_requests"),
    ("speculative/dspark_components/dspark_worker_v2.py", "cuda_graph_config"),
    ("speculative/dspark_components/dspark_worker_v2.py", "disable_cuda_graph"),
    ("speculative/dspark_components/dspark_worker_v2.py", "enable_dp_attention"),
    ("speculative/dspark_components/dspark_worker_v2.py", "page_size"),
    ("speculative/eagle_worker_v2.py", "enable_dp_attention"),
    ("speculative/eagle_worker_v2.py", "page_size"),
    ("speculative/frozen_kv_mtp_worker_v2.py", "enable_dp_attention"),
    ("speculative/frozen_kv_mtp_worker_v2.py", "page_size"),
    ("speculative/multi_layer_eagle_worker_v2.py", "enable_dp_attention"),
    ("speculative/multi_layer_eagle_worker_v2.py", "page_size"),
    ("speculative/ngram_worker.py", "page_size"),
    ("speculative/standalone_worker_v2.py", "enable_dp_attention"),
    ("speculative/standalone_worker_v2.py", "page_size"),
    ("utils/common.py", "page_size"),
    ("utils/cuda_vmm_transport_utils.py", "dp_size"),
    ("utils/cuda_vmm_transport_utils.py", "enable_dp_attention"),
    ("utils/cuda_vmm_transport_utils.py", "mm_feature_transport"),
    ("utils/offloader.py", "dp_size"),
}


class TestSuppliedInstanceExposure(CustomTestCase):
    def setUp(self):
        # Resolving the matrix writes process state on the way through (the
        # multimodal transport handler sets SGLANG_USE_CUDA_IPC_TRANSPORT, and
        # `EnvField.set()` flips a descriptor flag `os.environ` does not carry).
        # Leaking it makes *later* files in the same worker fail, which is how
        # this was found -- so the case restores what it touched.
        super().setUp()
        state = (dict(os.environ), self._env_field_flags())
        self.addCleanup(self._restore_process_state, state)

    @staticmethod
    def _env_field_flags() -> dict:
        from sglang.srt.environ import EnvField, envs

        flags = {}
        for klass in reversed(type(envs).__mro__):
            for name, field in vars(klass).items():
                if isinstance(field, EnvField):
                    flags[name] = field._set_to_none
        return flags

    @staticmethod
    def _restore_process_state(state) -> None:
        from sglang.srt.environ import envs

        saved_environ, saved_flags = state
        os.environ.clear()
        os.environ.update(saved_environ)
        for name, was_none in saved_flags.items():
            getattr(type(envs), name)._set_to_none = was_none

    def _config_dir(self) -> str:
        config_dir = tempfile.mkdtemp(prefix="supplied_instance_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        with open(os.path.join(config_dir, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        return config_dir

    def _resolution_written_fields(self) -> set:
        """The union of what resolution fills in across the matrix.

        Every entry must resolve. A silently skipped one would shrink this set,
        which makes pinned pairs look like they disappeared -- the list would
        then drift by environment rather than by code, and the failure would
        point at the wrong thing.
        """
        model_path = self._config_dir()
        written = set()
        for extra in _MATRIX:
            try:
                resolved = ServerArgs(
                    model_path=model_path, device="cuda", random_seed=42, **extra
                )
            except Exception as exc:
                self.fail(
                    f"the matrix entry {extra} did not resolve in this "
                    f"environment ({type(exc).__name__}: {exc}); the written-field "
                    "union would be short and the pinned list would drift"
                )
            for field in dataclasses.fields(resolved):
                if field.name in _PASSED or field.name in extra:
                    continue
                if field.default is dataclasses.MISSING:
                    continue
                if getattr(resolved, field.name) != field.default:
                    written.add(field.name)
        return written

    def _supplied_instance_reads(self) -> set:
        """Three spellings of the same read: ``server_args.field`` off the
        parameter, ``getattr(server_args, "field", default)`` with a literal
        name, and the *parked* form -- ``self.x = server_args`` in a method
        that takes the parameter, read as ``self.x.field`` anywhere in the
        class. Parking under a different object, a container, or a computed
        name stays invisible, like in every census of this family."""
        pairs = set()
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
            if rel.startswith(_OWNERS):
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                params = {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)}
                if "server_args" not in params:
                    continue
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "server_args"
                        and isinstance(node.ctx, ast.Load)
                    ):
                        pairs.add((rel, node.attr))
                    elif (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "getattr"
                        and node.args
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "server_args"
                        and len(node.args) >= 2
                        and isinstance(node.args[1], ast.Constant)
                        and isinstance(node.args[1].value, str)
                    ):
                        pairs.add((rel, node.args[1].value))
            for cls in ast.walk(tree):
                if not isinstance(cls, ast.ClassDef):
                    continue
                parked = set()
                for fn in cls.body:
                    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    fn_params = {
                        a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)
                    }
                    if "server_args" not in fn_params:
                        continue
                    for node in ast.walk(fn):
                        if (
                            isinstance(node, ast.Assign)
                            and len(node.targets) == 1
                            and isinstance(node.targets[0], ast.Attribute)
                            and isinstance(node.targets[0].value, ast.Name)
                            and node.targets[0].value.id == "self"
                            and isinstance(node.value, ast.Name)
                            and node.value.id == "server_args"
                        ):
                            parked.add(node.targets[0].attr)
                if not parked:
                    continue
                for node in ast.walk(cls):
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.ctx, ast.Load)
                        and isinstance(node.value, ast.Attribute)
                        and node.value.attr in parked
                        and isinstance(node.value.value, ast.Name)
                        and node.value.value.id == "self"
                    ):
                        pairs.add((rel, node.attr))
        return pairs

    def test_the_exposed_set_matches_the_pinned_list(self):
        written = self._resolution_written_fields()
        found = {pair for pair in self._supplied_instance_reads() if pair[1] in written}
        new = sorted(found - _EXPOSED)
        gone = sorted(_EXPOSED - found)
        self.assertEqual(
            ([], []),
            (new, gone),
            "the supplied-instance step-12 surface drifted.\n"
            f"  new (decide where the resolved value comes from): {new}\n"
            f"  gone (delete from _EXPOSED): {gone}",
        )


if __name__ == "__main__":
    unittest.main()
