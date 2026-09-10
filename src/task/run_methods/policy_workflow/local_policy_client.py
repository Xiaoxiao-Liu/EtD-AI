"""Local LoRA-backed PolicyClient with role-specific DPO adapter dispatch.

Mirrors the public surface of
``src.task.run_methods.policy_workflow.policy_client.PolicyClient`` so
``workflow.py`` can use either backend interchangeably.

Loading
-------
One base ``Qwen3-8B`` in bf16, plus the shared SFT adapter and any enabled
role-specific DPO adapters from
``src.task.train.paths``:

  - ``"sft"``              -> ``model/policy/sft/``
  - ``"dpo_routing"``      -> ``model/policy/dpo_routing/``
  - ``"dpo_sufficiency"``  -> ``model/policy/dpo_sufficiency/``
  - ``"dpo_judgment"``     -> ``model/policy/dpo_judgment/``

The RL-only ablation instead loads three base-initialized, unweighted-DPO
adapters from ``model/policy/rl_only_{routing,sufficiency,judgment}/`` and
does not load the SFT adapter.

DPO was trained as a continuation of SFT and saved as an independent adapter
(see ``src/task/train/dpo/sufficiency/train.py``). It therefore **replaces**
the SFT adapter for the sufficiency call; it does not stack on top.

Dispatch
--------
  - each role activates its own DPO adapter when enabled;
  - disabled roles fall back to ``"sft"``.

Prompts are built via ``src.task.train.sft.data.build_prompt_string`` +
``src.task.train.sft.prompts.render_user`` so they are byte-identical to
the SFT/DPO training format (hand-crafted ChatML, NOT ``apply_chat_template``).

Thread safety
-------------
**Not thread-safe.** Callers must serialise; ``workflow.run`` forces
``workers=1`` whenever the policy is local.
"""

from __future__ import annotations

from typing import Any, Optional

from src.task.run_methods.policy_workflow.policy_client import (
    JUDGE_LABELS,
    PolicyDecision,
    ROUTE_LABELS,
    SUFFICE_LABELS,
    _parse_label,
)
from src.task.train.paths import (
    BASE_MODEL_DIR,
    DPO_JUDGMENT_ADAPTER_DIR,
    DPO_ROUTING_ADAPTER_DIR,
    DPO_SUFFICIENCY_ADAPTER_DIR,
    RL_ONLY_JUDGMENT_ADAPTER_DIR,
    RL_ONLY_ROUTING_ADAPTER_DIR,
    RL_ONLY_SUFFICIENCY_ADAPTER_DIR,
    SFT_ADAPTER_DIR,
)
from src.task.train.sft.data import IM_END, build_prompt_string
from src.task.train.sft.prompts import render_user

_SFT_ADAPTER = "sft"
_DPO_ROUTING_ADAPTER = "dpo_routing"
_DPO_SUFFICIENCY_ADAPTER = "dpo_sufficiency"
_DPO_JUDGMENT_ADAPTER = "dpo_judgment"
_RL_ONLY_ROUTING_ADAPTER = "rl_only_routing"
_RL_ONLY_SUFFICIENCY_ADAPTER = "rl_only_sufficiency"
_RL_ONLY_JUDGMENT_ADAPTER = "rl_only_judgment"


class LocalPolicyClient:
    """Qwen3-8B + LoRA, plays Router / Gatekeeper / Verifier locally."""

    def __init__(
        self,
        *,
        base_model_dir=BASE_MODEL_DIR,
        sft_adapter_dir=SFT_ADAPTER_DIR,
        dpo_routing_adapter_dir=DPO_ROUTING_ADAPTER_DIR,
        dpo_sufficiency_adapter_dir=DPO_SUFFICIENCY_ADAPTER_DIR,
        dpo_judgment_adapter_dir=DPO_JUDGMENT_ADAPTER_DIR,
        rl_only_routing_adapter_dir=RL_ONLY_ROUTING_ADAPTER_DIR,
        rl_only_sufficiency_adapter_dir=RL_ONLY_SUFFICIENCY_ADAPTER_DIR,
        rl_only_judgment_adapter_dir=RL_ONLY_JUDGMENT_ADAPTER_DIR,
        rl_only: bool = False,
        use_dpo_routing: bool = False,
        use_dpo_sufficiency: bool = True,
        use_dpo_judgment: bool = False,
        max_new_tokens: int = 16,
        device: str = "cuda",
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not base_model_dir.exists():
            raise SystemExit(f"base model not found: {base_model_dir}")
        if not rl_only and not sft_adapter_dir.exists():
            raise SystemExit(
                f"SFT adapter not found: {sft_adapter_dir}\n"
                "  -> run SFT first: bash scripts/train/sft/train.sh"
            )
        requested = {
            "routing": (use_dpo_routing, dpo_routing_adapter_dir),
            "sufficiency": (use_dpo_sufficiency, dpo_sufficiency_adapter_dir),
            "judgment": (use_dpo_judgment, dpo_judgment_adapter_dir),
        }
        for role, (enabled, adapter_dir) in requested.items():
            if not rl_only and enabled and not adapter_dir.exists():
                raise SystemExit(
                    f"DPO {role} adapter not found: {adapter_dir}\n"
                    f"  -> train it first, or disable DPO for {role}."
                )
        rl_only_dirs = {
            "routing": rl_only_routing_adapter_dir,
            "sufficiency": rl_only_sufficiency_adapter_dir,
            "judgment": rl_only_judgment_adapter_dir,
        }
        if rl_only:
            for role, adapter_dir in rl_only_dirs.items():
                if not adapter_dir.exists():
                    raise SystemExit(
                        f"RL-only {role} adapter not found: {adapter_dir}\n"
                        "  -> run: bash scripts/train/rl_only/train_all_roles.sh"
                    )

        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit(
                "CUDA not available. Common gotcha on this box: "
                "CUDA_VISIBLE_DEVICES is set to an empty string. "
                "Prefix the command with `CUDA_VISIBLE_DEVICES=0 ` to force GPU 0."
            )

        # Tokenizer: prefer the one saved alongside the SFT adapter (matches
        # training-time tokenizer). Fall back to base if missing.
        preferred_tok_dir = (
            rl_only_routing_adapter_dir if rl_only else sft_adapter_dir
        )
        tok_dir = (
            preferred_tok_dir
            if (preferred_tok_dir / "tokenizer_config.json").exists()
            else base_model_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
        if im_end_id is None or im_end_id == tokenizer.unk_token_id:
            raise RuntimeError(
                f"tokenizer at {tok_dir} has no {IM_END!r} token; "
                "this should be a Qwen3 added-token."
            )

        base = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        base = base.to(device)
        base.config.use_cache = True  # speed up generation
        base.eval()

        if rl_only:
            model = PeftModel.from_pretrained(
                base,
                str(rl_only_routing_adapter_dir),
                adapter_name=_RL_ONLY_ROUTING_ADAPTER,
            )
            model.load_adapter(
                str(rl_only_sufficiency_adapter_dir),
                adapter_name=_RL_ONLY_SUFFICIENCY_ADAPTER,
            )
            model.load_adapter(
                str(rl_only_judgment_adapter_dir),
                adapter_name=_RL_ONLY_JUDGMENT_ADAPTER,
            )
        else:
            model = PeftModel.from_pretrained(
                base, str(sft_adapter_dir), adapter_name=_SFT_ADAPTER
            )
            if use_dpo_routing:
                model.load_adapter(str(dpo_routing_adapter_dir), adapter_name=_DPO_ROUTING_ADAPTER)
            if use_dpo_sufficiency:
                model.load_adapter(
                    str(dpo_sufficiency_adapter_dir),
                    adapter_name=_DPO_SUFFICIENCY_ADAPTER,
                )
            if use_dpo_judgment:
                model.load_adapter(str(dpo_judgment_adapter_dir), adapter_name=_DPO_JUDGMENT_ADAPTER)
        model.eval()

        self._torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.im_end_id = im_end_id
        self.pad_id = tokenizer.pad_token_id
        self.max_new_tokens = max_new_tokens
        self.use_dpo_routing = use_dpo_routing
        self.use_dpo_sufficiency = use_dpo_sufficiency
        self.use_dpo_judgment = use_dpo_judgment
        self.rl_only = rl_only

        # Surface name for logging (mirrors PolicyClient.model_name).
        enabled = [
            role for role, active in (
                ("routing", use_dpo_routing),
                ("sufficiency", use_dpo_sufficiency),
                ("judgment", use_dpo_judgment),
            ) if active
        ]
        self.model_name = (
            "qwen3-8b-policy-rl-only"
            if rl_only
            else "qwen3-8b-policy-sft" + (
                "+dpo_" + "+".join(enabled) if enabled else ""
            )
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _activate(self, task: str) -> None:
        if self.rl_only:
            self.model.set_adapter({
                "routing": _RL_ONLY_ROUTING_ADAPTER,
                "sufficiency": _RL_ONLY_SUFFICIENCY_ADAPTER,
                "judgment": _RL_ONLY_JUDGMENT_ADAPTER,
            }[task])
            return
        role_adapters = {
            "routing": (_DPO_ROUTING_ADAPTER, self.use_dpo_routing),
            "sufficiency": (_DPO_SUFFICIENCY_ADAPTER, self.use_dpo_sufficiency),
            "judgment": (_DPO_JUDGMENT_ADAPTER, self.use_dpo_judgment),
        }
        adapter, enabled = role_adapters[task]
        target = adapter if enabled else _SFT_ADAPTER
        self.model.set_adapter(target)

    def _generate(self, prompt_str: str) -> str:
        torch = self._torch
        inputs = self.tokenizer(
            prompt_str,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.im_end_id,
                pad_token_id=self.pad_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, prompt_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    def _call(
        self,
        task: str,
        input_obj: dict,
        allowed: tuple[str, ...],
    ) -> PolicyDecision:
        self._activate(task)
        user_text = render_user(task, input_obj)
        prompt_str = build_prompt_string(user_text)
        raw = self._generate(prompt_str)
        label = _parse_label(raw, allowed)
        return PolicyDecision(label=label, raw=raw)

    # ------------------------------------------------------------------ #
    # Public surface (matches PolicyClient)
    # ------------------------------------------------------------------ #

    def route(
        self, pico: Any, criterion: str, criterion_question: str
    ) -> PolicyDecision:
        return self._call(
            "routing",
            {
                "PICO": pico,
                "criterion": criterion,
                "criterion_question": criterion_question,
            },
            ROUTE_LABELS,
        )

    def suffice(
        self,
        pico: Any,
        criterion: str,
        criterion_question: str,
        rag_evidence: str,
    ) -> PolicyDecision:
        return self._call(
            "sufficiency",
            {
                "PICO": pico,
                "criterion": criterion,
                "criterion_question": criterion_question,
                "rag_evidence": rag_evidence,
            },
            SUFFICE_LABELS,
        )

    def judge(
        self,
        pico: Any,
        criterion: str,
        criterion_question: str,
        candidate: dict,
        source: str,
        rag_evidence: Optional[str] = None,
        sufficiency_assessment: Optional[str] = None,
    ) -> PolicyDecision:
        input_obj: dict = {
            "PICO": pico,
            "criterion": criterion,
            "criterion_question": criterion_question,
            "source": source,
            "candidate": candidate,
        }
        if source == "rag_evidence" and rag_evidence:
            input_obj["rag_evidence"] = rag_evidence
            input_obj["sufficiency_assessment"] = sufficiency_assessment or ""
        return self._call("judgment", input_obj, JUDGE_LABELS)
