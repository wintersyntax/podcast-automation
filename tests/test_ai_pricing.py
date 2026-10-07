"""RED/GREEN contract tests for TASK-076 Task 3: exact preset pricing and
conservative reservation authority.

Runtime/implementation tests here use frozen fixtures only — never a live
OpenRouter fetch — per the design's "External prices are not hard-coded
into this design document. Runtime/implementation tests use frozen
fixtures; production admission uses the current trusted pricing evidence."
"""

from decimal import Decimal
import unittest
from unittest.mock import patch

from podcast_engine.ai_pricing import (
    AudioPricing,
    ConservativeTextPricingBound,
    ModelPricing,
    OPENROUTER_AUDIO_MODELS_URL,
    PricingResolutionError,
    ResolvedPreset,
    derive_audio_reservation_usd,
    derive_text_reservation_usd,
    resolve_audio_model_pricing,
    resolve_model_pricing,
    resolve_preset_model_and_pricing,
    resolve_stage_reservation_usd,
)
from podcast_engine.preset_provenance import PresetProvenance


def _catalog_entry(model_id="prov/model", *, prompt="0.0000008", completion="0.0000024", context_length=128000):
    return {
        "id": model_id,
        "pricing": {"prompt": prompt, "completion": completion},
        "context_length": context_length,
    }


def _audio_catalog_entry(model_id="prov/audio-model", *, usd_per_second="0.000075"):
    """Mirror the real OpenRouter output_modalities=transcription shape.

    TASK-076 Task 9: live evidence (captured 2026-09-21 against
    openai/gpt-transcribe) shows duration-priced audio/STT catalog entries
    reuse the text-pricing "prompt" field name for USD-per-second and
    report context_length=0 / max_completion_tokens=0 -- neither is a
    valid token-pricing bound, so audio resolution must not require them
    the way resolve_model_pricing does for text models.
    """

    return {
        "id": model_id,
        "pricing": {"prompt": usd_per_second, "completion": "0"},
        "context_length": 0,
        "architecture": {"modality": "audio->transcription"},
        "top_provider": {"context_length": 0, "max_completion_tokens": 0},
    }


class ResolveModelPricingTests(unittest.TestCase):
    def test_resolves_pricing_for_the_exact_model_id(self):
        catalog = [
            _catalog_entry("other/model", prompt="0.000001", completion="0.000002"),
            _catalog_entry(
                "prov/model",
                prompt="0.0000008",
                completion="0.0000024",
                context_length=128000,
            ),
        ]

        def transport(url, api_key):
            self.assertEqual(api_key, "test-key")
            return catalog

        pricing = resolve_model_pricing(
            "prov/model",
            api_key="test-key",
            transport=transport,
            now=lambda: "2026-09-16T00:00:00Z",
        )

        self.assertIsInstance(pricing, ModelPricing)
        self.assertEqual(pricing.model_id, "prov/model")
        self.assertEqual(pricing.prompt_usd_per_token, Decimal("0.0000008"))
        self.assertEqual(pricing.completion_usd_per_token, Decimal("0.0000024"))
        self.assertEqual(pricing.context_length, 128000)
        self.assertEqual(pricing.captured_at, "2026-09-16T00:00:00Z")
        self.assertTrue(pricing.evidence_sha256.startswith("sha256:"))

    def test_model_not_in_catalog_fails_closed(self):
        with self.assertRaises(PricingResolutionError):
            resolve_model_pricing(
                "missing/model",
                api_key="test-key",
                transport=lambda url, api_key: [_catalog_entry("other/model")],
                now=lambda: "2026-09-16T00:00:00Z",
            )

    def test_duplicate_model_ids_are_ambiguous(self):
        catalog = [_catalog_entry("dup/model"), _catalog_entry("dup/model")]
        with self.assertRaises(PricingResolutionError):
            resolve_model_pricing(
                "dup/model",
                api_key="test-key",
                transport=lambda url, api_key: catalog,
                now=lambda: "2026-09-16T00:00:00Z",
            )

    def test_non_decimal_pricing_string_fails_closed(self):
        catalog = [_catalog_entry("prov/model", prompt="not-a-number")]
        with self.assertRaises(PricingResolutionError):
            resolve_model_pricing(
                "prov/model",
                api_key="test-key",
                transport=lambda url, api_key: catalog,
                now=lambda: "2026-09-16T00:00:00Z",
            )

    def test_negative_pricing_fails_closed(self):
        catalog = [_catalog_entry("prov/model", prompt="-0.0001")]
        with self.assertRaises(PricingResolutionError):
            resolve_model_pricing(
                "prov/model",
                api_key="test-key",
                transport=lambda url, api_key: catalog,
                now=lambda: "2026-09-16T00:00:00Z",
            )

    def test_missing_context_length_fails_closed(self):
        entry = _catalog_entry("prov/model")
        del entry["context_length"]
        with self.assertRaises(PricingResolutionError):
            resolve_model_pricing(
                "prov/model",
                api_key="test-key",
                transport=lambda url, api_key: [entry],
                now=lambda: "2026-09-16T00:00:00Z",
            )

    def test_empty_model_id_is_rejected(self):
        with self.assertRaises(PricingResolutionError):
            resolve_model_pricing(
                "",
                api_key="test-key",
                transport=lambda url, api_key: [],
                now=lambda: "2026-09-16T00:00:00Z",
            )

    def test_never_reads_the_generated_inspection_snapshot_file(self):
        """Pricing resolution must never read docs/openrouter-presets.md:
        that snapshot is generated inspection material only and can be
        stale relative to the live provider by design.
        """

        catalog = [_catalog_entry("prov/model")]
        with patch(
            "pathlib.Path.read_text",
            side_effect=AssertionError(
                "resolve_model_pricing must not read any file for pricing evidence"
            ),
        ):
            pricing = resolve_model_pricing(
                "prov/model",
                api_key="test-key",
                transport=lambda url, api_key: catalog,
                now=lambda: "2026-09-16T00:00:00Z",
            )

        self.assertEqual(pricing.model_id, "prov/model")


def _verified_preset(*, config, system_prompt="System prompt"):
    return PresetProvenance(
        status="verified",
        slug="podcast-summary",
        preset_id="preset-id",
        version_id="version-id",
        version=1,
        config=config,
        system_prompt=system_prompt,
        config_digest="sha256:config",
        system_prompt_digest="sha256:prompt",
    )



class ResolveAudioModelPricingTests(unittest.TestCase):
    """TASK-076 Task 9: duration-priced audio/STT pricing evidence.

    openai/gpt-transcribe (and other duration-priced STT models) are not
    returned by the default /api/v1/models catalog at all -- OpenRouter
    only lists them when queried with output_modalities=transcription
    (live-confirmed 2026-09-21: the unfiltered catalog had 446 entries and
    zero matches, including zero fuzzy "transcri"/"whisper" matches; the
    filtered catalog had 21 entries including an exact match). Resolution
    must therefore hit a dedicated URL, not filter the general catalog.
    """

    def test_resolves_pricing_for_the_exact_model_id(self):
        catalog = [
            _audio_catalog_entry("other/audio", usd_per_second="0.0001"),
            _audio_catalog_entry("openai/gpt-transcribe", usd_per_second="0.000075"),
        ]

        def transport(url, api_key):
            self.assertEqual(url, OPENROUTER_AUDIO_MODELS_URL)
            self.assertEqual(api_key, "test-key")
            return catalog

        pricing = resolve_audio_model_pricing(
            "openai/gpt-transcribe",
            api_key="test-key",
            transport=transport,
            now=lambda: "2026-09-21T00:00:00Z",
        )

        self.assertIsInstance(pricing, AudioPricing)
        self.assertEqual(pricing.model_id, "openai/gpt-transcribe")
        self.assertEqual(pricing.usd_per_second, Decimal("0.000075"))
        self.assertEqual(pricing.captured_at, "2026-09-21T00:00:00Z")
        self.assertEqual(pricing.source_api, OPENROUTER_AUDIO_MODELS_URL)
        self.assertTrue(pricing.evidence_sha256.startswith("sha256:"))

    def test_zero_context_length_is_accepted(self):
        """Real duration-priced entries report context_length=0; that is
        not a text-pricing bound and must not fail closed here."""

        catalog = [_audio_catalog_entry("prov/audio-model", usd_per_second="0.00005")]
        pricing = resolve_audio_model_pricing(
            "prov/audio-model",
            api_key="test-key",
            transport=lambda url, api_key: catalog,
            now=lambda: "2026-09-21T00:00:00Z",
        )
        self.assertEqual(pricing.usd_per_second, Decimal("0.00005"))

    def test_model_not_in_catalog_fails_closed(self):
        with self.assertRaises(PricingResolutionError):
            resolve_audio_model_pricing(
                "missing/audio-model",
                api_key="test-key",
                transport=lambda url, api_key: [_audio_catalog_entry("other/audio-model")],
                now=lambda: "2026-09-21T00:00:00Z",
            )

    def test_duplicate_model_ids_are_ambiguous(self):
        catalog = [_audio_catalog_entry("dup/audio-model"), _audio_catalog_entry("dup/audio-model")]
        with self.assertRaises(PricingResolutionError):
            resolve_audio_model_pricing(
                "dup/audio-model",
                api_key="test-key",
                transport=lambda url, api_key: catalog,
                now=lambda: "2026-09-21T00:00:00Z",
            )

    def test_non_decimal_pricing_string_fails_closed(self):
        catalog = [_audio_catalog_entry("prov/audio-model", usd_per_second="not-a-number")]
        with self.assertRaises(PricingResolutionError):
            resolve_audio_model_pricing(
                "prov/audio-model",
                api_key="test-key",
                transport=lambda url, api_key: catalog,
                now=lambda: "2026-09-21T00:00:00Z",
            )

    def test_negative_pricing_fails_closed(self):
        catalog = [_audio_catalog_entry("prov/audio-model", usd_per_second="-0.0001")]
        with self.assertRaises(PricingResolutionError):
            resolve_audio_model_pricing(
                "prov/audio-model",
                api_key="test-key",
                transport=lambda url, api_key: catalog,
                now=lambda: "2026-09-21T00:00:00Z",
            )

    def test_zero_pricing_fails_closed(self):
        """A free/zero per-second rate is not credible pricing evidence
        for a paid provider and must not silently reserve $0."""

        catalog = [_audio_catalog_entry("prov/audio-model", usd_per_second="0")]
        with self.assertRaises(PricingResolutionError):
            resolve_audio_model_pricing(
                "prov/audio-model",
                api_key="test-key",
                transport=lambda url, api_key: catalog,
                now=lambda: "2026-09-21T00:00:00Z",
            )

    def test_empty_model_id_is_rejected(self):
        with self.assertRaises(PricingResolutionError):
            resolve_audio_model_pricing(
                "",
                api_key="test-key",
                transport=lambda url, api_key: [],
                now=lambda: "2026-09-21T00:00:00Z",
            )

class ResolvePresetModelAndPricingTests(unittest.TestCase):
    def test_single_model_config_resolves_one_pricing(self):
        preset = _verified_preset(config={"model": "prov/model"})
        catalog = [_catalog_entry("prov/model", prompt="0.000001", completion="0.000003")]
        calls = []

        def transport(url, api_key):
            calls.append(url)
            return catalog

        resolved = resolve_preset_model_and_pricing(
            preset,
            api_key="test-key",
            transport=transport,
            now=lambda: "2026-09-16T00:00:00Z",
        )

        self.assertIsInstance(resolved, ResolvedPreset)
        self.assertEqual(resolved.slug, "podcast-summary")
        self.assertIs(resolved.config, preset.config)
        self.assertIs(resolved.system_prompt, preset.system_prompt)
        self.assertEqual(resolved.pricing_bound.model_ids, ("prov/model",))
        self.assertEqual(resolved.pricing_bound.prompt_usd_per_token, Decimal("0.000001"))
        self.assertEqual(resolved.pricing_bound.completion_usd_per_token, Decimal("0.000003"))
        self.assertEqual(len(calls), 1)

    def test_fallback_model_list_uses_the_most_expensive_candidate(self):
        preset = _verified_preset(
            config={"models": ["cheap/model", "expensive/model"]}
        )
        catalog = [
            _catalog_entry("cheap/model", prompt="0.0000001", completion="0.0000004", context_length=200000),
            _catalog_entry("expensive/model", prompt="0.0000003", completion="0.0000025", context_length=100000),
        ]

        resolved = resolve_preset_model_and_pricing(
            preset,
            api_key="test-key",
            transport=lambda url, api_key: catalog,
            now=lambda: "2026-09-16T00:00:00Z",
        )

        self.assertEqual(
            set(resolved.pricing_bound.model_ids), {"cheap/model", "expensive/model"}
        )
        # Worst-case prompt/completion price: the maximum across candidates,
        # since OpenRouter fallback routing could land on either model.
        self.assertEqual(resolved.pricing_bound.prompt_usd_per_token, Decimal("0.0000003"))
        self.assertEqual(resolved.pricing_bound.completion_usd_per_token, Decimal("0.0000025"))
        # Worst-case context length: the tightest (smallest) across
        # candidates, since that is the least headroom actually available.
        self.assertEqual(resolved.pricing_bound.context_length, 100000)
        self.assertEqual(len(resolved.pricing_bound.pricings), 2)

    def test_primary_and_fallback_models_are_both_in_the_reservation_bound(self):
        preset = _verified_preset(
            config={"model": "cheap/model", "models": ["expensive/model"]}
        )
        catalog = [
            _catalog_entry("cheap/model", prompt="0.0000001", completion="0.0000004", context_length=200000),
            _catalog_entry("expensive/model", prompt="0.0000003", completion="0.0000025", context_length=100000),
        ]

        resolved = resolve_preset_model_and_pricing(
            preset,
            api_key="test-key",
            transport=lambda url, api_key: catalog,
            now=lambda: "2026-09-16T00:00:00Z",
        )

        self.assertEqual(
            resolved.pricing_bound.model_ids,
            ("cheap/model", "expensive/model"),
        )
        self.assertEqual(resolved.pricing_bound.completion_usd_per_token, Decimal("0.0000025"))
        self.assertEqual(resolved.pricing_bound.context_length, 100000)

    def test_unverified_preset_is_rejected(self):
        preset = PresetProvenance(status="unavailable", reason="missing_api_key")
        with self.assertRaises(ValueError):
            resolve_preset_model_and_pricing(
                preset,
                api_key="test-key",
                transport=lambda url, api_key: [],
                now=lambda: "2026-09-16T00:00:00Z",
            )

    def test_preset_missing_model_identity_is_rejected(self):
        preset = _verified_preset(config={"temperature": 0})
        with self.assertRaises(ValueError):
            resolve_preset_model_and_pricing(
                preset,
                api_key="test-key",
                transport=lambda url, api_key: [],
                now=lambda: "2026-09-16T00:00:00Z",
            )

    def test_never_re_fetches_a_mutable_preset_alias(self):
        """The function must bind pricing to the exact snapshot it was
        given, never re-resolve the preset through a fresh (and possibly
        different) mutable @preset/<slug> lookup. Structurally, this
        function takes no preset-fetching transport at all — only a
        pricing transport — so there is nothing it could use to re-fetch
        the preset even if it tried.
        """

        preset = _verified_preset(config={"model": "prov/model"})
        catalog = [_catalog_entry("prov/model")]
        pricing_calls = []

        def pricing_transport(url, api_key):
            pricing_calls.append((url, api_key))
            return catalog

        resolved = resolve_preset_model_and_pricing(
            preset,
            api_key="test-key",
            transport=pricing_transport,
            now=lambda: "2026-09-16T00:00:00Z",
        )

        self.assertEqual(len(pricing_calls), 1)
        self.assertIs(resolved.config, preset.config)


class DeriveTextReservationUsdTests(unittest.TestCase):
    def test_matches_manual_byte_upper_bound_computation(self):
        request_bytes = b'{"batch": "x" * 100}'
        system_prompt = "System prompt text"
        max_completion_tokens = 1000
        prompt_price = Decimal("0.000001")
        completion_price = Decimal("0.000003")

        reservation = derive_text_reservation_usd(
            request_bytes=request_bytes,
            system_prompt=system_prompt,
            max_completion_tokens=max_completion_tokens,
            prompt_usd_per_token=prompt_price,
            completion_usd_per_token=completion_price,
            context_length=1_000_000,
        )

        expected_prompt_tokens = len(request_bytes) + len(system_prompt.encode("utf-8"))
        expected = (
            Decimal(expected_prompt_tokens) * prompt_price
            + Decimal(max_completion_tokens) * completion_price
        )
        self.assertEqual(reservation, expected)

    def test_multibyte_utf8_characters_count_their_full_byte_length(self):
        # "café" encodes to 5 UTF-8 bytes even though it is 4 characters.
        reservation_multibyte = derive_text_reservation_usd(
            request_bytes=b"",
            system_prompt="café",
            max_completion_tokens=1,
            prompt_usd_per_token=Decimal("1"),
            completion_usd_per_token=Decimal("0"),
            context_length=1000,
        )
        self.assertEqual(reservation_multibyte, Decimal(5))

    def test_rejects_a_request_that_exceeds_context_length(self):
        with self.assertRaises(ValueError):
            derive_text_reservation_usd(
                request_bytes=b"x" * 100,
                system_prompt="y" * 100,
                max_completion_tokens=1000,
                prompt_usd_per_token=Decimal("0.000001"),
                completion_usd_per_token=Decimal("0.000001"),
                context_length=500,
            )

    def test_rejects_non_positive_max_completion_tokens(self):
        with self.assertRaises(ValueError):
            derive_text_reservation_usd(
                request_bytes=b"x",
                system_prompt="y",
                max_completion_tokens=0,
                prompt_usd_per_token=Decimal("0.000001"),
                completion_usd_per_token=Decimal("0.000001"),
                context_length=1000,
            )

    def test_rejects_float_pricing_input(self):
        with self.assertRaises(ValueError):
            derive_text_reservation_usd(
                request_bytes=b"x",
                system_prompt="y",
                max_completion_tokens=10,
                prompt_usd_per_token=0.000001,
                completion_usd_per_token=Decimal("0.000001"),
                context_length=1000,
            )

    def test_rejects_boolean_pricing_input(self):
        with self.assertRaises(ValueError):
            derive_text_reservation_usd(
                request_bytes=b"x",
                system_prompt="y",
                max_completion_tokens=10,
                prompt_usd_per_token=True,
                completion_usd_per_token=Decimal("0.000001"),
                context_length=1000,
            )

    def test_rejects_negative_pricing_input(self):
        with self.assertRaises(ValueError):
            derive_text_reservation_usd(
                request_bytes=b"x",
                system_prompt="y",
                max_completion_tokens=10,
                prompt_usd_per_token=Decimal("-0.000001"),
                completion_usd_per_token=Decimal("0.000001"),
                context_length=1000,
            )

    def test_rejects_nan_and_infinite_pricing_input(self):
        for bad in (Decimal("NaN"), Decimal("Infinity")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    derive_text_reservation_usd(
                        request_bytes=b"x",
                        system_prompt="y",
                        max_completion_tokens=10,
                        prompt_usd_per_token=bad,
                        completion_usd_per_token=Decimal("0.000001"),
                        context_length=1000,
                    )


class DeriveAudioReservationUsdTests(unittest.TestCase):
    def test_matches_manual_computation(self):
        reservation = derive_audio_reservation_usd(
            max_billable_seconds=Decimal("185"),
            usd_per_second=Decimal("0.0001"),
        )
        self.assertEqual(reservation, Decimal("0.0185"))

    def test_accepts_int_inputs(self):
        reservation = derive_audio_reservation_usd(
            max_billable_seconds=185,
            usd_per_second=Decimal("0.0001"),
        )
        self.assertEqual(reservation, Decimal("0.0185"))

    def test_fractional_duration_is_billed_as_whole_seconds(self):
        reservation = derive_audio_reservation_usd(
            max_billable_seconds=Decimal("15.6"),
            usd_per_second=Decimal("0.000075"),
        )
        self.assertEqual(reservation, Decimal("16") * Decimal("0.000075"))

    def test_rejects_non_positive_duration(self):
        with self.assertRaises(ValueError):
            derive_audio_reservation_usd(
                max_billable_seconds=Decimal("0"),
                usd_per_second=Decimal("0.0001"),
            )

    def test_rejects_float_duration(self):
        with self.assertRaises(ValueError):
            derive_audio_reservation_usd(
                max_billable_seconds=185.0,
                usd_per_second=Decimal("0.0001"),
            )

    def test_rejects_boolean_duration(self):
        with self.assertRaises(ValueError):
            derive_audio_reservation_usd(
                max_billable_seconds=True,
                usd_per_second=Decimal("0.0001"),
            )

    def test_rejects_negative_price(self):
        with self.assertRaises(ValueError):
            derive_audio_reservation_usd(
                max_billable_seconds=Decimal("10"),
                usd_per_second=Decimal("-0.0001"),
            )

    def test_rejects_nan_and_infinite_price(self):
        for bad in (Decimal("NaN"), Decimal("Infinity")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    derive_audio_reservation_usd(
                        max_billable_seconds=Decimal("10"),
                        usd_per_second=bad,
                    )


if __name__ == "__main__":
    unittest.main()


class ResolveStageReservationUsdTests(unittest.TestCase):
    def _resolved_preset(self, *, max_tokens=1000, prompt="0.000001", completion="0.000003", context_length=1000000):
        preset = _verified_preset(
            config={"model": "prov/model", "max_tokens": max_tokens},
            system_prompt="System prompt",
        )
        catalog = [_catalog_entry("prov/model", prompt=prompt, completion=completion, context_length=context_length)]
        return resolve_preset_model_and_pricing(
            preset,
            api_key="test-key",
            transport=lambda url, api_key: catalog,
            now=lambda: "2026-09-16T00:00:00Z",
        )

    def test_matches_derive_text_reservation_usd_using_the_resolved_presets_own_max_tokens(self):
        resolved = self._resolved_preset(max_tokens=500)
        request_bytes = b"x" * 100

        reservation = resolve_stage_reservation_usd(
            resolved_preset=resolved, worst_case_request_bytes=request_bytes
        )

        expected = derive_text_reservation_usd(
            request_bytes=request_bytes,
            system_prompt=resolved.system_prompt,
            max_completion_tokens=500,
            prompt_usd_per_token=resolved.pricing_bound.prompt_usd_per_token,
            completion_usd_per_token=resolved.pricing_bound.completion_usd_per_token,
            context_length=resolved.pricing_bound.context_length,
        )
        self.assertEqual(reservation, expected)
        self.assertGreater(reservation, Decimal("0"))

    def test_rejects_a_resolved_preset_with_no_max_tokens_configured(self):
        preset = _verified_preset(config={"model": "prov/model"})
        catalog = [_catalog_entry("prov/model")]
        resolved = resolve_preset_model_and_pricing(
            preset,
            api_key="test-key",
            transport=lambda url, api_key: catalog,
            now=lambda: "2026-09-16T00:00:00Z",
        )
        with self.assertRaises(PricingResolutionError):
            resolve_stage_reservation_usd(
                resolved_preset=resolved, worst_case_request_bytes=b"x"
            )

    def test_rejects_a_non_positive_max_tokens(self):
        resolved = self._resolved_preset(max_tokens=0)
        with self.assertRaises(PricingResolutionError):
            resolve_stage_reservation_usd(
                resolved_preset=resolved, worst_case_request_bytes=b"x"
            )

    def test_rejects_a_non_resolved_preset_object(self):
        with self.assertRaises(ValueError):
            resolve_stage_reservation_usd(
                resolved_preset={"not": "a ResolvedPreset"},
                worst_case_request_bytes=b"x",
            )
