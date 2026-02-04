"""Edge case tests for tokenguard."""

import pytest

from tokenguard import TokenTracker, tokenguard, calculate_cost, get_model_cost, set_model_cost


class TestEdgeCases:
    def test_zero_tokens(self):
        """Test tracking zero tokens."""
        tracker = TokenTracker(budget=1.00)
        usage = tracker.track(input_tokens=0, output_tokens=0, model="gpt-4")

        assert usage.cost == 0.0
        assert tracker.total_cost == 0.0

    def test_very_large_tokens(self):
        """Test tracking very large token counts."""
        tracker = TokenTracker(budget=1000000.00, raise_on_exceed=False)
        # 1 billion tokens each
        usage = tracker.track(input_tokens=1_000_000_000, output_tokens=1_000_000_000, model="gpt-4")

        # Should calculate without overflow
        assert usage.cost > 0

    def test_budget_zero(self):
        """Test with zero budget - should exceed immediately."""
        tracker = TokenTracker(budget=0.0, raise_on_exceed=False)
        tracker.track(input_tokens=1, output_tokens=1, model="gpt-4")

        assert tracker.is_over_budget

    def test_budget_negative_remaining(self):
        """Test remaining is never negative."""
        tracker = TokenTracker(budget=0.01, raise_on_exceed=False)
        tracker.track(input_tokens=10000, output_tokens=10000, model="gpt-4")

        assert tracker.remaining == 0.0

    def test_alert_at_zero(self):
        """Test alert_at=0 triggers immediately."""
        alerts = []
        tracker = TokenTracker(
            budget=1.00,
            alert_at=0.0,
            on_alert=lambda t, u: alerts.append(u),
            raise_on_exceed=False,
        )
        tracker.track(input_tokens=1, output_tokens=1, model="gpt-4")

        assert len(alerts) == 1

    def test_alert_at_one(self):
        """Test alert_at=1.0 only triggers at budget limit."""
        alerts = []
        tracker = TokenTracker(
            budget=0.10,
            alert_at=1.0,
            on_alert=lambda t, u: alerts.append(u),
            raise_on_exceed=False,
        )

        # This should not trigger alert (cost ~0.06, less than budget)
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert len(alerts) == 0

        # This should trigger alert (total now ~0.12, over budget)
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert len(alerts) == 1

    def test_all_anthropic_models(self):
        """Test that all Anthropic models have pricing."""
        models = [
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-3.5-sonnet",
            "claude-3.5-haiku",
            "claude-sonnet-4",
            "claude-opus-4",
        ]
        for model in models:
            costs = get_model_cost(model)
            assert "input" in costs
            assert "output" in costs

    def test_all_openai_models(self):
        """Test that all OpenAI models have pricing."""
        models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
            "o1-preview",
            "o3-mini",
        ]
        for model in models:
            costs = get_model_cost(model)
            assert "input" in costs
            assert "output" in costs

    def test_all_google_models(self):
        """Test that all Google models have pricing."""
        models = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
        ]
        for model in models:
            costs = get_model_cost(model)
            assert "input" in costs
            assert "output" in costs

    def test_decorator_missing_fields(self):
        """Test decorator handles dict missing required fields."""
        @tokenguard(budget=1.00)
        def partial_result():
            return {"input_tokens": 100}  # Missing output_tokens and model

        result = partial_result()
        assert result == {"input_tokens": 100}
        # Should not track since fields are missing
        assert partial_result.tracker.call_count == 0

    def test_decorator_none_result(self):
        """Test decorator handles None result."""
        @tokenguard(budget=1.00)
        def returns_none():
            return None

        result = returns_none()
        assert result is None
        assert returns_none.tracker.call_count == 0

    def test_tracker_thread_safety(self):
        """Test that tracker is thread-safe."""
        import threading

        tracker = TokenTracker(budget=10000.00, raise_on_exceed=False)
        errors = []

        def track_many():
            try:
                for _ in range(100):
                    tracker.track(input_tokens=100, output_tokens=50, model="gpt-4")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=track_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.call_count == 1000

    def test_custom_cost_both_rates(self):
        """Test calculate_cost with both custom rates."""
        cost = calculate_cost(
            1000, 1000, "gpt-4",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )
        assert cost == pytest.approx(0.003)

    def test_get_model_cost_returns_copy(self):
        """Test that get_model_cost returns a copy, not the original."""
        costs1 = get_model_cost("gpt-4")
        costs1["input"] = 999.0  # Modify the returned dict
        costs2 = get_model_cost("gpt-4")
        assert costs2["input"] != 999.0  # Original should be unchanged

    def test_prefix_match_longest_wins(self):
        """Test that prefix matching picks the longest match."""
        # "o1-mini-xxx" should match "o1-mini", not "o1"
        costs = get_model_cost("o1-mini-2025-01-31")
        # o1-mini has different pricing than o1
        o1_mini_costs = get_model_cost("o1-mini")
        assert costs["input"] == o1_mini_costs["input"]
        assert costs["output"] == o1_mini_costs["output"]

    def test_negative_budget_raises(self):
        """Test that negative budget raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            TokenTracker(budget=-1.00)

    def test_alert_at_out_of_range_raises(self):
        """Test that alert_at outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            TokenTracker(budget=1.00, alert_at=1.5)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            TokenTracker(budget=1.00, alert_at=-0.1)

    def test_negative_input_tokens_raises(self):
        """Test that negative input_tokens raises ValueError."""
        tracker = TokenTracker(budget=1.00)
        with pytest.raises(ValueError, match="input_tokens must be non-negative"):
            tracker.track(input_tokens=-100, output_tokens=50, model="gpt-4")

    def test_negative_output_tokens_raises(self):
        """Test that negative output_tokens raises ValueError."""
        tracker = TokenTracker(budget=1.00)
        with pytest.raises(ValueError, match="output_tokens must be non-negative"):
            tracker.track(input_tokens=100, output_tokens=-50, model="gpt-4")

    def test_set_model_cost_negative_input_raises(self):
        """Test that negative input cost raises ValueError."""
        with pytest.raises(ValueError, match="Input cost must be non-negative"):
            set_model_cost("test-model", input=-0.01, output=0.02)

    def test_set_model_cost_negative_output_raises(self):
        """Test that negative output cost raises ValueError."""
        with pytest.raises(ValueError, match="Output cost must be non-negative"):
            set_model_cost("test-model", input=0.01, output=-0.02)

    def test_budget_hit_fires_once(self):
        """Test that on_budget_hit callback only fires once."""
        hits = []

        def on_hit(tracker, usage):
            hits.append(usage)

        tracker = TokenTracker(
            budget=0.05,
            on_budget_hit=on_hit,
            raise_on_exceed=False,
        )

        # First call exceeds budget - should fire callback
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert len(hits) == 1

        # Second call also over budget - should NOT fire callback again
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert len(hits) == 1

    def test_reset_allows_budget_hit_to_fire_again(self):
        """Test that reset() allows budget_hit to fire again."""
        hits = []

        def on_hit(tracker, usage):
            hits.append(usage)

        tracker = TokenTracker(
            budget=0.05,
            on_budget_hit=on_hit,
            raise_on_exceed=False,
        )

        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert len(hits) == 1

        tracker.reset()

        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert len(hits) == 2

    def test_calculate_cost_negative_input_rate_raises(self):
        """Test that negative input_cost_per_1k raises ValueError."""
        with pytest.raises(ValueError, match="input_cost_per_1k must be non-negative"):
            calculate_cost(1000, 1000, "gpt-4", input_cost_per_1k=-0.01)

    def test_calculate_cost_negative_output_rate_raises(self):
        """Test that negative output_cost_per_1k raises ValueError."""
        with pytest.raises(ValueError, match="output_cost_per_1k must be non-negative"):
            calculate_cost(1000, 1000, "gpt-4", output_cost_per_1k=-0.01)

    def test_calculate_cost_unknown_model_raises(self):
        """Test that calculate_cost raises for unknown model without custom rates."""
        with pytest.raises(ValueError, match="Unknown model"):
            calculate_cost(1000, 1000, "unknown-model-xyz")

    def test_calculate_cost_unknown_model_with_both_rates(self):
        """Test that unknown model works when both custom rates provided."""
        # Should not raise because we don't need to look up the model
        cost = calculate_cost(
            1000, 1000, "unknown-model-xyz",
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.02,
        )
        assert cost == pytest.approx(0.03)

    def test_decorator_passes_arguments(self):
        """Test that decorator correctly passes function arguments."""
        @tokenguard(budget=1.00)
        def add_tokens(a: int, b: int, model: str = "gpt-4") -> dict:
            return {
                "result": a + b,
                "input_tokens": a,
                "output_tokens": b,
                "model": model,
            }

        result = add_tokens(100, 50, model="gpt-4")
        assert result["result"] == 150
        assert add_tokens.tracker.call_count == 1

    def test_decorator_function_raises_no_tracking(self):
        """Test that decorator doesn't track when function raises."""
        @tokenguard(budget=1.00)
        def failing_func():
            raise RuntimeError("Intentional error")

        with pytest.raises(RuntimeError, match="Intentional error"):
            failing_func()

        # Should not track since function raised before returning
        assert failing_func.tracker.call_count == 0

    def test_set_model_cost_zero_rates_allowed(self):
        """Test that zero cost rates are allowed (for free models)."""
        set_model_cost("free-model-test", input=0.0, output=0.0)
        costs = get_model_cost("free-model-test")
        assert costs["input"] == 0.0
        assert costs["output"] == 0.0

        # Calculate cost with free model
        cost = calculate_cost(1000, 1000, "free-model-test")
        assert cost == 0.0

    def test_reset_all_resets_callback_flags(self):
        """Test that reset_all() resets both alert and budget_hit flags."""
        alerts = []
        hits = []

        tracker = TokenTracker(
            budget=0.05,
            alert_at=0.5,
            on_alert=lambda t, u: alerts.append(u),
            on_budget_hit=lambda t, u: hits.append(u),
            raise_on_exceed=False,
        )

        # First call triggers both
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert len(alerts) == 1
        assert len(hits) == 1

        # reset_all should reset both flags
        tracker.reset_all()

        # After reset_all, both should fire again
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert len(alerts) == 2
        assert len(hits) == 2

    def test_decorator_raise_on_exceed_false(self):
        """Test decorator with raise_on_exceed=False allows continued tracking."""
        @tokenguard(budget=0.01, raise_on_exceed=False)
        def cheap_call():
            return {"input_tokens": 1000, "output_tokens": 500, "model": "gpt-4"}

        # First call exceeds budget but doesn't raise
        result = cheap_call()
        assert result["input_tokens"] == 1000
        assert cheap_call.tracker.is_over_budget

        # Can continue calling
        cheap_call()
        assert cheap_call.tracker.call_count == 2

    def test_persisted_invalid_total_cost_type(self, tmp_path, monkeypatch):
        """Test that invalid total_cost type in persistence file is handled."""
        import json

        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)
        monkeypatch.setattr("tokenguard.core._today", lambda: "2026-02-04")

        # Pre-create a daily.json with invalid total_cost type
        daily_file = tmp_path / "daily.json"
        daily_file.write_text(json.dumps({
            "date": "2026-02-04",
            "total_cost": "not a number"  # Invalid type
        }))

        tracker = TokenTracker(budget=10.00, period="daily")
        # Should fall back to 0.0 when total_cost is not a number
        assert tracker.total_cost == 0.0

    def test_token_usage_explicit_timestamp(self):
        """Test TokenUsage with explicit timestamp value."""
        from tokenguard import TokenUsage

        explicit_time = 1234567890.123
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4",
            cost=0.05,
            timestamp=explicit_time,
        )
        assert usage.timestamp == explicit_time

    def test_is_over_budget_at_exact_limit(self):
        """Test is_over_budget when cost equals budget exactly."""
        # gpt-4: 0.03/1k input, 0.06/1k output
        # 1000 input + 0 output = $0.03
        tracker = TokenTracker(budget=0.03, raise_on_exceed=False)
        tracker.track(input_tokens=1000, output_tokens=0, model="gpt-4")

        # At exact limit, is_over_budget should be True (budget met)
        assert tracker.total_cost == pytest.approx(0.03)
        assert tracker.is_over_budget is True
        assert tracker.remaining == 0.0
