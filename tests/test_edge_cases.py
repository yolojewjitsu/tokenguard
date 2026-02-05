"""Edge case tests for tokenguard."""

import pytest

from tokenguard import (
    TokenTracker,
    TokenUsage,
    calculate_cost,
    get_model_cost,
    set_model_cost,
    token_budget,
    tokenguard,
)


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
        usage = tracker.track(
            input_tokens=1_000_000_000, output_tokens=1_000_000_000, model="gpt-4"
        )

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
            1000,
            1000,
            "gpt-4",
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
            1000,
            1000,
            "unknown-model-xyz",
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
        daily_file.write_text(
            json.dumps(
                {
                    "date": "2026-02-04",
                    "total_cost": "not a number",  # Invalid type
                }
            )
        )

        tracker = TokenTracker(budget=10.00, period="daily")
        # Should fall back to 0.0 when total_cost is not a number
        assert tracker.total_cost == 0.0

    def test_token_usage_explicit_timestamp(self):
        """Test TokenUsage with explicit timestamp value."""
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

    def test_calculate_cost_negative_input_tokens_raises(self):
        """Test that negative input_tokens raises ValueError in calculate_cost."""
        with pytest.raises(ValueError, match="input_tokens must be non-negative"):
            calculate_cost(-100, 100, "gpt-4")

    def test_calculate_cost_negative_output_tokens_raises(self):
        """Test that negative output_tokens raises ValueError in calculate_cost."""
        with pytest.raises(ValueError, match="output_tokens must be non-negative"):
            calculate_cost(100, -100, "gpt-4")

    def test_persisted_negative_total_cost_ignored(self, tmp_path, monkeypatch):
        """Test that negative total_cost in persistence file is ignored."""
        import json

        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)
        monkeypatch.setattr("tokenguard.core._today", lambda: "2026-02-04")

        # Pre-create a daily.json with negative total_cost
        daily_file = tmp_path / "daily.json"
        daily_file.write_text(
            json.dumps(
                {
                    "date": "2026-02-04",
                    "total_cost": -5.0,  # Invalid negative value
                }
            )
        )

        tracker = TokenTracker(budget=10.00, period="daily")
        # Should fall back to 0.0 when total_cost is negative
        assert tracker.total_cost == 0.0

    def test_context_manager_raise_on_exceed_false(self):
        """Test context manager with raise_on_exceed=False allows continued tracking."""
        with token_budget(budget=0.01, raise_on_exceed=False) as guard:
            # First track exceeds budget but doesn't raise
            guard.track(input_tokens=1000, output_tokens=500, model="gpt-4")
            assert guard.is_over_budget

            # Can continue tracking
            guard.track(input_tokens=100, output_tokens=50, model="gpt-4")
            assert guard.call_count == 2

    def test_budget_hit_without_callback(self):
        """Test that on_budget_hit=None doesn't crash when budget is hit."""
        tracker = TokenTracker(
            budget=0.05,
            on_budget_hit=None,  # No callback
            raise_on_exceed=False,
        )
        # Should not crash when budget is hit without callback
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert tracker.is_over_budget

    def test_calculate_cost_zero_tokens(self):
        """Test calculate_cost with zero tokens."""
        cost = calculate_cost(0, 0, "gpt-4")
        assert cost == 0.0

        # Zero input, some output
        cost = calculate_cost(0, 1000, "gpt-4")
        assert cost == pytest.approx(0.06)

        # Some input, zero output
        cost = calculate_cost(1000, 0, "gpt-4")
        assert cost == pytest.approx(0.03)

    def test_daily_persistence_shared_between_trackers(self, tmp_path, monkeypatch):
        """Test that multiple daily trackers share the same persisted cost."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # First tracker tracks some cost
        tracker1 = TokenTracker(budget=10.00, period="daily")
        tracker1.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        cost1 = tracker1.total_cost

        # Second tracker should see the persisted cost
        tracker2 = TokenTracker(budget=10.00, period="daily")
        assert tracker2.total_cost == pytest.approx(cost1)

        # Second tracker adds more
        tracker2.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        # Third tracker should see both
        tracker3 = TokenTracker(budget=10.00, period="daily")
        assert tracker3.total_cost == pytest.approx(cost1 * 2)

    def test_persisted_file_missing_period_key(self, tmp_path, monkeypatch):
        """Test that persistence file missing period key is handled."""
        import json

        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)
        monkeypatch.setattr("tokenguard.core._today", lambda: "2026-02-04")

        # Pre-create a daily.json missing the 'date' key
        daily_file = tmp_path / "daily.json"
        daily_file.write_text(
            json.dumps(
                {
                    "total_cost": 5.0  # Missing 'date' key
                }
            )
        )

        tracker = TokenTracker(budget=10.00, period="daily")
        # Should fall back to 0.0 when period key is missing (period rolled over)
        assert tracker.total_cost == 0.0

    def test_decorator_with_list_result(self):
        """Test that decorator does not track when result is a list."""

        @tokenguard(budget=1.00)
        def returns_list():
            return [{"input_tokens": 100, "output_tokens": 50, "model": "gpt-4"}]

        result = returns_list()
        assert isinstance(result, list)
        # Should not track since result is not a dict
        assert returns_list.tracker.call_count == 0

    def test_decorator_with_tuple_result(self):
        """Test that decorator does not track when result is a tuple."""

        @tokenguard(budget=1.00)
        def returns_tuple():
            return (100, 50, "gpt-4")

        result = returns_tuple()
        assert isinstance(result, tuple)
        # Should not track since result is not a dict
        assert returns_tuple.tracker.call_count == 0

    def test_report_when_over_budget(self):
        """Test report() returns correct is_over_budget when over budget."""
        tracker = TokenTracker(budget=0.05, raise_on_exceed=False)
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        report = tracker.report()
        assert report["is_over_budget"] is True
        assert report["remaining"] == 0.0
        assert report["total_cost"] > report["budget"]

    def test_token_usage_equality(self):
        """Test that TokenUsage instances with same values are equal."""
        timestamp = 1234567890.0
        usage1 = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4",
            cost=0.05,
            timestamp=timestamp,
        )
        usage2 = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4",
            cost=0.05,
            timestamp=timestamp,
        )
        # Dataclasses with default eq=True should be equal
        assert usage1 == usage2

    def test_token_usage_inequality(self):
        """Test that TokenUsage instances with different values are not equal."""
        timestamp = 1234567890.0
        usage1 = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4",
            cost=0.05,
            timestamp=timestamp,
        )
        usage2 = TokenUsage(
            input_tokens=200,  # Different
            output_tokens=50,
            model="gpt-4",
            cost=0.05,
            timestamp=timestamp,
        )
        assert usage1 != usage2

    def test_persisted_monthly_file_missing_period_key(self, tmp_path, monkeypatch):
        """Test that monthly persistence file missing period key is handled."""
        import json

        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)
        monkeypatch.setattr("tokenguard.core._this_month", lambda: "2026-02")

        # Pre-create a monthly.json missing the 'month' key
        monthly_file = tmp_path / "monthly.json"
        monthly_file.write_text(
            json.dumps(
                {
                    "total_cost": 5.0  # Missing 'month' key
                }
            )
        )

        tracker = TokenTracker(budget=10.00, period="monthly")
        # Should fall back to 0.0 when period key is missing
        assert tracker.total_cost == 0.0

    def test_track_negative_custom_input_rate_raises(self):
        """Test that track() with negative input_cost_per_1k raises ValueError."""
        tracker = TokenTracker(budget=1.00)
        with pytest.raises(ValueError, match="input_cost_per_1k must be non-negative"):
            tracker.track(
                input_tokens=100,
                output_tokens=50,
                model="gpt-4",
                input_cost_per_1k=-0.01,
            )

    def test_track_negative_custom_output_rate_raises(self):
        """Test that track() with negative output_cost_per_1k raises ValueError."""
        tracker = TokenTracker(budget=1.00)
        with pytest.raises(ValueError, match="output_cost_per_1k must be non-negative"):
            tracker.track(
                input_tokens=100,
                output_tokens=50,
                model="gpt-4",
                output_cost_per_1k=-0.01,
            )

    def test_is_over_budget_false_when_under(self):
        """Test is_over_budget returns False when under budget."""
        tracker = TokenTracker(budget=10.00)
        tracker.track(input_tokens=100, output_tokens=50, model="gpt-4")

        # Cost is ~0.006, well under $10 budget
        assert tracker.is_over_budget is False
        assert tracker.remaining > 0

    def test_persisted_json_list_handled(self, tmp_path, monkeypatch):
        """Test that persistence file containing JSON list is handled gracefully."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)
        monkeypatch.setattr("tokenguard.core._today", lambda: "2026-02-04")

        # Pre-create a daily.json with a list instead of dict
        daily_file = tmp_path / "daily.json"
        daily_file.write_text("[]")  # Valid JSON but wrong type

        tracker = TokenTracker(budget=10.00, period="daily")
        # Should fall back to 0.0 when JSON is not a dict
        assert tracker.total_cost == 0.0

    def test_calculate_cost_unknown_model_partial_input_rate_raises(self):
        """Test that unknown model with only input custom rate still raises."""
        # Need to look up model for the missing output rate
        with pytest.raises(ValueError, match="Unknown model"):
            calculate_cost(1000, 1000, "unknown-model-xyz", input_cost_per_1k=0.01)

    def test_calculate_cost_unknown_model_partial_output_rate_raises(self):
        """Test that unknown model with only output custom rate still raises."""
        # Need to look up model for the missing input rate
        with pytest.raises(ValueError, match="Unknown model"):
            calculate_cost(1000, 1000, "unknown-model-xyz", output_cost_per_1k=0.02)

    def test_tracker_initial_state(self):
        """Test tracker has correct initial state before any tracking."""
        tracker = TokenTracker(budget=10.00)

        assert tracker.total_cost == 0.0
        assert tracker.remaining == 10.00
        assert tracker.call_count == 0
        assert tracker.usage_history == []
        assert tracker.is_over_budget is False

    def test_track_returns_correct_usage(self):
        """Test that track() returns TokenUsage with correct fields."""
        tracker = TokenTracker(budget=10.00)
        usage = tracker.track(input_tokens=100, output_tokens=50, model="gpt-4")

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.model == "gpt-4"
        assert usage.cost > 0
        assert usage.timestamp > 0

    def test_persisted_json_null_handled(self, tmp_path, monkeypatch):
        """Test that persistence file containing JSON null is handled gracefully."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)
        monkeypatch.setattr("tokenguard.core._today", lambda: "2026-02-04")

        # Pre-create a daily.json with null (valid JSON but wrong type)
        daily_file = tmp_path / "daily.json"
        daily_file.write_text("null")

        tracker = TokenTracker(budget=10.00, period="daily")
        # Should fall back to 0.0 when JSON is null
        assert tracker.total_cost == 0.0

    def test_period_property_daily(self):
        """Test period property returns 'daily' for daily tracker."""
        tracker = TokenTracker(budget=1.00, period="daily")
        assert tracker.period == "daily"

    def test_period_property_monthly(self):
        """Test period property returns 'monthly' for monthly tracker."""
        tracker = TokenTracker(budget=1.00, period="monthly")
        assert tracker.period == "monthly"

    def test_report_initial_state(self):
        """Test report() returns correct values before any tracking."""
        tracker = TokenTracker(budget=5.00)
        report = tracker.report()

        assert report["total_cost"] == 0.0
        assert report["session_cost"] == 0.0
        assert report["persisted_cost"] == 0.0
        assert report["calls"] == 0
        assert report["remaining"] == 5.00
        assert report["budget"] == 5.00
        assert report["period"] == "session"
        assert report["is_over_budget"] is False

    def test_token_budget_exceeded_is_exception(self):
        """Test that TokenBudgetExceeded can be caught as Exception."""
        from tokenguard import TokenBudgetExceeded

        exc = TokenBudgetExceeded(budget=1.00, spent=1.50)
        assert isinstance(exc, Exception)

        # Verify it can be caught as generic Exception
        try:
            raise TokenBudgetExceeded(budget=1.00, spent=1.50, model="gpt-4")
        except Exception as e:
            assert isinstance(e, TokenBudgetExceeded)
            assert e.budget == 1.00

    def test_usage_history_multiple_tracks(self):
        """Test usage_history contains correct items after multiple tracks."""
        tracker = TokenTracker(budget=10.00)
        tracker.track(input_tokens=100, output_tokens=50, model="gpt-4")
        tracker.track(input_tokens=200, output_tokens=100, model="claude-3-haiku")
        tracker.track(input_tokens=300, output_tokens=150, model="gpt-4o")

        history = tracker.usage_history
        assert len(history) == 3
        assert history[0].input_tokens == 100
        assert history[0].model == "gpt-4"
        assert history[1].input_tokens == 200
        assert history[1].model == "claude-3-haiku"
        assert history[2].input_tokens == 300
        assert history[2].model == "gpt-4o"

    def test_track_unknown_model_partial_input_rate_raises(self):
        """Test that track() with unknown model and only input custom rate raises."""
        tracker = TokenTracker(budget=1.00)
        # Need to look up model for the missing output rate
        with pytest.raises(ValueError, match="Unknown model"):
            tracker.track(
                input_tokens=100,
                output_tokens=50,
                model="unknown-model-xyz",
                input_cost_per_1k=0.01,
            )

    def test_track_unknown_model_partial_output_rate_raises(self):
        """Test that track() with unknown model and only output custom rate raises."""
        tracker = TokenTracker(budget=1.00)
        # Need to look up model for the missing input rate
        with pytest.raises(ValueError, match="Unknown model"):
            tracker.track(
                input_tokens=100,
                output_tokens=50,
                model="unknown-model-xyz",
                output_cost_per_1k=0.02,
            )

    def test_empty_model_name_raises(self):
        """Test that empty string model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_cost("")

    def test_model_with_leading_whitespace_raises(self):
        """Test that model name with leading whitespace doesn't match."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_cost(" gpt-4")

    def test_model_with_trailing_whitespace_matches(self):
        """Test that model name with trailing whitespace matches via prefix."""
        # "gpt-4 ".startswith("gpt-4") is True, so it matches
        costs = get_model_cost("gpt-4 ")
        assert costs["input"] == 0.03
        assert costs["output"] == 0.06

    def test_calculate_cost_empty_model_with_both_rates(self):
        """Test that empty model name works when both custom rates provided."""
        # Should not raise because we don't need to look up the model
        cost = calculate_cost(
            1000,
            1000,
            "",
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.02,
        )
        assert cost == pytest.approx(0.03)

    def test_track_unknown_model_with_both_rates_succeeds(self):
        """Test that track() with unknown model but both rates works."""
        tracker = TokenTracker(budget=1.00)
        usage = tracker.track(
            input_tokens=1000,
            output_tokens=1000,
            model="unknown-model-xyz",
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.02,
        )
        assert usage.cost == pytest.approx(0.03)
        assert tracker.total_cost == pytest.approx(0.03)

    def test_tokenguard_decorator_invalid_budget_raises(self):
        """Test that tokenguard decorator with negative budget raises."""
        with pytest.raises(ValueError, match="non-negative"):

            @tokenguard(budget=-1.00)
            def func():
                return {"input_tokens": 100, "output_tokens": 50, "model": "gpt-4"}

    def test_tokenguard_decorator_invalid_alert_at_raises(self):
        """Test that tokenguard decorator with invalid alert_at raises."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):

            @tokenguard(budget=1.00, alert_at=1.5)
            def func():
                return {"input_tokens": 100, "output_tokens": 50, "model": "gpt-4"}

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):

            @tokenguard(budget=1.00, alert_at=-0.1)
            def func2():
                return {"input_tokens": 100, "output_tokens": 50, "model": "gpt-4"}

    def test_token_budget_invalid_budget_raises(self):
        """Test that token_budget context manager with negative budget raises."""
        with pytest.raises(ValueError, match="non-negative"):
            with token_budget(budget=-1.00):
                pass

    def test_token_budget_invalid_alert_at_raises(self):
        """Test that token_budget context manager with invalid alert_at raises."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            with token_budget(budget=1.00, alert_at=1.5):
                pass

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            with token_budget(budget=1.00, alert_at=-0.1):
                pass

    def test_decorator_with_string_token_values_raises_type_error(self):
        """Test that decorator raises TypeError when token fields are wrong type."""

        @tokenguard(budget=1.00)
        def returns_string_tokens():
            return {
                "input_tokens": "100",  # String instead of int
                "output_tokens": 50,
                "model": "gpt-4",
            }

        # Should raise TypeError when comparing string to 0
        with pytest.raises(TypeError):
            returns_string_tokens()

    def test_decorator_with_none_token_value_skips_tracking(self):
        """Test that decorator skips tracking when token field is None."""

        @tokenguard(budget=1.00)
        def returns_none_tokens():
            return {
                "input_tokens": None,  # Explicitly None
                "output_tokens": 50,
                "model": "gpt-4",
            }

        returns_none_tokens()
        # Should not track since input_tokens is None
        assert returns_none_tokens.tracker.call_count == 0

    def test_decorator_with_zero_value_tracks(self):
        """Test that decorator tracks when token fields are zero (not None)."""

        @tokenguard(budget=1.00)
        def returns_zero_tokens():
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "model": "gpt-4",
            }

        returns_zero_tokens()
        # Should track since values are 0, not None
        assert returns_zero_tokens.tracker.call_count == 1
        assert returns_zero_tokens.tracker.total_cost == 0.0

    def test_monthly_persistence_shared_between_trackers(self, tmp_path, monkeypatch):
        """Test that multiple monthly trackers share the same persisted cost."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # First tracker tracks some cost
        tracker1 = TokenTracker(budget=10.00, period="monthly")
        tracker1.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        cost1 = tracker1.total_cost

        # Second tracker should see the persisted cost
        tracker2 = TokenTracker(budget=10.00, period="monthly")
        assert tracker2.total_cost == pytest.approx(cost1)

        # Second tracker adds more
        tracker2.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        # Third tracker should see both
        tracker3 = TokenTracker(budget=10.00, period="monthly")
        assert tracker3.total_cost == pytest.approx(cost1 * 2)

    def test_reset_preserves_persisted_cost_monthly(self, tmp_path, monkeypatch):
        """Test reset() for monthly tracker preserves persisted cost."""
        import json
        import time

        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # Pre-create a monthly.json with existing cost
        monthly_file = tmp_path / "monthly.json"
        monthly_file.write_text(
            json.dumps({"month": time.strftime("%Y-%m"), "total_cost": 0.50})
        )

        tracker = TokenTracker(budget=10.00, period="monthly")
        assert tracker.total_cost == 0.50

        # Track some more
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert tracker.total_cost == pytest.approx(0.56)

        # Reset - should clear session but keep persisted
        tracker.reset()
        assert tracker.total_cost == 0.50  # Back to persisted only

    def test_alert_callback_exception_propagates(self):
        """Test that exception in on_alert callback propagates to caller."""

        def failing_alert(tracker, usage):
            raise RuntimeError("Alert callback failed")

        tracker = TokenTracker(
            budget=0.10,
            alert_at=0.5,
            on_alert=failing_alert,
            raise_on_exceed=False,
        )

        # Exception should propagate
        with pytest.raises(RuntimeError, match="Alert callback failed"):
            tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        # But the usage was still recorded before the callback was called
        assert tracker.call_count == 1

    def test_budget_hit_callback_exception_propagates(self):
        """Test that exception in on_budget_hit callback propagates to caller."""

        def failing_budget_hit(tracker, usage):
            raise RuntimeError("Budget hit callback failed")

        tracker = TokenTracker(
            budget=0.05,
            on_budget_hit=failing_budget_hit,
            raise_on_exceed=False,
        )

        # Exception should propagate
        with pytest.raises(RuntimeError, match="Budget hit callback failed"):
            tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        # But the usage was still recorded before the callback was called
        assert tracker.call_count == 1
        assert tracker.is_over_budget

    def test_persisted_empty_file_handled(self, tmp_path, monkeypatch):
        """Test that empty persistence file is handled gracefully."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # Pre-create an empty daily.json file
        daily_file = tmp_path / "daily.json"
        daily_file.write_text("")

        tracker = TokenTracker(budget=10.00, period="daily")
        # Should fall back to 0.0 when file is empty (invalid JSON)
        assert tracker.total_cost == 0.0

    def test_persisted_json_number_handled(self, tmp_path, monkeypatch):
        """Test that persistence file containing JSON number is handled gracefully."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # Pre-create a daily.json with a number instead of dict
        daily_file = tmp_path / "daily.json"
        daily_file.write_text("42")  # Valid JSON but wrong type

        tracker = TokenTracker(budget=10.00, period="daily")
        # Should fall back to 0.0 when JSON is a number (not a dict)
        assert tracker.total_cost == 0.0

    def test_persisted_json_string_handled(self, tmp_path, monkeypatch):
        """Test that persistence file containing JSON string is handled gracefully."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # Pre-create a monthly.json with a string instead of dict
        monthly_file = tmp_path / "monthly.json"
        monthly_file.write_text('"hello"')  # Valid JSON but wrong type

        tracker = TokenTracker(budget=10.00, period="monthly")
        # Should fall back to 0.0 when JSON is a string (not a dict)
        assert tracker.total_cost == 0.0

    def test_persisted_json_boolean_handled(self, tmp_path, monkeypatch):
        """Test that persistence file containing JSON boolean is handled gracefully."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # Pre-create a daily.json with a boolean instead of dict
        daily_file = tmp_path / "daily.json"
        daily_file.write_text("true")  # Valid JSON but wrong type

        tracker = TokenTracker(budget=10.00, period="daily")
        # Should fall back to 0.0 when JSON is a boolean (not a dict)
        assert tracker.total_cost == 0.0
