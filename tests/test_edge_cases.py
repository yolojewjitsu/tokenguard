"""Edge case tests for tokenguard."""

import pytest

from tokenguard import TokenTracker, tokenguard, calculate_cost, get_model_cost


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
