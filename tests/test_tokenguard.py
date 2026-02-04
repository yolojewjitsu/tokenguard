"""Tests for tokenguard."""

import pytest

from tokenguard import (
    TokenBudgetExceeded,
    TokenTracker,
    TokenUsage,
    calculate_cost,
    get_model_cost,
    set_model_cost,
    token_budget,
    tokenguard,
)


class TestModelCosts:
    def test_get_known_model(self):
        """Test getting cost for a known model."""
        costs = get_model_cost("gpt-4")
        assert costs["input"] == 0.03
        assert costs["output"] == 0.06

    def test_get_model_prefix_match(self):
        """Test prefix matching for model variants."""
        # gpt-4-0613 should match gpt-4
        costs = get_model_cost("gpt-4-0613")
        assert costs["input"] == 0.03

    def test_get_unknown_model_raises(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_cost("unknown-model-xyz")

    def test_set_model_cost(self):
        """Test setting custom model cost."""
        set_model_cost("test-custom-model-12345", input=0.01, output=0.02)
        costs = get_model_cost("test-custom-model-12345")
        assert costs["input"] == 0.01
        assert costs["output"] == 0.02
        # Note: This adds to global state but uses a unique name unlikely to conflict

    def test_override_existing_model_cost(self):
        """Test overriding existing model cost."""
        original = get_model_cost("gpt-3.5-turbo")
        set_model_cost("gpt-3.5-turbo", input=0.001, output=0.002)
        updated = get_model_cost("gpt-3.5-turbo")
        assert updated["input"] == 0.001
        # Restore original
        set_model_cost(
            "gpt-3.5-turbo", input=original["input"], output=original["output"]
        )


class TestCalculateCost:
    def test_basic_calculation(self):
        """Test basic cost calculation."""
        cost = calculate_cost(1000, 500, "gpt-4")
        # 1000 * 0.03 / 1000 + 500 * 0.06 / 1000 = 0.03 + 0.03 = 0.06
        assert cost == pytest.approx(0.06)

    def test_with_custom_rates(self):
        """Test calculation with custom rates."""
        cost = calculate_cost(
            1000,
            1000,
            "gpt-4",
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.02,
        )
        # 1000 * 0.01 / 1000 + 1000 * 0.02 / 1000 = 0.01 + 0.02 = 0.03
        assert cost == pytest.approx(0.03)

    def test_partial_custom_input_rate(self):
        """Test calculation with partial custom input rate."""
        cost = calculate_cost(
            1000,
            1000,
            "gpt-4",
            input_cost_per_1k=0.01,  # Custom input
            # output uses model default (0.06)
        )
        # 1000 * 0.01 / 1000 + 1000 * 0.06 / 1000 = 0.01 + 0.06 = 0.07
        assert cost == pytest.approx(0.07)

    def test_partial_custom_output_rate(self):
        """Test calculation with partial custom output rate."""
        cost = calculate_cost(
            1000,
            1000,
            "gpt-4",
            # input uses model default (0.03)
            output_cost_per_1k=0.01,  # Custom output
        )
        # 1000 * 0.03 / 1000 + 1000 * 0.01 / 1000 = 0.03 + 0.01 = 0.04
        assert cost == pytest.approx(0.04)


class TestTokenTracker:
    def test_basic_tracking(self):
        """Test basic token tracking."""
        tracker = TokenTracker(budget=1.00)
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        assert tracker.total_cost == pytest.approx(0.06)
        assert tracker.remaining == pytest.approx(0.94)
        assert tracker.call_count == 1

    def test_multiple_tracks(self):
        """Test tracking multiple calls."""
        tracker = TokenTracker(budget=1.00)
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        assert tracker.total_cost == pytest.approx(0.12)
        assert tracker.call_count == 2

    def test_budget_exceeded_raises(self):
        """Test that exceeding budget raises exception."""
        tracker = TokenTracker(budget=0.05, raise_on_exceed=True)

        with pytest.raises(TokenBudgetExceeded) as exc:
            tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        assert exc.value.budget == 0.05
        assert exc.value.spent == pytest.approx(0.06)

    def test_budget_exceeded_no_raise(self):
        """Test that raise_on_exceed=False suppresses exception."""
        tracker = TokenTracker(budget=0.05, raise_on_exceed=False)
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        assert tracker.is_over_budget
        assert tracker.total_cost == pytest.approx(0.06)

    def test_alert_callback(self):
        """Test alert callback at threshold."""
        alerts = []

        def on_alert(tracker, usage):
            alerts.append((tracker.total_cost, usage.cost))

        tracker = TokenTracker(
            budget=0.10,
            alert_at=0.5,
            on_alert=on_alert,
            raise_on_exceed=False,
        )

        # First call: 0.06, which is > 50% of 0.10
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        assert len(alerts) == 1
        assert alerts[0][0] == pytest.approx(0.06)

    def test_alert_fires_once(self):
        """Test that alert only fires once."""
        alerts = []

        def on_alert(tracker, usage):
            alerts.append(usage)

        tracker = TokenTracker(
            budget=0.50,
            alert_at=0.1,
            on_alert=on_alert,
            raise_on_exceed=False,
        )

        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        assert len(alerts) == 1

    def test_reset_allows_alert_to_fire_again(self):
        """Test that reset() allows alert to fire again."""
        alerts = []

        def on_alert(tracker, usage):
            alerts.append(usage)

        tracker = TokenTracker(
            budget=0.10,
            alert_at=0.5,
            on_alert=on_alert,
            raise_on_exceed=False,
        )

        # First call triggers alert (0.06 > 50% of 0.10)
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert len(alerts) == 1

        tracker.reset()

        # After reset, alert should fire again
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert len(alerts) == 2

    def test_budget_hit_callback(self):
        """Test on_budget_hit callback."""
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

    def test_reset(self):
        """Test reset clears session usage."""
        tracker = TokenTracker(budget=1.00)
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        tracker.reset()

        assert tracker.total_cost == 0.0
        assert tracker.call_count == 0

    def test_report(self):
        """Test report returns correct summary."""
        tracker = TokenTracker(budget=1.00)
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        report = tracker.report()

        assert report["total_cost"] == pytest.approx(0.06)
        assert report["session_cost"] == pytest.approx(0.06)
        assert report["persisted_cost"] == 0.0
        assert report["calls"] == 1
        assert report["budget"] == 1.00
        assert report["period"] == "session"
        assert report["remaining"] == pytest.approx(0.94)
        assert report["is_over_budget"] is False

    def test_invalid_period_raises(self):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match="Invalid period"):
            TokenTracker(budget=1.00, period="invalid")

    def test_usage_history(self):
        """Test usage_history returns copy of usage list."""
        tracker = TokenTracker(budget=1.00)
        tracker.track(input_tokens=100, output_tokens=50, model="gpt-4")

        history = tracker.usage_history
        assert len(history) == 1
        assert history[0].input_tokens == 100

    def test_usage_history_is_copy(self):
        """Test that mutating usage_history doesn't affect internal state."""
        tracker = TokenTracker(budget=1.00)
        tracker.track(input_tokens=100, output_tokens=50, model="gpt-4")

        history = tracker.usage_history
        history.clear()  # Mutate the returned list

        # Internal state should be unaffected
        assert tracker.call_count == 1
        assert len(tracker.usage_history) == 1

    def test_budget_property(self):
        """Test budget property returns correct value."""
        tracker = TokenTracker(budget=5.50)
        assert tracker.budget == 5.50

    def test_period_property(self):
        """Test period property returns correct value."""
        tracker = TokenTracker(budget=1.00, period="session")
        assert tracker.period == "session"

    def test_track_with_custom_costs(self):
        """Test track() with custom cost overrides."""
        tracker = TokenTracker(budget=1.00)
        usage = tracker.track(
            input_tokens=1000,
            output_tokens=1000,
            model="gpt-4",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )
        # 1000 * 0.001 / 1000 + 1000 * 0.002 / 1000 = 0.001 + 0.002 = 0.003
        assert usage.cost == pytest.approx(0.003)
        assert tracker.total_cost == pytest.approx(0.003)

    def test_budget_exceeded_has_model(self):
        """Test that TokenBudgetExceeded has correct model attribute."""
        tracker = TokenTracker(budget=0.01, raise_on_exceed=True)

        with pytest.raises(TokenBudgetExceeded) as exc:
            tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        assert exc.value.model == "gpt-4"

    def test_alert_at_without_callback(self):
        """Test that alert_at works without on_alert callback."""
        # Should not crash when alert_at is set but on_alert is None
        tracker = TokenTracker(
            budget=0.10,
            alert_at=0.5,
            on_alert=None,  # No callback
            raise_on_exceed=False,
        )
        # This should trigger alert threshold but not crash
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert tracker.total_cost == pytest.approx(0.06)

    def test_alert_and_budget_hit_same_call(self):
        """Test that both alert and budget_hit can fire on same call."""
        alerts = []
        hits = []

        tracker = TokenTracker(
            budget=0.05,
            alert_at=0.5,
            on_alert=lambda t, u: alerts.append(u),
            on_budget_hit=lambda t, u: hits.append(u),
            raise_on_exceed=False,
        )

        # Cost ~0.06 exceeds both alert (0.025) and budget (0.05)
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        assert len(alerts) == 1
        assert len(hits) == 1


class TestDailyMonthlyPersistence:
    def test_daily_tracker_creates_file(self, tmp_path, monkeypatch):
        """Test daily tracker creates persistence file."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        tracker = TokenTracker(budget=10.00, period="daily")
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        daily_file = tmp_path / "daily.json"
        assert daily_file.exists()

    def test_monthly_tracker_creates_file(self, tmp_path, monkeypatch):
        """Test monthly tracker creates persistence file."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        tracker = TokenTracker(budget=10.00, period="monthly")
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        monthly_file = tmp_path / "monthly.json"
        assert monthly_file.exists()

    def test_daily_tracker_loads_persisted(self, tmp_path, monkeypatch):
        """Test daily tracker loads persisted cost from file."""
        import json
        import time

        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # Pre-create a daily.json with existing cost
        daily_file = tmp_path / "daily.json"
        daily_file.write_text(
            json.dumps({"date": time.strftime("%Y-%m-%d"), "total_cost": 0.50})
        )

        tracker = TokenTracker(budget=10.00, period="daily")
        assert tracker.total_cost == 0.50

    def test_monthly_tracker_loads_persisted(self, tmp_path, monkeypatch):
        """Test monthly tracker loads persisted cost from file."""
        import json
        import time

        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # Pre-create a monthly.json with existing cost
        monthly_file = tmp_path / "monthly.json"
        monthly_file.write_text(
            json.dumps({"month": time.strftime("%Y-%m"), "total_cost": 1.25})
        )

        tracker = TokenTracker(budget=10.00, period="monthly")
        assert tracker.total_cost == 1.25

    def test_daily_rollover_resets(self, tmp_path, monkeypatch):
        """Test daily tracker resets cost when date changes."""
        import json

        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # Pre-create a daily.json with old date
        daily_file = tmp_path / "daily.json"
        daily_file.write_text(
            json.dumps(
                {
                    "date": "2020-01-01",  # Old date
                    "total_cost": 5.00,
                }
            )
        )

        tracker = TokenTracker(budget=10.00, period="daily")
        assert tracker.total_cost == 0.0  # Should reset

    def test_monthly_rollover_resets(self, tmp_path, monkeypatch):
        """Test monthly tracker resets cost when month changes."""
        import json

        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # Pre-create a monthly.json with old month
        monthly_file = tmp_path / "monthly.json"
        monthly_file.write_text(
            json.dumps(
                {
                    "month": "2020-01",  # Old month
                    "total_cost": 5.00,
                }
            )
        )

        tracker = TokenTracker(budget=10.00, period="monthly")
        assert tracker.total_cost == 0.0  # Should reset

    def test_invalid_json_resets(self, tmp_path, monkeypatch):
        """Test tracker handles invalid JSON gracefully."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # Pre-create invalid JSON file
        daily_file = tmp_path / "daily.json"
        daily_file.write_text("not valid json {{{")

        tracker = TokenTracker(budget=10.00, period="daily")
        assert tracker.total_cost == 0.0

    def test_file_read_error_resets(self, tmp_path, monkeypatch):
        """Test tracker handles file read errors gracefully."""
        import os

        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # Pre-create file with no read permissions
        daily_file = tmp_path / "daily.json"
        daily_file.write_text('{"date": "2026-01-01", "total_cost": 5.0}')
        os.chmod(daily_file, 0o000)

        try:
            tracker = TokenTracker(budget=10.00, period="daily")
            # Should gracefully fall back to 0.0 on permission error
            assert tracker.total_cost == 0.0
        finally:
            # Restore permissions so tmp_path cleanup can work
            os.chmod(daily_file, 0o644)

    def test_reset_all_daily(self, tmp_path, monkeypatch):
        """Test reset_all clears daily persistence file."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        tracker = TokenTracker(budget=10.00, period="daily")
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        daily_file = tmp_path / "daily.json"
        assert daily_file.exists()

        tracker.reset_all()
        assert not daily_file.exists()
        assert tracker.total_cost == 0.0

    def test_reset_all_monthly(self, tmp_path, monkeypatch):
        """Test reset_all clears monthly persistence file."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        tracker = TokenTracker(budget=10.00, period="monthly")
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        monthly_file = tmp_path / "monthly.json"
        assert monthly_file.exists()

        tracker.reset_all()
        assert not monthly_file.exists()
        assert tracker.total_cost == 0.0

    def test_reset_all_session_no_file(self, tmp_path, monkeypatch):
        """Test reset_all for session tracker doesn't touch files."""
        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        tracker = TokenTracker(budget=10.00, period="session")
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        tracker.reset_all()

        assert tracker.total_cost == 0.0
        # No files should be created or touched for session

    def test_reset_preserves_persisted_cost(self, tmp_path, monkeypatch):
        """Test reset() for daily tracker preserves persisted cost."""
        import json
        import time

        monkeypatch.setattr("tokenguard.core._get_storage_dir", lambda: tmp_path)

        # Pre-create a daily.json with existing cost
        daily_file = tmp_path / "daily.json"
        daily_file.write_text(
            json.dumps({"date": time.strftime("%Y-%m-%d"), "total_cost": 0.50})
        )

        tracker = TokenTracker(budget=10.00, period="daily")
        assert tracker.total_cost == 0.50

        # Track some more
        tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
        assert tracker.total_cost == pytest.approx(0.56)

        # Reset - should clear session but keep persisted
        tracker.reset()
        assert tracker.total_cost == 0.50  # Back to persisted only

    def test_get_storage_dir_creates_directory(self, tmp_path, monkeypatch):
        """Test _get_storage_dir creates the .tokenguard directory."""
        from tokenguard.core import _get_storage_dir
        from pathlib import Path

        # Mock Path.home() to return tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        storage_dir = _get_storage_dir()

        assert storage_dir == tmp_path / ".tokenguard"
        assert storage_dir.exists()
        assert storage_dir.is_dir()


class TestTokenguardDecorator:
    def test_decorator_tracks_dict_result(self):
        """Test decorator extracts tokens from dict result."""

        @tokenguard(budget=1.00)
        def mock_llm_call():
            return {
                "result": "Hello!",
                "input_tokens": 100,
                "output_tokens": 50,
                "model": "gpt-4",
            }

        result = mock_llm_call()

        assert result["result"] == "Hello!"
        assert mock_llm_call.tracker.call_count == 1
        assert mock_llm_call.tracker.total_cost > 0

    def test_decorator_ignores_non_dict(self):
        """Test decorator ignores non-dict results."""

        @tokenguard(budget=1.00)
        def simple_func():
            return "just a string"

        result = simple_func()

        assert result == "just a string"
        assert simple_func.tracker.call_count == 0

    def test_decorator_budget_exceeded(self):
        """Test decorator raises on budget exceeded."""

        @tokenguard(budget=0.001)
        def expensive_call():
            return {
                "result": "done",
                "input_tokens": 1000,
                "output_tokens": 500,
                "model": "gpt-4",
            }

        with pytest.raises(TokenBudgetExceeded):
            expensive_call()

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""

        @tokenguard(budget=1.00)
        def documented_func():
            """This is a docstring."""
            return {"input_tokens": 10, "output_tokens": 5, "model": "gpt-4"}

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."

    def test_decorator_reset(self):
        """Test decorator reset method."""

        @tokenguard(budget=1.00)
        def mock_call():
            return {"input_tokens": 100, "output_tokens": 50, "model": "gpt-4"}

        mock_call()
        assert mock_call.tracker.call_count == 1

        mock_call.reset()
        assert mock_call.tracker.call_count == 0

    def test_decorator_report(self):
        """Test decorator report method."""

        @tokenguard(budget=1.00)
        def mock_call():
            return {"input_tokens": 100, "output_tokens": 50, "model": "gpt-4"}

        mock_call()
        report = mock_call.report()

        assert report["calls"] == 1
        assert report["budget"] == 1.00

    def test_decorator_with_alert_callback(self):
        """Test decorator with on_alert callback."""
        alerts = []

        @tokenguard(
            budget=0.10,
            alert_at=0.5,
            on_alert=lambda t, u: alerts.append(u),
            raise_on_exceed=False,
        )
        def mock_call():
            return {"input_tokens": 1000, "output_tokens": 500, "model": "gpt-4"}

        mock_call()  # Cost ~0.06 > 50% of 0.10
        assert len(alerts) == 1

    def test_decorator_with_budget_hit_callback(self):
        """Test decorator with on_budget_hit callback."""
        hits = []

        @tokenguard(
            budget=0.05,
            on_budget_hit=lambda t, u: hits.append(u),
            raise_on_exceed=False,
        )
        def mock_call():
            return {"input_tokens": 1000, "output_tokens": 500, "model": "gpt-4"}

        mock_call()  # Cost ~0.06 > budget 0.05
        assert len(hits) == 1


class TestTokenBudgetContext:
    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        with token_budget(budget=1.00) as guard:
            guard.track(input_tokens=100, output_tokens=50, model="gpt-4")

            assert guard.total_cost > 0
            assert guard.call_count == 1

    def test_context_manager_report(self):
        """Test context manager report."""
        with token_budget(budget=1.00) as guard:
            guard.track(input_tokens=100, output_tokens=50, model="gpt-4")
            report = guard.report()

            assert report["calls"] == 1
            assert report["budget"] == 1.00

    def test_context_manager_budget_exceeded(self):
        """Test context manager raises on budget exceeded."""
        with pytest.raises(TokenBudgetExceeded):
            with token_budget(budget=0.001) as guard:
                guard.track(input_tokens=1000, output_tokens=500, model="gpt-4")

    def test_context_manager_with_alert_callback(self):
        """Test context manager with on_alert callback."""
        alerts = []

        with token_budget(
            budget=0.10,
            alert_at=0.5,
            on_alert=lambda t, u: alerts.append(u),
            raise_on_exceed=False,
        ) as guard:
            guard.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        assert len(alerts) == 1

    def test_context_manager_with_budget_hit_callback(self):
        """Test context manager with on_budget_hit callback."""
        hits = []

        with token_budget(
            budget=0.05,
            on_budget_hit=lambda t, u: hits.append(u),
            raise_on_exceed=False,
        ) as guard:
            guard.track(input_tokens=1000, output_tokens=500, model="gpt-4")

        assert len(hits) == 1


class TestTokenUsage:
    def test_token_usage_fields(self):
        """Test TokenUsage dataclass fields."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4",
            cost=0.05,
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.model == "gpt-4"
        assert usage.cost == 0.05
        assert usage.timestamp > 0


class TestTokenBudgetExceeded:
    def test_exception_attributes(self):
        """Test exception has correct attributes."""
        exc = TokenBudgetExceeded(budget=1.00, spent=1.50, model="gpt-4")

        assert exc.budget == 1.00
        assert exc.spent == 1.50
        assert exc.model == "gpt-4"

    def test_exception_str(self):
        """Test exception string representation."""
        exc = TokenBudgetExceeded(budget=1.00, spent=1.50)
        assert "exceeded" in str(exc).lower()
        assert "1.50" in str(exc)

    def test_exception_repr(self):
        """Test exception repr."""
        exc = TokenBudgetExceeded(budget=1.00, spent=1.50, model="gpt-4")
        r = repr(exc)
        assert "TokenBudgetExceeded" in r
        assert "budget=1.0" in r
        assert "spent=1.5" in r
        assert "model='gpt-4'" in r

    def test_exception_repr_no_model(self):
        """Test exception repr when model is None."""
        exc = TokenBudgetExceeded(budget=1.00, spent=1.50)
        r = repr(exc)
        assert "TokenBudgetExceeded" in r
        assert "model=None" in r
