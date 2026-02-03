"""Core token tracking and budget enforcement for LLM API calls."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

__all__ = [
    "TokenBudgetExceeded",
    "TokenTracker",
    "TokenUsage",
    "calculate_cost",
    "get_model_cost",
    "set_model_cost",
    "token_budget",
    "tokenguard",
]

T = TypeVar("T")

# Default costs per 1K tokens (USD)
_MODEL_COSTS: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "o1-preview": {"input": 0.015, "output": 0.06},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
    # Anthropic
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3.5-haiku": {"input": 0.0008, "output": 0.004},
    "claude-sonnet-4": {"input": 0.003, "output": 0.015},
    "claude-opus-4": {"input": 0.015, "output": 0.075},
    # Google
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
}

_costs_lock = threading.Lock()


class TokenBudgetExceeded(Exception):
    """Raised when token budget is exceeded.

    Attributes:
        budget: The budget limit that was exceeded.
        spent: The amount spent when budget was exceeded.
        model: The model that caused the budget to be exceeded.

    """

    __slots__ = ("budget", "model", "spent")

    def __init__(
        self,
        budget: float,
        spent: float,
        model: str | None = None,
    ) -> None:
        self.budget = budget
        self.spent = spent
        self.model = model
        super().__init__(
            f"Token budget exceeded: ${spent:.4f} spent, ${budget:.4f} limit"
        )

    def __repr__(self) -> str:
        return f"TokenBudgetExceeded(budget={self.budget}, spent={self.spent})"


@dataclass
class TokenUsage:
    """Record of a single LLM API call's token usage."""

    input_tokens: int
    output_tokens: int
    model: str
    cost: float
    timestamp: float = field(default_factory=time.time)


def get_model_cost(model: str) -> dict[str, float]:
    """Get the cost per 1K tokens for a model.

    Args:
        model: Model name (e.g., "gpt-4", "claude-3-sonnet").

    Returns:
        Dict with "input" and "output" costs per 1K tokens.

    Raises:
        ValueError: If model is not found in the pricing database.

    """
    with _costs_lock:
        # Try exact match first
        if model in _MODEL_COSTS:
            return _MODEL_COSTS[model].copy()

        # Try prefix match (e.g., "gpt-4-0613" matches "gpt-4")
        for known_model in _MODEL_COSTS:
            if model.startswith(known_model):
                return _MODEL_COSTS[known_model].copy()

    raise ValueError(
        f"Unknown model: {model!r}. Use set_model_cost() to add custom pricing."
    )


def set_model_cost(
    model: str,
    *,
    input: float,  # noqa: A002 - shadowing builtin is intentional for API clarity
    output: float,
) -> None:
    """Set or override the cost per 1K tokens for a model.

    Args:
        model: Model name.
        input: Cost per 1K input tokens in USD.
        output: Cost per 1K output tokens in USD.

    """
    with _costs_lock:
        _MODEL_COSTS[model] = {"input": input, "output": output}


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    *,
    input_cost_per_1k: float | None = None,
    output_cost_per_1k: float | None = None,
) -> float:
    """Calculate the cost of an LLM API call.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model name for pricing lookup.
        input_cost_per_1k: Override input cost per 1K tokens.
        output_cost_per_1k: Override output cost per 1K tokens.

    Returns:
        Total cost in USD.

    """
    if input_cost_per_1k is not None and output_cost_per_1k is not None:
        input_rate = input_cost_per_1k
        output_rate = output_cost_per_1k
    else:
        costs = get_model_cost(model)
        input_rate = input_cost_per_1k if input_cost_per_1k is not None else costs["input"]
        output_rate = output_cost_per_1k if output_cost_per_1k is not None else costs["output"]

    return (input_tokens * input_rate / 1000) + (output_tokens * output_rate / 1000)


def _get_storage_dir() -> Path:
    """Get the tokenguard storage directory."""
    storage_dir = Path.home() / ".tokenguard"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def _today() -> str:
    """Get today's date as YYYY-MM-DD."""
    return time.strftime("%Y-%m-%d")


def _this_month() -> str:
    """Get current month as YYYY-MM."""
    return time.strftime("%Y-%m")


class TokenTracker:
    """Track token usage and enforce budgets.

    Example:
        tracker = TokenTracker(budget=5.00, period="session")
        tracker.track(input_tokens=150, output_tokens=200, model="gpt-4")
        print(tracker.total_cost)  # $0.0135

    """

    def __init__(
        self,
        budget: float,
        *,
        period: str = "session",
        alert_at: float | None = None,
        on_alert: Callable[[TokenTracker, TokenUsage], Any] | None = None,
        on_budget_hit: Callable[[TokenTracker, TokenUsage], Any] | None = None,
        raise_on_exceed: bool = True,
    ) -> None:
        """Initialize the tracker.

        Args:
            budget: Maximum budget in USD.
            period: "session", "daily", or "monthly".
            alert_at: Fraction of budget to trigger alert (e.g., 0.8 for 80%).
            on_alert: Callback when alert threshold is reached.
            on_budget_hit: Callback when budget is exceeded.
            raise_on_exceed: Whether to raise TokenBudgetExceeded.

        """
        if period not in ("session", "daily", "monthly"):
            raise ValueError(f"Invalid period: {period!r}. Use 'session', 'daily', or 'monthly'.")

        self._budget = budget
        self._period = period
        self._alert_at = alert_at
        self._on_alert = on_alert
        self._on_budget_hit = on_budget_hit
        self._raise_on_exceed = raise_on_exceed

        self._usage: list[TokenUsage] = []
        self._alert_fired = False
        self._lock = threading.Lock()

        # Load persisted usage for daily/monthly
        self._persisted_cost = 0.0
        if period in ("daily", "monthly"):
            self._persisted_cost = self._load_persisted_cost()

    @property
    def budget(self) -> float:
        """The budget limit in USD."""
        return self._budget

    @property
    def period(self) -> str:
        """The budget period."""
        return self._period

    @property
    def total_cost(self) -> float:
        """Total cost spent in current period."""
        with self._lock:
            session_cost = sum(u.cost for u in self._usage)
            return self._persisted_cost + session_cost

    @property
    def remaining(self) -> float:
        """Remaining budget in USD."""
        return max(0.0, self._budget - self.total_cost)

    @property
    def is_over_budget(self) -> bool:
        """Whether the budget has been exceeded."""
        return self.total_cost >= self._budget

    @property
    def usage_history(self) -> list[TokenUsage]:
        """Copy of usage history for this session."""
        with self._lock:
            return list(self._usage)

    @property
    def call_count(self) -> int:
        """Number of tracked calls in this session."""
        with self._lock:
            return len(self._usage)

    def track(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        *,
        input_cost_per_1k: float | None = None,
        output_cost_per_1k: float | None = None,
    ) -> TokenUsage:
        """Track a single LLM API call.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name for pricing.
            input_cost_per_1k: Override input cost per 1K tokens.
            output_cost_per_1k: Override output cost per 1K tokens.

        Returns:
            The TokenUsage record.

        Raises:
            TokenBudgetExceeded: If budget is exceeded and raise_on_exceed=True.

        """
        cost = calculate_cost(
            input_tokens,
            output_tokens,
            model,
            input_cost_per_1k=input_cost_per_1k,
            output_cost_per_1k=output_cost_per_1k,
        )

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            cost=cost,
        )

        # Determine what callbacks/actions to run OUTSIDE the lock
        alert_callback: Callable[[TokenTracker, TokenUsage], Any] | None = None
        budget_hit_callback: Callable[[TokenTracker, TokenUsage], Any] | None = None
        raise_exception = False

        with self._lock:
            self._usage.append(usage)
            total = self._persisted_cost + sum(u.cost for u in self._usage)

            # Persist for daily/monthly
            if self._period in ("daily", "monthly"):
                self._persist_cost(total - self._persisted_cost)

            # Check alert threshold
            if (
                self._alert_at is not None
                and not self._alert_fired
                and total >= self._budget * self._alert_at
            ):
                self._alert_fired = True
                alert_callback = self._on_alert

            # Check budget
            if total >= self._budget:
                budget_hit_callback = self._on_budget_hit
                raise_exception = self._raise_on_exceed

        # Call callbacks OUTSIDE the lock to avoid deadlock
        if alert_callback is not None:
            alert_callback(self, usage)

        if budget_hit_callback is not None:
            budget_hit_callback(self, usage)

        if raise_exception:
            raise TokenBudgetExceeded(self._budget, total, model)

        return usage

    def reset(self) -> None:
        """Reset session usage (does not affect persisted daily/monthly)."""
        with self._lock:
            self._usage.clear()
            self._alert_fired = False

    def reset_all(self) -> None:
        """Reset all usage including persisted data."""
        with self._lock:
            self._usage.clear()
            self._alert_fired = False
            self._persisted_cost = 0.0

            if self._period == "daily":
                self._get_daily_file().unlink(missing_ok=True)
            elif self._period == "monthly":
                self._get_monthly_file().unlink(missing_ok=True)

    def report(self) -> dict[str, Any]:
        """Get a summary report of usage.

        Returns:
            Dict with total_cost, calls, remaining, and other stats.

        """
        with self._lock:
            session_cost = sum(u.cost for u in self._usage)
            total = self._persisted_cost + session_cost
            return {
                "total_cost": total,
                "session_cost": session_cost,
                "persisted_cost": self._persisted_cost,
                "calls": len(self._usage),
                "remaining": max(0.0, self._budget - total),
                "budget": self._budget,
                "period": self._period,
                "is_over_budget": total >= self._budget,
            }

    def _get_daily_file(self) -> Path:
        """Get the daily usage file path."""
        return _get_storage_dir() / "daily.json"

    def _get_monthly_file(self) -> Path:
        """Get the monthly usage file path."""
        return _get_storage_dir() / "monthly.json"

    def _load_persisted_cost(self) -> float:
        """Load persisted cost for daily/monthly period."""
        if self._period == "daily":
            file = self._get_daily_file()
            period_key = "date"
            current_period = _today()
        else:
            file = self._get_monthly_file()
            period_key = "month"
            current_period = _this_month()

        if not file.exists():
            return 0.0

        try:
            data = json.loads(file.read_text())
            if data.get(period_key) != current_period:
                # Period rolled over, reset
                return 0.0
            return data.get("total_cost", 0.0)
        except (json.JSONDecodeError, KeyError):
            return 0.0

    def _persist_cost(self, session_cost: float) -> None:
        """Persist cost for daily/monthly period."""
        if self._period == "daily":
            file = self._get_daily_file()
            data = {"date": _today(), "total_cost": self._persisted_cost + session_cost}
        else:
            file = self._get_monthly_file()
            data = {"month": _this_month(), "total_cost": self._persisted_cost + session_cost}

        file.write_text(json.dumps(data))


def tokenguard(
    budget: float,
    *,
    alert_at: float | None = None,
    on_alert: Callable[[TokenTracker, TokenUsage], Any] | None = None,
    on_budget_hit: Callable[[TokenTracker, TokenUsage], Any] | None = None,
    raise_on_exceed: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for tracking token usage and enforcing budgets.

    The decorated function must return a dict with 'input_tokens', 'output_tokens',
    and 'model' keys, or the function must accept a 'tracker' keyword argument.

    Example:
        @tokenguard(budget=1.00, alert_at=0.80)
        def agent_call(prompt: str) -> dict:
            response = client.chat.completions.create(...)
            return {
                "result": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "model": "gpt-4",
            }

    Args:
        budget: Maximum budget in USD for the lifetime of this decorated function.
        alert_at: Fraction of budget to trigger alert (e.g., 0.8 for 80%).
        on_alert: Callback when alert threshold is reached.
        on_budget_hit: Callback when budget is exceeded.
        raise_on_exceed: Whether to raise TokenBudgetExceeded.

    Note:
        This decorator does not support async functions. For async code,
        use TokenTracker directly.

    """
    tracker = TokenTracker(
        budget=budget,
        period="session",
        alert_at=alert_at,
        on_alert=on_alert,
        on_budget_hit=on_budget_hit,
        raise_on_exceed=raise_on_exceed,
    )

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = fn(*args, **kwargs)

            # Try to extract token info from result
            if isinstance(result, dict):
                input_tokens = result.get("input_tokens")
                output_tokens = result.get("output_tokens")
                model = result.get("model")

                if input_tokens is not None and output_tokens is not None and model is not None:
                    tracker.track(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model=model,
                    )

            return result

        # Attach tracker methods to wrapper
        wrapper.tracker = tracker  # type: ignore[attr-defined]
        wrapper.reset = tracker.reset  # type: ignore[attr-defined]
        wrapper.report = tracker.report  # type: ignore[attr-defined]
        wrapper.total_cost = property(lambda self: tracker.total_cost)  # type: ignore[attr-defined]

        return wrapper

    return decorator


@contextmanager
def token_budget(
    budget: float,
    *,
    alert_at: float | None = None,
    on_alert: Callable[[TokenTracker, TokenUsage], Any] | None = None,
    on_budget_hit: Callable[[TokenTracker, TokenUsage], Any] | None = None,
    raise_on_exceed: bool = True,
) -> Iterator[TokenTracker]:
    """Context manager for scoped token budget tracking.

    Example:
        with token_budget(budget=2.00, alert_at=0.75) as guard:
            result = call_llm(prompt)
            guard.track(input_tokens=100, output_tokens=50, model="gpt-4")
            print(guard.report())

    Args:
        budget: Maximum budget in USD.
        alert_at: Fraction of budget to trigger alert.
        on_alert: Callback when alert threshold is reached.
        on_budget_hit: Callback when budget is exceeded.
        raise_on_exceed: Whether to raise TokenBudgetExceeded.

    Yields:
        TokenTracker instance for tracking usage.

    """
    tracker = TokenTracker(
        budget=budget,
        period="session",
        alert_at=alert_at,
        on_alert=on_alert,
        on_budget_hit=on_budget_hit,
        raise_on_exceed=raise_on_exceed,
    )
    yield tracker
