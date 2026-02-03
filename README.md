# TokenGuard

Simple, zero-dependency token tracking and budget enforcement for LLM API calls.

Part of the **Guard Suite** for AI agent reliability:
- [LoopGuard](https://github.com/yolojewjitsu/loopguard) - Prevents infinite loops
- [EvalGuard](https://github.com/yolojewjitsu/evalguard) - Validates agent outputs
- [FailGuard](https://github.com/yolojewjitsu/failguard) - Detects silent failures
- **TokenGuard** - Controls costs

## Installation

```bash
pip install tokenguard
```

## Quick Start

### Track Token Costs

```python
from tokenguard import TokenTracker

tracker = TokenTracker(budget=5.00)

# After any LLM call
tracker.track(input_tokens=150, output_tokens=200, model="gpt-4")

print(tracker.total_cost)      # $0.0135
print(tracker.remaining)       # $4.9865
print(tracker.is_over_budget)  # False
```

### Budget Enforcement

```python
from tokenguard import TokenTracker, TokenBudgetExceeded

tracker = TokenTracker(budget=1.00, alert_at=0.8)

try:
    # This will raise TokenBudgetExceeded if budget is exceeded
    tracker.track(input_tokens=50000, output_tokens=25000, model="gpt-4")
except TokenBudgetExceeded as e:
    print(f"Budget exceeded: ${e.spent:.2f} of ${e.budget:.2f}")
```

### Context Manager

```python
from tokenguard import token_budget

with token_budget(budget=2.00, alert_at=0.75) as guard:
    # Your LLM calls here
    guard.track(input_tokens=100, output_tokens=50, model="claude-3-sonnet")
    guard.track(input_tokens=200, output_tokens=100, model="claude-3-sonnet")

    print(guard.report())
    # {"total_cost": 0.0045, "calls": 2, "remaining": 1.9955, ...}
```

### Decorator (Auto-extract from Result)

```python
from tokenguard import tokenguard

@tokenguard(budget=1.00, alert_at=0.80)
def agent_call(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "result": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "model": "gpt-4",
    }

# Usage tracked automatically
result = agent_call("Hello!")
print(agent_call.tracker.total_cost)
```

## Features

### Built-in Model Pricing

TokenGuard includes pricing for 15+ popular models:

**OpenAI:** gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo, o1, o1-mini, o3-mini

**Anthropic:** claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3.5-sonnet, claude-3.5-haiku, claude-sonnet-4, claude-opus-4

**Google:** gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash

### Custom Pricing

```python
from tokenguard import set_model_cost, calculate_cost

# Add custom model
set_model_cost("my-fine-tuned-model", input=0.01, output=0.02)

# Or override at track time
tracker.track(
    input_tokens=100,
    output_tokens=50,
    model="gpt-4",
    input_cost_per_1k=0.025,  # Volume discount
    output_cost_per_1k=0.05,
)
```

### Alert Callbacks

```python
def alert_handler(tracker, usage):
    print(f"Warning: {tracker.total_cost / tracker.budget * 100:.0f}% of budget used")

def budget_handler(tracker, usage):
    print(f"Budget exceeded! Spent ${tracker.total_cost:.2f}")

tracker = TokenTracker(
    budget=10.00,
    alert_at=0.80,
    on_alert=alert_handler,
    on_budget_hit=budget_handler,
    raise_on_exceed=False,  # Don't raise, just call handler
)
```

### Daily/Monthly Budgets

```python
# Track daily spending (persisted to ~/.tokenguard/daily.json)
daily_tracker = TokenTracker(budget=10.00, period="daily")

# Track monthly spending (persisted to ~/.tokenguard/monthly.json)
monthly_tracker = TokenTracker(budget=100.00, period="monthly")

# Usage persists across sessions
daily_tracker.track(input_tokens=1000, output_tokens=500, model="gpt-4")
# ... restart your app ...
# Daily tracker will remember previous spending
```

## Use with Guard Suite

```python
from loopguard import loopguard
from evalguard import check
from failguard import failguard
from tokenguard import tokenguard

@loopguard(max_repeats=5)
@tokenguard(budget=1.00, alert_at=0.80)
@failguard(max_latency_drift=2.0)
@check(not_contains=["ERROR", "I don't know"])
def reliable_agent(query: str) -> dict:
    response = llm.complete(query)
    return {
        "result": response.content,
        "input_tokens": response.usage.input,
        "output_tokens": response.usage.output,
        "model": "gpt-4",
    }
```

## API Reference

### TokenTracker

```python
TokenTracker(
    budget: float,              # Max USD
    period: str = "session",    # "session", "daily", or "monthly"
    alert_at: float = None,     # Alert at this fraction (0.0-1.0)
    on_alert: Callable = None,  # Called when alert threshold reached
    on_budget_hit: Callable = None,  # Called when budget exceeded
    raise_on_exceed: bool = True,    # Raise TokenBudgetExceeded
)
```

**Properties:**
- `total_cost` - Total spent in current period
- `remaining` - Budget remaining
- `is_over_budget` - Whether budget exceeded
- `call_count` - Number of tracked calls
- `usage_history` - List of TokenUsage records

**Methods:**
- `track(input_tokens, output_tokens, model, ...)` - Track a call
- `reset()` - Reset session usage
- `reset_all()` - Reset all usage including persisted
- `report()` - Get summary dict

### tokenguard decorator

```python
@tokenguard(
    budget: float,              # Max USD for this function
    alert_at: float = None,
    on_alert: Callable = None,
    on_budget_hit: Callable = None,
    raise_on_exceed: bool = True,
)
def my_func() -> dict:
    # Must return dict with input_tokens, output_tokens, model
    ...
```

### token_budget context manager

```python
with token_budget(budget=1.00, alert_at=0.8) as guard:
    guard.track(...)
```

## License

MIT
