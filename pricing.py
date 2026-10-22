"""Token-cost calculation for the BFSI observability challenge."""

# USD price per 1 million tokens.
# Training/lab pricing table used by this project.
MODEL_PRICING = {
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
    },
}


def calculate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate estimated USD cost for one model call."""

    pricing = MODEL_PRICING.get(model)

    if pricing is None:
        return 0.0

    input_cost = (
        input_tokens / 1_000_000
    ) * pricing["input"]

    output_cost = (
        output_tokens / 1_000_000
    ) * pricing["output"]

    return input_cost + output_cost