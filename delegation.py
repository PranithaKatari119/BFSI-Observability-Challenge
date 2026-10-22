"""Fraud-referral delegated sub-agent for the BFSI challenge."""

import json

from llm_provider import chat_raw
from tools import TOOL_SPECS, TOOL_FUNCTIONS
from otel_setup import tracer


SUB_AGENT_MAX_STEPS = 4


FRAUD_SUB_AGENT_SYSTEM_PROMPT = (
    "You are the Fraud Referral Sub-Agent for Lloyds Financial (training scenario). "
    "You are delegated a single transaction amount and must decide, using lookup_policy "
    "to confirm the current threshold, whether it must be referred to the Fraud Operations "
    "queue. Answer with just 'REFER' or 'CLEAR' and one short sentence why."
)


def delegate_to_fraud_check(transaction_amount: float) -> dict:
    """Run the fraud-referral sub-agent inside the current trace context."""

    # Step 4:
    # start_as_current_span automatically uses the currently active
    # OpenTelemetry context as the parent.
    with tracer.start_as_current_span("sub_agent_turn") as span:

        span.set_attribute(
            "lab.delegated_to",
            "fraud_referral_sub_agent",
        )

        messages = [
            {
                "role": "system",
                "content": FRAUD_SUB_AGENT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Transaction amount: GBP "
                    f"{transaction_amount:.2f}. "
                    "Should this be referred?"
                ),
            },
        ]

        verdict_text = None
        sub_steps = 0

        for sub_steps in range(
            1,
            SUB_AGENT_MAX_STEPS + 1,
        ):
            turn = chat_raw(
                messages=messages,
                tools=TOOL_SPECS,
            )

            if not turn.tool_calls:
                verdict_text = turn.content
                break

            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                        for tc in turn.tool_calls
                    ],
                }
            )

            for tc in turn.tool_calls:
                args = json.loads(
                    tc.arguments or "{}"
                )

                fn = TOOL_FUNCTIONS[tc.name]

                result = fn(**args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    }
                )

        if verdict_text and "REFER" in verdict_text.upper():
            verdict = "REFER"
        elif verdict_text is None:
            verdict = "UNRESOLVED"
        else:
            verdict = "CLEAR"

        span.set_attribute(
            "bfsi.sub_agent.steps",
            sub_steps,
        )

        span.set_attribute(
            "bfsi.sub_agent.verdict",
            verdict,
        )

        return {
            "verdict": verdict,
            "detail": verdict_text,
            "sub_steps": sub_steps,
        }