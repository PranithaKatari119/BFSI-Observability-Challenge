"""OpenAI-only chat wrapper. THIS FILE HAS NO OBSERVABILITY OF ANY KIND -
no spans, and `usage` below is read from the API response and then
immediately discarded. The token counts are right there in `response.usage`
on every single call; nothing converts them into a cost or records them
anywhere. That's Step 5 of the outline: add pricing.py, and both a
lab.cost_usd span attribute AND a proper cost metric - once Step 1 has
given you a tracer and a meter to record them with."""


import os

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from otel_setup import tracer, cost_histogram
from pricing import calculate_cost_usd


ENV_FILE = Path(__file__).resolve().parent / ".env"

load_dotenv(
    dotenv_path=ENV_FILE
)


DEFAULT_MODEL = "gpt-4o-mini"

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Create the OpenAI client when first required."""

    global _client

    if _client is None:
        api_key = os.environ.get(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                f"Check your .env file at: {ENV_FILE}"
            )

        _client = OpenAI(
            api_key=api_key
        )

    return _client


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class AgentTurn:
    content: Optional[str]

    tool_calls: list[ToolCall] = field(
        default_factory=list
    )

    finish_reason: str = ""
    raw: object = None


def chat_raw(
    messages,
    tools=None,
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = None,
) -> AgentTurn:
    """Call OpenAI and record tracing, tokens and estimated cost."""

    client = _get_client()

    # Every LLM request gets its own span.
    # start_as_current_span ensures it becomes a child of
    # agent_turn or sub_agent_turn when those spans are active.
    with tracer.start_as_current_span(
        "model_call"
    ) as span:

        # ----------------------------------------------------
        # Request attributes
        # ----------------------------------------------------

        span.set_attribute(
            "gen_ai.system",
            "openai",
        )

        span.set_attribute(
            "gen_ai.request.model",
            model,
        )

        kwargs = {
            "model": model,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        if temperature is not None:
            kwargs["temperature"] = temperature

        # ----------------------------------------------------
        # OpenAI API call
        # ----------------------------------------------------

        response = client.chat.completions.create(
            **kwargs
        )

        choice = response.choices[0]

        # ----------------------------------------------------
        # Actual model returned by OpenAI
        # ----------------------------------------------------

        response_model = getattr(
            response,
            "model",
            model,
        )

        span.set_attribute(
            "gen_ai.response.model",
            response_model,
        )

        # ----------------------------------------------------
        # Token usage
        # ----------------------------------------------------

        input_tokens = 0
        output_tokens = 0

        if response.usage is not None:

            input_tokens = (
                response.usage.prompt_tokens
                or 0
            )

            output_tokens = (
                response.usage.completion_tokens
                or 0
            )

        span.set_attribute(
            "gen_ai.usage.input_tokens",
            input_tokens,
        )

        span.set_attribute(
            "gen_ai.usage.output_tokens",
            output_tokens,
        )

        # ----------------------------------------------------
        # Cost calculation
        # ----------------------------------------------------

        cost_usd = calculate_cost_usd(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        span.set_attribute(
            "lab.cost_usd",
            cost_usd,
        )

        # ----------------------------------------------------
        # Proper OpenTelemetry metric
        # ----------------------------------------------------

        cost_histogram.record(
            cost_usd,
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": model,
            },
        )

        # ----------------------------------------------------
        # Existing tool-call conversion
        # ----------------------------------------------------

        tool_calls = []

        for tc in (
            choice.message.tool_calls or []
        ):
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
            )

        return AgentTurn(
            content=choice.message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            raw=response,
        )
