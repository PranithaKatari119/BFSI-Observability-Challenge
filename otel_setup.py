"""OpenTelemetry setup for the BFSI Observability Challenge."""

import os

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource

# ---- Tracing imports ----
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)

# ---- Metrics imports ----
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)


# ============================================================
# Shared resource information
# ============================================================

resource = Resource.create(
    {
        "service.name": "bfsi-virtual-assistant",
    }
)


# ============================================================
# OTLP endpoint
# ============================================================

OTLP_ENDPOINT = os.getenv(
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "http://localhost:4317",
)


# ============================================================
# TRACE SETUP
# ============================================================

trace_provider = TracerProvider(
    resource=resource,
)

trace.set_tracer_provider(
    trace_provider
)


try:
    otlp_span_exporter = OTLPSpanExporter(
        endpoint=OTLP_ENDPOINT,
        insecure=True,
    )

    trace_provider.add_span_processor(
        BatchSpanProcessor(
            otlp_span_exporter
        )
    )

except Exception:
    # Console fallback for local debugging.
    trace_provider.add_span_processor(
        BatchSpanProcessor(
            ConsoleSpanExporter()
        )
    )


tracer = trace.get_tracer(
    "bfsi-virtual-assistant"
)


# ============================================================
# METRIC SETUP
# ============================================================

otlp_metric_exporter = OTLPMetricExporter(
    endpoint=OTLP_ENDPOINT,
    insecure=True,
)


metric_reader = PeriodicExportingMetricReader(
    otlp_metric_exporter,
    export_interval_millis=5000,
)


meter_provider = MeterProvider(
    resource=resource,
    metric_readers=[
        metric_reader,
    ],
)


metrics.set_meter_provider(
    meter_provider
)


meter = metrics.get_meter(
    "bfsi-virtual-assistant"
)


# ============================================================
# COST HISTOGRAM
# ============================================================

cost_histogram = meter.create_histogram(
    name="lab.cost_usd",
    description="Estimated USD cost of each model call",
    unit="USD",
)