# 🔭 OpenTelemetry

With the integration of OpenTelemetry, TruLens can now work seamlessly with existing setups using OpenTelemetry in both:

1. Having TruLens use spans emitted by non-TruLens code.
1. Having existing setup take advantage of spans emitted by TruLens.

Our [semantic conventions](./semantic_conventions) also lay out how to both emit spans natively used by TruLens' metric computation without using the TruLens python library (such as in the case when tracing in other languages) and/or what to expect of TruLens spans when developing against our emitted OpenTelemetry traces.
