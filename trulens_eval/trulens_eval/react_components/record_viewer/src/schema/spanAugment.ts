/**
 * This file contains type augments for the JSON-schema generated types.
 * Typically contains enums (because they're not easily transmitted
 * over via type generation from schemas)
 */

/**
 * Represents the canonical set of status codes of a finished Span.
 */
export enum StatusCode {
  /**
   * The default status.
   */
  UNSET = 0,

  /**
   * The operation has been validated by an Application developer or Operator to have completed successfully.
   */
  OK = 1,

  /**
   * The operation contains an error.
   */
  ERROR = 2,
}

/**
 * Specifies additional details on how this span relates to its parent span.
 *
 * Note that this enumeration is experimental and likely to change. See
 * https://github.com/open-telemetry/opentelemetry-specification/pull/226.
 */
export enum SpanKind {
  /**
   * Default value. Indicates that the span is used internally in the
   * application.
   */
  INTERNAL = 0,

  /**
   * Indicates that the span describes an operation that handles a remote
   * request.
   */
  SERVER = 1,

  /**
   * Indicates that the span describes a request to some remote service.
   */
  CLIENT = 2,

  /**
   * Indicates that the span describes a producer sending a message to a
   * broker. Unlike client and server, there is usually no direct critical
   * path latency relationship between producer and consumer spans.
   */
  PRODUCER = 3,

  /**
   * Indicates that the span describes a consumer receiving a message from a
   * broker. Unlike client and server, there is usually no direct critical
   * path latency relationship between producer and consumer spans.
   */
  CONSUMER = 4,
}
