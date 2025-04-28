import { SpanAttributes, SpanType } from '@/constants/span';
import type { Attributes } from '@/types/attributes';
import { getStringAttribute } from '@/functions/getAttribute';

/**
 * Process the span type attribute by showing 'Not Specified' if the value is 'UNKNOWN'
 */
export const processSpanType = (attributes: Attributes): void => {
  if (!attributes[SpanAttributes.SPAN_TYPE]) return;

  const spanType = getStringAttribute(attributes, SpanAttributes.SPAN_TYPE);
  delete attributes[SpanAttributes.SPAN_TYPE];

  attributes[SpanAttributes.SPAN_TYPE] = spanType === SpanType.UNKNOWN ? 'Not Specified' : spanType;
};
