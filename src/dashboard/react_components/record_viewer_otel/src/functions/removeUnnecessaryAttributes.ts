import { SpanAttributes } from '@/constants/span';
import type { Attributes } from '@/types/attributes';

export const removeUnnecessaryAttributes = (attributes: Attributes): void => {
  // Not needed since we have function name
  delete attributes['name'];

  // Not needed since the user is always in an application context
  delete attributes[SpanAttributes.APP_NAME];
  delete attributes[SpanAttributes.APP_VERSION];

  // Deleting these attributes since they are not needed for the UI
  delete attributes[SpanAttributes.RECORD_ID];
  delete attributes[SpanAttributes.RUN_NAME];
  delete attributes[SpanAttributes.INPUT_ID];

  // Displaying the span type is handled by the UI
  delete attributes[SpanAttributes.SPAN_TYPE];
};
