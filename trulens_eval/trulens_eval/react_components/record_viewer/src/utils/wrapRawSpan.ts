import type { SpanRaw } from '@/schema/span';
import { Span } from './Span';

const wrapRawSpan = (rawSpan: SpanRaw) => {
  Object.keys(rawSpan).forEach((key) => {
    Object.defineProperty(rawSpan, key, {
      get() {
        return rawSpan.attributes[Span.vendorAttr(key)] as (typeof rawSpan)[key];
      },
    });
  });

  return rawSpan;
};
