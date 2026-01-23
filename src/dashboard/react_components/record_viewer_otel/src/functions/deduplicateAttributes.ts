import { uniq } from './uniq';
import { getSpanAttributeName } from './getSpanAttributeName';
import { getStableKey } from './getStableKey';
import { Attributes } from '@/types/attributes';

/**
 * Processor that deduplicates attributes with the same value so that we show
 * duplicates only once. The key of the attribute will be a concatenation of
 * the original keys.
 */
export const deduplicateAttributes = (attributes: Attributes): void => {
  const duplicateValuesChecker = new Map<string, string[]>();

  Object.entries(attributes).forEach(([key, value]) => {
    const stableKey = getStableKey(value);
    const existing = duplicateValuesChecker.get(stableKey);
    if (existing) {
      existing.push(key);
    } else {
      duplicateValuesChecker.set(stableKey, [key]);
    }
  });

  duplicateValuesChecker.forEach((attributeKeys) => {
    if (attributeKeys.length <= 1) return;

    attributes[
      uniq(attributeKeys.map((attributeKey) => getSpanAttributeName(attributeKey)))
        .sort((a, b) => a.localeCompare(b))
        .join(' | ')
    ] = attributes[attributeKeys[0]];
    attributeKeys.forEach((key) => delete attributes[key]);
  });
};
