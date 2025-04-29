import { SpanAttributes } from '@/constants/span';
import type { Attributes } from '@/types/attributes';
import { getNumericalAttribute, getStringAttribute } from '@/functions/getAttribute';
import { formatNumber } from '@/functions/formatters';

/**
 * Process the cost attribute by combining the cost value and currency into a single string
 * if both exist
 */
export const processCostAttributes = (attributes: Attributes): void => {
  const costValue = getNumericalAttribute(attributes, SpanAttributes.COST_COST);
  const costCurrency = getStringAttribute(attributes, SpanAttributes.COST_COST_CURRENCY);

  if (costValue === null || costCurrency === null) return;

  let combinedCost = `${costValue < 1 ? costValue.toFixed(5) : costValue.toFixed(2)} ${costCurrency}`;

  try {
    // Try to format it nicer - e.g. $123.45 instead of 123.45 USD if the currency is recognized.
    combinedCost = formatNumber(costValue, {
      style: 'currency',
      currency: costCurrency,
      minimumFractionDigits: costValue < 1 ? 5 : 2,
      maximumFractionDigits: costValue < 1 ? 5 : 2,
    });
  } catch {
    // Currency not recognized - do nothing.
  }

  delete attributes[SpanAttributes.COST_COST];
  delete attributes[SpanAttributes.COST_COST_CURRENCY];
  attributes[SpanAttributes.COST_COST] = combinedCost;
};
