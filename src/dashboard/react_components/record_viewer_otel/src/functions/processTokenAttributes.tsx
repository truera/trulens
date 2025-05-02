import { SpanAttributes } from '@/constants/span';
import { Attributes } from '@/types/attributes';
import { getNumericalAttribute } from './getAttribute';
import { ReactNode } from 'react';
import Panel from '@/Panel';
import { Typography } from '@mui/material';
import { getSpanAttributeName } from './getSpanAttributeName';

export const processTokenAttributes = (attributes: Attributes, results: Record<string, ReactNode>): void => {
  const tokenAttributeNameMapping: {
    [key in (typeof SpanAttributes)[keyof typeof SpanAttributes]]?: string;
  } = {
    [SpanAttributes.COST_NUM_PROMPT_TOKENS]: 'Prompt',
    [SpanAttributes.COST_NUM_COMPLETION_TOKENS]: 'Completion',
    [SpanAttributes.COST_NUM_GUARDRAIL_TOKENS]: 'Guardrail',
    [SpanAttributes.COST_NUM_CORTEX_GUARDRAIL_TOKENS]: 'Cortex Guardrail',
    [SpanAttributes.COST_NUM_TOKENS]: 'Total',
  };

  const tokenAttributes = Object.keys(tokenAttributeNameMapping);

  const validTokenAttributes = tokenAttributes.filter(
    (tokenAttribute) => getNumericalAttribute(attributes, tokenAttribute) !== null
  );
  if (validTokenAttributes.length === 0) return;

  const title = 'Token count (Usage)';

  results[title] = (
    <Panel header="Token count (Usage)">
      {validTokenAttributes.map((tokenAttribute) => (
        <Typography key={tokenAttribute}>
          <b>
            {tokenAttributeNameMapping[tokenAttribute as (typeof SpanAttributes)[keyof typeof SpanAttributes]] ??
              getSpanAttributeName(tokenAttribute) ??
              'Unknown'}
          </b>{' '}
          - {getNumericalAttribute(attributes, tokenAttribute)}
        </Typography>
      ))}
    </Panel>
  );

  // Delete the attributes to not show them twice.
  tokenAttributes.forEach((tokenAttribute) => delete attributes[tokenAttribute]);
};
