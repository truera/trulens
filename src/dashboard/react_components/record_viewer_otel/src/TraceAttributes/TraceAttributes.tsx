import { deduplicateAttributes } from '@/functions/deduplicateAttributes';
import { getSpanAttributeName } from '@/functions/getSpanAttributeName';
import { processCostAttributes } from '@/functions/processCostAttributes';
import { processSpanType } from '@/functions/processSpanType';
import { removeUnnecessaryAttributes } from '@/functions/removeUnnecessaryAttributes';
import { sortSpanKeys } from '@/functions/sortSpanKeys';
import type { Attributes } from '@/types/attributes';
import Panel from '@/Panel';
import { useMemo, ReactNode, Fragment } from 'react';
import { TraceContent } from '@/TraceContent/TraceContent';
import { processTokenAttributes } from '@/functions/processTokenAttributes';
import { Typography } from '@mui/material';

export interface TraceAttributesProps {
  attributes: Attributes;
}

/**
 * Thin utility component to handle the rendering of OTEL trace attributes, including
 * - Combining cost and currency
 * - Showing 'Not Specified' for unknown span types
 * - Deduplicating certain attributes
 * - Sorting the attributes
 * and other custom logic as needed.
 */
export const TraceAttributes = (props: TraceAttributesProps) => {
  const { attributes } = props;

  const displayResults = useMemo(() => {
    // Create a copy to not risk mutating the original.
    const attributesToProcess: Attributes = { ...attributes };
    const results: Record<string, ReactNode> = {};

    removeUnnecessaryAttributes(attributesToProcess);
    processCostAttributes(attributesToProcess);
    processTokenAttributes(attributesToProcess, results);
    processSpanType(attributesToProcess);
    deduplicateAttributes(attributesToProcess);

    // Eventually return an array of react nodes
    const sortedAttributes = Object.entries(attributesToProcess).sort(([aKey], [bKey]) => sortSpanKeys(aKey, bKey));

    sortedAttributes.forEach(([key, value]) => {
      results[getSpanAttributeName(key)] = (
        <Panel header={getSpanAttributeName(key)} key={key}>
          <TraceContent rawValue={value} />
        </Panel>
      );
    });

    return Object.keys(results)
      .sort((a, b) => a.localeCompare(b))
      .map((key) => <Fragment key={key}>{results[key]}</Fragment>);
  }, [attributes]);

  if (displayResults.length === 0) {
    return <Typography>No attributes to display</Typography>;
  }

  return <>{displayResults}</>;
};
