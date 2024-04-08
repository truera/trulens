import React from 'react';
import { Skeleton, Typography } from '@mui/material';
import LabelAndValue from 'components/UI/LabelAndValue/LabelAndValue';

export function LabelAndValueSkeleton({ label }: { label: string }) {
  return (
    <LabelAndValue
      label={label}
      value={
        <Typography>
          <Skeleton width={30} />
        </Typography>
      }
    />
  );
}
