import { Box, SxProps } from '@mui/material';
import { Fragment } from 'react';

type GridLinesProps = {
  totalWidth: number;
  totalTime: number;
};

const MIN_WIDTH = 100;

const SECOND = 1000;
const MINUTE = 60 * SECOND;

const TIME_OPTIONS = [100, 500, 1 * SECOND, 5 * SECOND, 10 * SECOND, 30 * SECOND, MINUTE];

export default function GridLines({ totalWidth, totalTime }: GridLinesProps) {
  const maxCols = Math.floor(totalWidth / MIN_WIDTH);

  const timeOptionCols = TIME_OPTIONS.map((timeOption) => Math.floor(totalTime / timeOption));

  const timeOptionIndex = timeOptionCols.findIndex((c) => c < maxCols);
  const timeOption = timeOptionIndex !== -1 ? TIME_OPTIONS[timeOptionIndex] : 1;
  const numCols = Math.floor(totalTime / timeOption);
  const widthPerCol = (timeOption / totalTime) * totalWidth;

  return (
    <Box sx={containerSx}>
      {Array(numCols)
        .fill(undefined)
        .map((_c, i) => (
          <Fragment
            // eslint-disable-next-line react/no-array-index-key
            key={i}
          >
            <Box sx={{ ...lineSx, left: (i + 1) * widthPerCol }} />
            <span
              className="detail"
              style={{
                position: 'absolute',
                left: (i + 1) * widthPerCol + 4,
              }}
            >
              {(i + 1) * timeOption}ms
            </span>
          </Fragment>
        ))}
    </Box>
  );
}

const containerSx: SxProps = {
  display: 'flex',
  flexDirection: 'row',
  position: 'relative',
  gridColumnStart: 1,
  gridRowStart: 1,
  overflow: 'hidden',
};

const lineSx: SxProps = {
  height: '100%',
  minHeight: 20,
  width: '1px',
  backgroundColor: '#E0E0E0',
  position: 'absolute',
};
