import ReactJson, { ReactJsonViewProps } from '@microlink/react-json-view';
import { useColorScheme } from '@mui/material';
import { Streamlit } from 'streamlit-component-lib';

/**
 * Utility component as a wrapper around react-json-view with default params
 */
export default function JSONViewer({ src }: { src: ReactJsonViewProps['src'] }) {
  const { mode } = useColorScheme();

  return (
    // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions
    <div
      // Streamlit misses the need to update height when the json is collapsed
      onClick={() => Streamlit.setFrameHeight()}
    >
      <ReactJson
        src={src ?? {}}
        name={null}
        collapsed={2}
        collapseStringsAfterLength={140}
        style={{ fontSize: '14px', fontFamily: 'Source Code Pro' }}
        theme={mode === 'light' ? 'rjv-default' : 'harmonic'}
      />
    </div>
  );
}
