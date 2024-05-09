import ReactJson, { ReactJsonViewProps } from '@microlink/react-json-view';
import { Streamlit } from 'streamlit-component-lib';

/**
 * Utility component as a rapper around react-json-view with default params
 */
export default function JSONViewer({ src }: { src: ReactJsonViewProps['src'] }) {
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
        style={{ fontSize: '14px' }}
      />
    </div>
  );
}
