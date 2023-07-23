import { StreamlitComponentBase, withStreamlitConnection } from 'streamlit-component-lib';
import { ReactNode } from 'react';
import './RecordViewer.css';
import { getStartAndEndTimesForNode, getTreeDepth } from './utils/treeUtils';
import { DataRaw } from './utils/types';
import GridLines from './GridLines';
import { createTreeFromCalls } from './utils/utils';
import { TIME_DISPLAY_HEIGHT_BUFFER } from './utils/styling';
import TimelineBars, { BAR_HEIGHT } from './TimelineBars';
import RecordTable from './RecordTable/RecordTable';

class RecordViewer extends StreamlitComponentBase {
  public render = (): ReactNode => {
    /**
     * Extracting args and theme from streamlit args
     */

    // This seems to currently be the best way to type args, since
    // StreamlitComponentBase appears happy to just give it "any".
    const { record_json: recordJSON, app_json: appJSON } = this.props.args as DataRaw;

    const { font: fontFamily } = this.props.theme as { font: string };
    const { width } = this.props as { width: number };

    /**
     * Actual code begins
     */
    const root = createTreeFromCalls(recordJSON, appJSON);
    const treeDepth = getTreeDepth(root);
    const { timeTaken: totalTime } = getStartAndEndTimesForNode(root);

    return (
      <div style={{ fontFamily, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <span className="detail">Total time taken: {totalTime / 1000}s</span>
        <div
          className="timeline-container"
          style={{
            gridTemplateColumns: width,
            gridTemplateRows: BAR_HEIGHT * treeDepth + TIME_DISPLAY_HEIGHT_BUFFER,
          }}
        >
          <GridLines totalWidth={width} totalTime={totalTime} />
          <TimelineBars root={root} />
        </div>
        <RecordTable root={root} />
      </div>
    );
  };
}

// "withStreamlitConnection" is a wrapper function. It bootstraps the
// connection between your component and the Streamlit app, and handles
// passing arguments from Python -> Component.
//
// You don't need to edit withStreamlitConnection (but you're welcome to!).
const connectedRecordViewer = withStreamlitConnection(RecordViewer);
export default connectedRecordViewer;
