import { StreamlitComponentBase, withStreamlitConnection } from 'streamlit-component-lib';
import { ReactNode } from 'react';
import './RecordViewer.css';
import { getStartAndEndTimesForNode, getTreeDepth } from './treeUtils';
import { DataRaw } from './types';
import GridLines from './GridLines';
import { createTreeFromCalls } from './utils';
import { TIME_DISPLAY_HEIGHT_BUFFER } from './styling';
import TimelineBars, { BAR_HEIGHT } from './TimelineBars';

class RecordViewer extends StreamlitComponentBase {
  public render = (): ReactNode => {
    /**
     * Extracting args and theme from streamlit args
     */

    // This seems to currently be the best way to type args, since
    // StreamlitComponentBase appears happy to just give it "any".
    const { record_json: recordJSON } = this.props.args as DataRaw;

    const { font: fontFamily } = this.props.theme as { font: string };
    const { width } = this.props as { width: number };

    /**
     * Actual code begins
     */
    const root = createTreeFromCalls(recordJSON);
    const treeDepth = getTreeDepth(root);
    const { timeTaken: totalTime } = getStartAndEndTimesForNode(root);

    return (
      <div style={{ fontFamily }}>
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
