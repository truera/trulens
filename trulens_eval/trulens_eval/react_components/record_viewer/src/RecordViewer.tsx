import { StreamlitComponentBase, withStreamlitConnection } from 'streamlit-component-lib';
import { ReactNode } from 'react';
import './RecordViewer.css';
import { getStartAndEndTimesForNode, getTreeDepth } from './treeUtils';
import { DataRaw } from './types';
import { GridLines } from './TimelineBars';
import { createTreeFromCalls } from './utils';
import Tree from './Tree';

class RecordViewer extends StreamlitComponentBase {
  public render = (): ReactNode => {
    // This seems to currently be the best way to type args, since
    // StreamlitComponentBase appears happy to just give it "any".
    const { record_json } = this.props.args as DataRaw;

    const { font: fontFamily } = this.props.theme as { font: string };
    const { width } = this.props as { width: number };

    const tree = createTreeFromCalls(record_json);
    const treeDepth = getTreeDepth(tree);
    const { timeTaken: totalTime } = getStartAndEndTimesForNode(tree);

    const modifiedWidth = width - 16;

    return (
      <div style={{ fontFamily }}>
        <span className="detail">Total time taken: {totalTime / 1000}s</span>
        <div
          className="timeline-container"
          style={{
            width: modifiedWidth,
            gridTemplateRows: 32 * treeDepth + 16,
            gridTemplateColumns: modifiedWidth,
            height: 32 * treeDepth + 16,
          }}
        >
          <GridLines totalWidth={modifiedWidth} totalTime={totalTime} />
          <Tree tree={tree} />
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
