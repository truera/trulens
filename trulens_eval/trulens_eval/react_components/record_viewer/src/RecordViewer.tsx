import { Streamlit, StreamlitComponentBase, withStreamlitConnection } from 'streamlit-component-lib';
import { ReactNode } from 'react';
import './RecordViewer.css';
import { getStartAndEndTimesForNode, getTreeDepth } from './treeUtils';
import { DataRaw, StackTreeNode } from './types';
import { createTreeFromCalls } from './utils';
import { Box, Tooltip } from '@mui/material';
import { GridLines } from './TimelineBars';

class RecordViewer extends StreamlitComponentBase {
  public render = (): ReactNode => {
    // This seems to currently be the best way to type args, since
    // StreamlitComponentBase appears happy to just give it "any".
    const { record_json } = this.props.args as DataRaw;

    const { font: fontFamily } = this.props.theme as { font: string };
    const { width } = this.props as { width: number };

    const tree = createTreeFromCalls(record_json);
    const treeDepth = getTreeDepth(tree);
    const { startTime: treeStart, timeTaken: totalTime, endTime: treeEnd } = getStartAndEndTimesForNode(tree);

    const renderTree = () => {
      const children: ReactNode[] = [];

      const recursiveRender = (node: StackTreeNode, depth: number) => {
        const { startTime, timeTaken } = getStartAndEndTimesForNode(node);
        if (startTime >= treeEnd) return;

        const { name, methodName, path } = node;

        const description = (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <b>{name}</b>
            <span>
              <b>Time taken:</b> {timeTaken}ms
            </span>
            {methodName && (
              <span>
                <b>Method name:</b> {methodName}
              </span>
            )}
            {path && (
              <span>
                <b>Path:</b> {path}
              </span>
            )}
          </Box>
        );

        children.push(
          <Tooltip title={description} arrow>
            <div
              className="timeline"
              style={{
                left: `${((startTime - treeStart) / totalTime) * 100}%`,
                width: `${(timeTaken / totalTime) * 100}%`,
                top: depth * 32 + 16,
                fontFamily,
              }}
              onClick={() => {
                Streamlit.setComponentValue(node.raw?.perf.start_time ?? null);
              }}
            >
              <span className="timeline-component-name">{node.name}</span>
              <span className="timeline-time-taken">{timeTaken}ms</span>
            </div>
          </Tooltip>
        );

        for (const child of node.children ?? []) {
          recursiveRender(child, depth + 1);
        }
      };

      recursiveRender(tree, 0);

      return (
        <div
          style={{
            position: 'relative',
            gridColumnStart: 1,
            gridRowStart: 1,
          }}
        >
          {children}
        </div>
      );
    };

    const modifiedWidth = width - 16;

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, fontFamily, boxSizing: 'border-box' }}>
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
          {renderTree()}
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
