import { ReactNode } from 'react';
import { StreamlitComponentBase, withStreamlitConnection } from 'streamlit-component-lib';

import RecordInfo from '@/RecordInfo';
import { ThemeProvider } from '@/utils/ThemeProvider';
import { DataRaw } from '@/utils/types';
import { createNodeMap, createTreeFromCalls } from '@/utils/utils';

/**
 * This component serves as our entryway into streamlit. Keeping the logic here at a minimum,
 * primarily parsing and shaping the args/props. Primary component can be found in RecordInfo.
 */
class RecordViewer extends StreamlitComponentBase {
  public render = (): ReactNode => {
    /**
     * Extracting args and theme from streamlit args
     */
    // This seems to currently be the best way to type args, since
    // StreamlitComponentBase appears happy to just give it "any".
    const { spans } = this.props.args as DataRaw;

    /**
     * Actual code begins
     */
    const root = createTreeFromCalls(spans);
    console.log({ root });

    const nodeMap = createNodeMap(root);

    return (
      <ThemeProvider streamlitTheme={this.props.theme}>
        <RecordInfo root={root} nodeMap={nodeMap} />
      </ThemeProvider>
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
