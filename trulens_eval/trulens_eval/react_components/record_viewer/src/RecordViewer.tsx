import { StreamlitComponentBase, withStreamlitConnection } from 'streamlit-component-lib';
import { ReactNode } from 'react';
import { DataRaw } from './utils/types';
import { createNodeMap, createTreeFromCalls } from './utils/utils';
import RecordInfo from './RecordInfo';

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
    const { record_json: recordJSON, app_json: appJSON } = this.props.args as DataRaw;

    /**
     * Actual code begins
     */
    const root = createTreeFromCalls(recordJSON, appJSON.app_id);
    const nodeMap = createNodeMap(root);

    return <RecordInfo root={root} recordJSON={recordJSON} nodeMap={nodeMap} appJSON={appJSON} />;
  };
}

// "withStreamlitConnection" is a wrapper function. It bootstraps the
// connection between your component and the Streamlit app, and handles
// passing arguments from Python -> Component.
//
// You don't need to edit withStreamlitConnection (but you're welcome to!).
const connectedRecordViewer = withStreamlitConnection(RecordViewer);
export default connectedRecordViewer;
