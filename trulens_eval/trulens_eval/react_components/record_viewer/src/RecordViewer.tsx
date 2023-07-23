import { StreamlitComponentBase, withStreamlitConnection } from 'streamlit-component-lib';
import { ReactNode } from 'react';
import { DataRaw } from './utils/types';
import { createTreeFromCalls } from './utils/utils';
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

    /**
     * Actual code begins
     */
    const root = createTreeFromCalls(recordJSON, appJSON);

    return (
      <div style={{ fontFamily, display: 'flex', flexDirection: 'column', gap: 16 }}>
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
