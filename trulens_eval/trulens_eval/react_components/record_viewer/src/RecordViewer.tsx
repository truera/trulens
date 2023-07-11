import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import { ReactNode } from "react"
import "./RecordViewer.css"
import { getStartAndEndTimesForNode, getTreeDepth } from "./treeUtils"
import { DataRaw, StackTreeNode } from "./types"
import { createTreeFromCalls } from "./utils"

class RecordViewer extends StreamlitComponentBase {
  public render = (): ReactNode => {
    // This seems to currently be the best way to type args, since
    // StreamlitComponentBase appears happy to just give it "any".
    const { record_json } = this.props.args as DataRaw

    const { font: fontFamily } = this.props.theme as { font: string }

    const tree = createTreeFromCalls(record_json)
    const treeDepth = getTreeDepth(tree)
    const { startTime: treeStart, timeTaken: totalTime } =
      getStartAndEndTimesForNode(tree)

    const renderTree = () => {
      const children: ReactNode[] = []

      const recursiveRender = (node: StackTreeNode, depth: number) => {
        const { startTime, timeTaken } = getStartAndEndTimesForNode(node)

        children.push(
          <div
            className="timeline"
            style={{
              left: `${((startTime - treeStart) / totalTime) * 100}%`,
              width: `${(timeTaken / totalTime) * 100}%`,
              top: depth * 32,
              fontFamily,
            }}
            onClick={() => {
              Streamlit.setComponentValue(node.raw?.perf.start_time ?? null)
            }}
          >
            <span className="timeline-component-name">{node.name}</span>
            <span className="timeline-time-taken">{timeTaken}ms</span>
          </div>
        )

        for (const child of node.children ?? []) {
          recursiveRender(child, depth + 1)
        }
      }

      recursiveRender(tree, 0)

      return (
        <div
          style={{
            height: 32 * treeDepth + 8, // + 8 for padding
            position: "relative",
          }}
        >
          {children}
        </div>
      )
    }

    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <span style={{ fontFamily }}>
          Total time taken: {totalTime / 1000}s
        </span>
        {renderTree()}
      </div>
    )
  }
}

// "withStreamlitConnection" is a wrapper function. It bootstraps the
// connection between your component and the Streamlit app, and handles
// passing arguments from Python -> Component.
//
// You don't need to edit withStreamlitConnection (but you're welcome to!).
const connectedRecordViewer = withStreamlitConnection(RecordViewer)
export default connectedRecordViewer
