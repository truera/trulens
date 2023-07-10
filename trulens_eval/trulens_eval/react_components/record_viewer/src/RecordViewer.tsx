import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import { ReactNode } from "react"
import "./RecordViewer.css"
import { getStartAndEndTimesForNode } from "./treeUtils"
import { DataRaw, StackTreeNode } from "./types"
import { createTreeFromCalls } from "./utils"
import { COLORS, COLOR_SCHEME } from "./theme"

class RecordViewer extends StreamlitComponentBase {
  public render = (): ReactNode => {
    // This seems to currently be the best way to type args, since
    // StreamlitComponentBase appears happy to just give it "any".
    const { record_json } = this.props.args as DataRaw

    const tree = createTreeFromCalls(record_json)

    const renderTree = () => {
      const { startTime: treeStart, endTime: treeEnd } =
        getStartAndEndTimesForNode(tree)

      const totalTime = treeStart - treeEnd

      const children: ReactNode[] = []

      const recursiveRender = (node: StackTreeNode, depth: number) => {
        const { startTime, endTime } = getStartAndEndTimesForNode(node)

        children.push(
          <div
            className="timeline"
            style={{
              backgroundColor: COLORS[COLOR_SCHEME[depth]][300],
              color: COLORS[COLOR_SCHEME[depth]][900],
              border: `2px solid ${COLORS[COLOR_SCHEME[depth]][900]}`,
              left: `${((startTime - treeStart) / totalTime) * 100}%`,
              width: `${((endTime - startTime) / totalTime) * 100}%`,
              top: depth * 32,
              padding: 4,
              boxSizing: "border-box",
            }}
            onClick={() => {
              Streamlit.setComponentValue(node.raw?.perf.start_time ?? null)
            }}
          >
            {node.name}
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
            height: 32 * children.length,
            position: "relative",
          }}
        >
          {children}
        </div>
      )
    }

    return <div>{renderTree()}</div>
  }
}

// "withStreamlitConnection" is a wrapper function. It bootstraps the
// connection between your component and the Streamlit app, and handles
// passing arguments from Python -> Component.
//
// You don't need to edit withStreamlitConnection (but you're welcome to!).
const connectedRecordViewer = withStreamlitConnection(RecordViewer)
export default connectedRecordViewer
