import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import { ReactNode } from "react"
import "./App.css"
import {
  CallJSONRaw,
  RecordJSONRaw,
  StackJSONRaw,
  StackTreeNode,
} from "./types"
import { getNameFromCell, getStartAndEndTimes } from "./utils"

// let's make an assumption that the nodes are
// 1. the stack cell method obj name must match
// 2. the stack cell must be within the time

const addCallToTree = (
  tree: StackTreeNode,
  call: CallJSONRaw,
  stack: StackJSONRaw[],
  index: number
) => {
  const stackCell = stack[index]

  if (!tree.children) tree.children = []

  // otherwise, we are deciding which node to go in
  let matchingNode = tree.children.find(
    (node) =>
      node.name === getNameFromCell(stackCell) &&
      (node.startTime ?? 0) <= new Date(call.perf.start_time) &&
      (node.endTime ?? Infinity) >= new Date(call.perf.end_time)
  )

  // if we are currently at the top most cell of the stack
  if (index === stack.length - 1) {
    const { startTime, endTime } = getStartAndEndTimes(call.perf)

    if (matchingNode) {
      matchingNode.startTime = startTime
      matchingNode.endTime = endTime
      matchingNode.raw = call

      return
    }

    tree.children.push({
      children: undefined,
      name: getNameFromCell(stackCell),
      startTime,
      endTime,
      raw: call,
    })

    return
  }

  if (!matchingNode) {
    const newNode = {
      children: [],
      name: getNameFromCell(stackCell),
    }

    // otherwise create a new node
    tree.children.push(newNode)
    matchingNode = newNode
  }

  addCallToTree(matchingNode, call, stack, index + 1)
}

const createTreeFromCalls = (recordJSON: RecordJSONRaw) => {
  const tree: StackTreeNode = {
    children: [],
    name: "App",
    startTime: new Date(recordJSON.perf.start_time),
    endTime: new Date(recordJSON.perf.end_time),
  }

  for (const call of recordJSON.calls) {
    addCallToTree(tree, call, call.stack, 0)
  }

  return tree
}

const COLOR_SCHEME = [
  "red",
  "orange",
  "yellow",
  "green",
  "blue",
  "indigo",
  "violet",
]

const COLORS: Record<string, { 300: string; 900: string }> = {
  red: {
    300: "#feb2b2",
    900: "#742a2a",
  },
  orange: {
    300: "#fbd38d",
    900: "#7b341e",
  },
  yellow: {
    300: "#faf089",
    900: "#744210",
  },
  green: {
    300: "#9ae6b4",
    900: "#22543d",
  },
  blue: {
    300: "#90cdf4",
    900: "#2a4365",
  },
  indigo: {
    300: "#a3bffa",
    900: "#3c366b",
  },
}

/**
 * This is a React-based component template. The `render()` function is called
 * automatically when your component should be re-rendered.
 */
class RecordViewer extends StreamlitComponentBase {
  public render = (): ReactNode => {
    const { record_json } = this.props.args

    const tree = createTreeFromCalls(record_json)
    const renderTree = () => {
      const totalTime =
        (tree.endTime?.getTime() ?? 0) - (tree.startTime?.getTime() ?? 0)

      const children: ReactNode[] = []

      const recursiveRender = (node: StackTreeNode, depth: number) => {
        const start = node.startTime?.getTime() ?? 0
        const end = node.endTime?.getTime() ?? 0

        children.push(
          <div
            className="timeline"
            style={{
              backgroundColor: COLORS[COLOR_SCHEME[depth]][300],
              color: COLORS[COLOR_SCHEME[depth]][900],
              border: `2px solid ${COLORS[COLOR_SCHEME[depth]][900]}`,
              left: `${
                ((start - (tree.startTime?.getTime() ?? 0)) / totalTime) * 100
              }%`,
              width: `${((end - start) / totalTime) * 100}%`,
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
export default withStreamlitConnection(RecordViewer)
