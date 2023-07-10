import { StackTreeNode } from "./types"

/**
 * Gets the start and end times for the tree node provided, defaulting to 0.
 *
 * @param node - the tree node to obtain the start and end times for
 * @returns an object containing the start and end times based on the tree node
 * provided.
 */
export const getStartAndEndTimesForNode = (node: StackTreeNode) => {
  return {
    startTime: node.startTime?.getTime() ?? 0,
    endTime: node.endTime?.getTime() ?? 0,
  }
}
