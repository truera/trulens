import { PerfJSONRaw, StackJSONRaw } from "./types"

/**
 * Gets the name of the calling function in the stack cell.
 *
 * TODO: Is this the best name to use for the component/stack?
 *
 * See README.md for an overview of the terminology.
 *
 * @param stackCell - StackJSONRaw Cell in the stack of a call.
 * @returns name of the calling function in the stack cell.
 */
export const getNameFromCell = (stackCell: StackJSONRaw) => {
  return stackCell.method.obj.cls.name
}

/**
 * Gets the start and end times based on the performance
 * data provided.
 * 
 * @param perf - PerfJSONRaw The performance data to be analyzed
 * @returns an object containing the start and end times based on the performance
 * data provided
 */
export const getStartAndEndTimes = (perf: PerfJSONRaw) => {
    return ({
        startTime: perf?.start_time ? new Date(perf.start_time): undefined,
        endTime: perf?.end_time ? new Date(perf.end_time): undefined,
    })
}