/**
 * This file keeps track of the theming constants.
 */

export const COLOR_SCHEME = [
  "red",
  "orange",
  "yellow",
  "green",
  "blue",
  "indigo",
  "violet",
]

export const COLORS: Record<string, { 300: string; 900: string }> = {
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
