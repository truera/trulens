/**
 * Generates a stable string key from a value.
 *
 * This function handles different types of values:
 * - For arrays: sorts the array and converts to JSON string
 * - For objects: sorts object entries and converts to JSON string
 * - For other values: converts to string
 *
 * This is useful for creating consistent, deterministic keys for caching or comparison.
 *
 * @param value - The value to convert to a stable string key
 * @returns A string representation of the value that remains consistent for equivalent values
 */
export const getStableKey = (value: unknown): string => {
  if (Array.isArray(value)) {
    return JSON.stringify(value.sort());
  }
  if (value && typeof value === 'object') {
    return JSON.stringify(Object.entries(value).sort());
  }
  return String(value);
};
