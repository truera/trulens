export const splitStringToTwoPartsByDelimiter = (input: string, delimiter: string): [string, string][] => {
  const results: [string, string][] = [];

  const splitInput = input.split(delimiter);
  for (let i = splitInput.length - 1; i > 0; i--) {
    const newKey = splitInput.slice(0, i).join(delimiter);
    results.push([newKey, splitInput.slice(i).join(delimiter)]);
  }

  return results;
};
