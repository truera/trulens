/**
 * TruApp — wraps an app object so that its designated "main" method is
 * automatically traced with a RECORD_ROOT span, removing the need to call
 * `withRecord()` on every invocation.
 *
 * Mirrors the Python `TruBasicApp` / `TruChain` pattern.
 */

import { withRecord, type WithRecordOptions } from "./instrument.js";

type AnyFunction = (...args: any[]) => any;

export interface TruAppOptions<
  TApp extends Record<string, any>,
  TMethod extends keyof TApp,
> {
  /** Name of the method on the wrapped app to treat as the entry point. */
  mainMethod: TMethod & string;
  /**
   * Extract the human-readable "input" from the method's arguments.
   * The return value is recorded as `RECORD_ROOT.INPUT`.
   * Defaults to `JSON.stringify(args)`.
   */
  mainInput?: (...args: Parameters<TApp[TMethod]>) => unknown;
  /**
   * Additional per-record options (e.g. `runName`, `groundTruthOutput`).
   */
  recordOptions?: Omit<WithRecordOptions, "input">;
}

/**
 * Wrap `app` so that calling the main method automatically creates a
 * `RECORD_ROOT` span — no per-call `withRecord()` needed.
 *
 * Returns a proxied version of the app. All other methods/properties are
 * forwarded untouched.
 *
 * @example
 * ```ts
 * const rag = new SimpleRAG();
 * const app = createTruApp(rag, {
 *   mainMethod: "query",
 *   mainInput: (question) => question,
 * });
 * const answer = await app.query("What is RAG?");
 * ```
 */
export function createTruApp<
  TApp extends Record<string, any>,
  TMethod extends keyof TApp & string,
>(target: TApp, opts: TruAppOptions<TApp, TMethod>): TApp {
  return new Proxy(target, {
    get(obj, prop, receiver) {
      if (prop === opts.mainMethod) {
        return (...args: Parameters<TApp[TMethod]>) => {
          const { mainInput, recordOptions } = opts;
          const input =
            mainInput !== undefined
              ? mainInput(...args)
              : JSON.stringify(args);

          return withRecord(
            () => (obj[opts.mainMethod] as AnyFunction).apply(obj, args),
            { ...recordOptions, input }
          );
        };
      }

      return Reflect.get(obj, prop, receiver);
    },
  });
}
