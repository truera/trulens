/**
 * Deterministic app ID computation compatible with the Python TruLens SDK.
 *
 * Ports `_recursive_hash` and `obj_id_of_obj` from
 * `trulens.core.utils.json` so that the same (app_name, app_version)
 * pair produces an identical `app_id` in both languages.
 */

import { createHash } from "node:crypto";

type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

function recursiveHash(value: JsonValue): string {
  if (Array.isArray(value)) {
    const hashes = value.map((v) => recursiveHash(v));
    return recursiveHash(hashes.sort().join(""));
  }
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value).sort();
    let acc = "";
    for (const k of keys) {
      acc += `${k}:${recursiveHash(value[k] as JsonValue)},`;
    }
    return recursiveHash(acc);
  }
  return createHash("md5").update(String(value), "utf-8").digest("hex");
}

/**
 * Compute a deterministic app ID from `app_name` and `app_version`,
 * matching Python's `AppDefinition._compute_app_id`.
 */
export function computeAppId(appName: string, appVersion: string): string {
  const hash = recursiveHash({
    app_name: appName,
    app_version: appVersion,
  });
  return `app_hash_${hash}`;
}
