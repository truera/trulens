from typing import Any, Dict, Union


def flatten_metadata(metadata: dict):
    results = {}
    for k, v in metadata.items():
        if isinstance(v, dict):
            for k2, v2 in flatten_metadata(v).items():
                results[f"{k}.{k2}"] = v2
        else:
            results[k] = str(v)
    return results


def nest_dict(d: Dict[str, Any]):
    result = {}
    for key, value in d.items():
        nested_subdict = nest_metadata(key, value)
        nested_update(result, nested_subdict)
    return result


def nest_metadata(
    key: str, value: Union[Dict[str, Any], Any]
) -> Dict[str, Any]:
    metadata_keys = key.split(".")
    root_dict = {}
    ptr = root_dict
    for i, key in enumerate(metadata_keys):
        if i == len(metadata_keys) - 1:
            ptr[key] = value
        else:
            ptr[key] = {}
            ptr = ptr[key]
    return root_dict


def nested_update(metadata: dict, update: dict):
    for k, v in update.items():
        if isinstance(v, dict) and k in metadata:
            nested_update(metadata[k], v)
        else:
            metadata[k] = v
