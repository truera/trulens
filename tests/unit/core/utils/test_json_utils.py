import hashlib

from trulens.core.utils.json import _recursive_hash


def test_recursive_hash():
    hash_fn = lambda hash_content: hashlib.md5(
        hash_content.encode("utf-8")
    ).hexdigest()

    values = ["hello", 5, 3.4, True, None]
    values_dict = {
        "str": "hello",
        "int": 5,
        "float": 3.4,
        "bool": True,
        "none": None,
        "list": values,
    }
    hash_dict = {}
    hash_list = []
    for key, value in values_dict.items():
        hash_value = hash_fn(str(value))
        hash_list.append(hash_value)
        hash_dict[key] = hash_value
    hash_dict["list"] = hash_fn("".join(hash_list))

    agg_str = ""
    for key, value in hash_dict.items():
        agg_str += f"{key}:{value},"
        assert _recursive_hash(value) == hash_dict[key]

    assert _recursive_hash(values_dict) == hash_fn(agg_str)
