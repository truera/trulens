import pytest
from trulens.core.utils.json import _recursive_hash


@pytest.mark.parametrize(
    "test_input,matching_input,expected",
    [
        ("", None, STR_HASH := "d41d8cd98f00b204e9800998ecf8427e"),
        ("hello", None, STR_HASH := "5d41402abc4b2a76b9719d911017c592"),
        (5, "5", INT_HASH := "e4da3b7fbbce2345d7772b0674a318d5"),
        (3.4, "3.4", FLOAT_HASH := "31053ad0506e935470ca21b43cae98cf"),
        (True, "True", BOOL_HASH := "f827cf462f62848df37c5e1e94a4da74"),
        (None, "None", NONE_HASH := "6adf97f83acf6453d4a6a4b1070f3754"),
        (
            ["hello", 5, 3.4, True, None],
            "".join(
                sorted(
                    [
                        STR_HASH,
                        INT_HASH,
                        FLOAT_HASH,
                        BOOL_HASH,
                        NONE_HASH,
                    ]
                )
            ),
            LIST_HASH := "cbbb36ccd53eee57843e0bc04fe78c6c",
        ),
        (
            {
                "str": "hello",
                "int": 5,
                "float": 3.4,
                "bool": True,
                "none": None,
                "list": ["hello", 5, 3.4, True, None],
            },
            f"bool:{BOOL_HASH},float:{FLOAT_HASH},int:{INT_HASH},list:{LIST_HASH},none:{NONE_HASH},str:{STR_HASH},",
            "0f7ee6af37886286ac9b8384254a557d",
        ),
    ],
)
def test_recursive_hash(test_input, matching_input, expected):
    result = _recursive_hash(test_input)

    if matching_input is not None:
        matching_result = _recursive_hash(matching_input)
        assert (
            result == matching_result
        ), f"Failed on {test_input}: Hashes for {test_input} ({result}) should match {matching_input} ({matching_result})"

    assert (
        result == expected
    ), f"Failed on {test_input}: got {result}, expected {expected}"
