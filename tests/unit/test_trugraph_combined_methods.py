import pytest


def _materialize_methods(default):
    """Helper that mirrors TruGraph CombinedInstrument METHODS handling."""
    try:
        lc_methods = (
            default.METHODS()
            if callable(getattr(default, "METHODS", None))
            else default.METHODS
        )
    except Exception:
        lc_methods = []
    return list(lc_methods or [])


@pytest.mark.skipif(
    pytest.importorskip("langgraph", reason="langgraph not installed") is None,
    reason="langgraph not installed",
)
def test_combined_methods_handles_callable_and_list(monkeypatch):
    # Import after ensuring dependency
    from trulens.apps.langchain.tru_chain import LangChainInstrument
    from trulens.apps.langgraph.tru_graph import LangGraphInstrument

    # Snapshot original to restore later
    original_lc_methods = LangChainInstrument.Default.METHODS
    original_lg_methods = LangGraphInstrument.Default.METHODS
    try:
        # Case 1: both callable
        lc_1 = _materialize_methods(LangChainInstrument.Default)
        lg_1 = _materialize_methods(LangGraphInstrument.Default)
        assert isinstance(lc_1, list)
        assert isinstance(lg_1, list)

        # Case 2: LangChain METHODS is a list
        fake_list = ["dummy_method_entry"]
        monkeypatch.setattr(
            LangChainInstrument.Default, "METHODS", fake_list, raising=False
        )
        lc_2 = _materialize_methods(LangChainInstrument.Default)
        assert lc_2 == fake_list

        # Case 3: LangGraph METHODS is a list
        monkeypatch.setattr(
            LangGraphInstrument.Default, "METHODS", fake_list, raising=False
        )
        lg_2 = _materialize_methods(LangGraphInstrument.Default)
        assert lg_2 == fake_list

        # Combined should concatenate without TypeError
        combined = lc_2 + lg_2
        assert isinstance(combined, list)
        assert combined == fake_list + fake_list
    finally:
        # Restore
        monkeypatch.setattr(
            LangChainInstrument.Default,
            "METHODS",
            original_lc_methods,
            raising=False,
        )
        monkeypatch.setattr(
            LangGraphInstrument.Default,
            "METHODS",
            original_lg_methods,
            raising=False,
        )
