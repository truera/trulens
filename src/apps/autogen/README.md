# trulens-apps-autogen

Instrumentation for [AutoGen (AG2)](https://github.com/ag2ai/ag2) multi-agent
conversations.

```bash
pip install trulens-apps-autogen
```

```python
from trulens.apps.autogen import TruAutoGen

tru_recorder = TruAutoGen(user_proxy, app_name="debate", app_version="v1")

with tru_recorder as recording:
    user_proxy.initiate_chat(assistant, message="Why is the sky blue?")
```

Refer to the `autogen_group_chat_quality.ipynb` notebook for a worked example.
