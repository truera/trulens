# trulens-providers-api-route

Use [API Route](https://www.api-route.com) models for _TruLens_ text feedback
evaluation through the existing _OpenAI_ compatible provider implementation.

```shell
pip install trulens-providers-api-route
```

Set `API_ROUTE_API_KEY` to a key created in
[API Keys](https://www.api-route.com/api-keys). Inference requires account credit.
The default endpoint is `https://global.api-route.com/v1`.

```python
from trulens.providers.api_route import provider as api_route_provider

provider = api_route_provider.APIRoute(model_engine="gpt-6.1-sol")
score = provider.relevance("What is a cow?", "A cow is an animal.")
```

Choose a model ID available to your key using the authenticated `/v1/models`
endpoint. IDs are passed unchanged, without an additional provider prefix.
Explicit `api_key` and `base_url` arguments override `API_ROUTE_API_KEY` and
`API_ROUTE_BASE_URL`, respectively. Structured output and request options depend
on the selected model; the inherited capability fallback handles unsupported
options.

Moderation feedback is not supported by this integration. `reports_costs` is
false because _OpenAI_ price estimates are not API Route billing prices;
OpenAI endpoint instrumentation can still track token usage.

Run the package's offline tests from the repository root:

```shell
pytest src/providers/api_route/tests
```
