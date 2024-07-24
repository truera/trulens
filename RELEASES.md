# Releases

Releases are organized in `<major>.<minor>.<patch>` style. A release is made
about every week around tuesday-thursday. Releases increment the `minor` version
number. Occasionally bug-fix releases occur after a weekly release. Those
increment only the `patch` number. No releases have yet made a `major` version
increment. Those are expected to be major releases that introduce large number
of breaking changes.

## 0.33.0

### What's Changed
* timeouts for wait_for_feedback_results by @sfc-gh-pmardziel in https://github.com/truera/trulens/pull/1267
* TruLens Streamlit components by @sfc-gh-jreini in https://github.com/truera/trulens/pull/1224
* Run the dashboard on an unused port by default by @sfc-gh-jreini in https://github.com/truera/trulens/pull/1280 and @sfc-gh-jreini in https://github.com/truera/trulens/pull/1275

### Documentation Updates
* Reflect Snowflake SQLAlchemy Release in "Connect to Snowflake" Docs by @sfc-gh-jreini in https://github.com/truera/trulens/pull/1281
* Update guardrails examples by @sfc-gh-jreini in https://github.com/truera/trulens/pull/1275

### Bug Fixes
* Remove duplicated tests by @sfc-gh-dkurokawa in https://github.com/truera/trulens/pull/1283
* fix LlamaIndex streaming response import by @sfc-gh-chu in https://github.com/truera/trulens/pull/1276

## 0.32.0

### What's Changed
* Context filtering guardrails by @sfc-gh-jreini in https://github.com/truera/trulens/pull/1192
* Query optimizations for TruLens dashboard resulting in 4-32x benchmarked speedups by @sfc-gh-chu in https://github.com/truera/trulens/pull/1216
* Logging in Snowflake database by @sfc-gh-chu in https://github.com/truera/trulens/pull/1216
* Snowflake Cortex feedback provider by @sfc-gh-dhuang in https://github.com/truera/trulens/pull/1202
* improve langchain prompting using native messages by @nicoloboschi in https://github.com/truera/trulens/pull/1194
* fix groundedness with no supporting evidence by @nicoloboschi in https://github.com/truera/trulens/pull/1193
* Improve Microsecond support by @sfc-gh-gtokernliang in https://github.com/truera/trulens/pull/1195
* SkipEval exception by @sfc-gh-pmardziel in https://github.com/truera/trulens/pull/1200
* Update pull_request_template.md by @sfc-gh-jreini in https://github.com/truera/trulens/pull/1234
* Use rounding instead of flooring in feedback score extraction by @sfc-gh-dhuang in https://github.com/truera/trulens/pull/1244

### Documentation
* Benchmarking Snowflake arctic-instruct feedback function of groundedness by @sfc-gh-dhuang in https://github.com/truera/trulens/pull/1185
* Evaluation Benchmarks Page by @sfc-gh-jreini in https://github.com/truera/trulens/pull/1190
* Documentation for snowflake sqlalchemy implementation by @sfc-gh-chu in https://github.com/truera/trulens/pull/1216*
* Documentation for logging in snowflake database by @sfc-gh-chu in https://github.com/truera/trulens/pull/1216
* Documentation for cortex provider by @sfc-gh-dhuang in https://github.com/truera/trulens/pull/1202

### Examples
* Context filtering guardrails added to quickstarts by @sfc-gh-jreini in https://github.com/truera/trulens/pull/1192
* Update Arctic model notebook to use new Cortex provider by @sfc-gh-dhuang in https://github.com/truera/trulens/pull/1202
* New example showing cortex finetuning by @sfc-gh-dhuang in https://github.com/truera/trulens/pull/1202
* show how to add cost/latency/usage details in virtual records by @sfc-gh-jreini in https://github.com/truera/trulens/pull/1197

### Bug Fixes
* Enable formatting during PR build. Also format code that wasn't formatted. by @sfc-gh-dkurokawa in https://github.com/truera/trulens/pull/1212
* Fix test cases generation - normalization step for SummEval score by @sfc-gh-dhuang in https://github.com/truera/trulens/pull/1217
* Enable regex to extract floats in score generation by @sfc-gh-dhuang in https://github.com/truera/trulens/pull/1223
* Fix cost tracking in OpenAI and LiteLLM endpoints by @sfc-gh-dhuang in https://github.com/truera/trulens/pull/1228
* remove deprecated legacy caching by @sfc-gh-jreini in https://github.com/truera/trulens/pull/1233
* Remove remaining streamlit legacy caching by @JushBJJ in https://github.com/truera/trulens/pull/1246

## 0.31.0

### What's Changed

* Parallelize groundedness LLM calls for speedup by @sfc-gh-dhuang in https://github.com/truera/trulens/pull/1180
* Option for quieter deferred evaluation by @epinzur in https://github.com/truera/trulens/pull/1178
* Support for langchain >=0.2.x retrievers via instrumenting the `invoke` method by @nicoloboschi in https://github.com/truera/trulens/pull/1187


### Examples
* ❄️ Snowflake Arctic quickstart by @joshreini1 in https://github.com/truera/trulens/pull/1156

### Bug fixes
* Fix a few more old groundedness references + llamaindex agent toolspec import by @daniel-huang-1230 in https://github.com/truera/trulens/pull/1161
* Very minor fix of print statement by @sfc-gh-dhuang in https://github.com/truera/trulens/pull/1173
* Fix sidebar logo formatting by @sfc-gh-chu in https://github.com/truera/trulens/pull/1169\
* [bugfix] prevent stack overflow in jsonify by @piotrm0 in https://github.com/truera/trulens/pull/1176

**Full Changelog**: https://github.com/truera/trulens/compare/trulens-eval-0.30.1...trulens-eval-0.31.0
## 0.30.1

### What's Changed
* update comprehensiveness by @daniel-huang-1230 and @joshreini1 in https://github.com/truera/trulens/pull/1064
* glossary additions by @piotrm0 in https://github.com/truera/trulens/pull/1144

### Bug Fixes
* Add langchain-community to optional requirements  by @joshreini1 in https://github.com/truera/trulens/pull/1146
* Checks for use of openai endpoint by @piotrm0 in https://github.com/truera/trulens/pull/1154

**Full Changelog**: https://github.com/truera/trulens/compare/trulens-eval-0.29.0...trulens-eval-0.30.1
## 0.29.0

## Breaking Changes
In this release, we re-aligned the groundedness feedback function with other LLM-based feedback functions. It's now faster and easier to define a groundedness feedback function, and can be done with a standard LLM provider rather than importing groundedness on its own. In addition, the custom groundedness aggregation required is now done by default.

Before:
```python
from trulens.feedback.provider.openai import OpenAI
from trulens.feedback import Groundedness

provider = OpenAI() # or any other LLM-based provider
grounded = Groundedness(groundedness_provider=provider)
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)
```

After:
```python
provider = OpenAI()
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
)
```

This change also applies to the NLI-based groundedness feedback function available from the Huggingface provider.

Before:
```python
from trulens.feedback.provider.openai import Huggingface
from trulens.feedback import Groundedness

from trulens.feedback.provider import Huggingface
huggingface_provider = Huggingface()
grounded = Groundedness(groundedness_provider=huggingface_provider)

f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)
```

After:
```python
from trulens.feedback import Feedback
from trulens.feedback.provider.hugs = Huggingface

huggingface_provider = Huggingface()

f_groundedness = (
    Feedback(huggingface_provider.groundedness_measure_with_nli, name = "Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
)
```

In addition to the change described above, below you can find the full release description.

## What's Changed
* update groundedness prompt by @bpmcgough in https://github.com/truera/trulens/pull/1112
* Default names for rag triad utility by @joshreini1 in https://github.com/truera/trulens/pull/1122
* Unify groundedness interface by @joshreini1 in https://github.com/truera/trulens/pull/1135

## Bug Fixes
* Fixed bug with trace view initialization when no feedback functions exist by @walnutdust in https://github.com/truera/trulens/pull/1108
* Remove references to running moderation endpoint on AzureOpenAI by @joshreini1 in https://github.com/truera/trulens/pull/1116
* swap rag utility (qs)relevance by @piotrm0 in https://github.com/truera/trulens/pull/1120
* Fix Link in Readme by @timbmg in https://github.com/truera/trulens/pull/1128
* chore: remove unused code cell by @stokedout in https://github.com/truera/trulens/pull/1113
* trurails: update to getattr by @joshreini1 in https://github.com/truera/trulens/pull/1130
* Fix typo in README.md by @eltociear in https://github.com/truera/trulens/pull/1136
* fix rag triad and awaitable calls by @piotrm0 in https://github.com/truera/trulens/pull/1110
* Remove placeholder feedback for asynchronous responses by @arn-tru in https://github.com/truera/trulens/pull/1127
* Stop iteration streams in openai cost tracking by @piotrm0 in https://github.com/truera/trulens/pull/1138

## Examples
* Show OSS models (and tracking) in LiteLLM application by @joshreini1 in https://github.com/truera/trulens/pull/1109

## New Contributors
* @stokedout made their first contribution in https://github.com/truera/trulens/pull/1113
* @timbmg made their first contribution in https://github.com/truera/trulens/pull/1128
* @bpmcgough made their first contribution in https://github.com/truera/trulens/pull/1112
* @eltociear made their first contribution in https://github.com/truera/trulens/pull/1136

**Full Changelog**: https://github.com/truera/trulens/compare/trulens-eval-0.28.0...trulens-eval-0.29.0
## 0.28.1

### Bug fixes

* Fix for missing `alembic.ini` in package build.

## 0.28.0

### What's Changed

* Meta-eval / feedback functions benchmarking notebooks, ranking-based eval
  utils, and docs update by @daniel-huang-1230 in
  https://github.com/truera/trulens/pull/991
* App delete functionality added by @arn-tru in
  https://github.com/truera/trulens/pull/1061
* Added test coverage to langchain provider by @arn-tru in
  https://github.com/truera/trulens/pull/1062
* Configurable table prefix by @piotrm0 in
  https://github.com/truera/trulens/pull/971
* Add example systemd service file by @piotrm0 in
  https://github.com/truera/trulens/pull/1072

### Bug fixes

* Queue fixed for python version lower than 3.9 by @arn-tru in
  https://github.com/truera/trulens/pull/1066
* Fix test-tru by @piotrm0 in https://github.com/truera/trulens/pull/1070
* Removed broken tests by @arn-tru in
  https://github.com/truera/trulens/pull/1076
* Fix legacy db missing abstract method by @piotrm0 in
  https://github.com/truera/trulens/pull/1077
* Release test fixes by @piotrm0 in https://github.com/truera/trulens/pull/1078
* Docs fixes by @piotrm0 in https://github.com/truera/trulens/pull/1075

### Examples

* MongoDB Atlas quickstart by @joshreini1 in
  https://github.com/truera/trulens/pull/1056
* OpenAI Assistants API (quickstart) by @joshreini1 in
  https://github.com/truera/trulens/pull/1041

**Full Changelog**:
https://github.com/truera/trulens/compare/trulens-eval-0.27.2...trulens-eval-0.28.0
