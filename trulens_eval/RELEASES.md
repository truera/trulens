# Releases

Releases are organized in `<major>.<minor>.<patch>` style. A release is made
about every week around tuesday-thursday. Releases increment the `minor` version
number. Occasionally bug-fix releases occur after a weekly release. Those
increment only the `patch` number. No releases have yet made a `major` version
increment. Those are expected to be major releases that introduce large number
of breaking changes.

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
