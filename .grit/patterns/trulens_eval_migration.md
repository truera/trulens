---
title: Upgrade to TruLens v1.x
tags: [python, trulens, migration]
---

# Upgrade to TruLens v1.x

This pattern upgrades code to TruLens v1.x by replacing deprecated imports and class usage.

```grit
engine marzano(0.1)
language python

any {
  `Tru($session)` => `TruSession($session)`,
  `from trulens_eval import Tru` => `from trulens.core import TruSession`,
  `from trulens_eval import Feedback` => `from trulens.core import Feedback`
}
