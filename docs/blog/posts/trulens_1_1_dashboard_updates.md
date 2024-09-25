---
categories:
  - General
date: 2024-09-25
---

# What's new in TruLens 1.1: Dashboard Comparison View, Multi-App Support, Metadata Editing, and More!

TruLens 1.1.0 has been released! This release includes a number of improvements to the TruLens dashboard, including a new comparison view and a more intuitive user interface. We have also made several improvements performance and usability.

<!-- more -->

## Dashboard Highlights

An overhaul of the TruLens dashboard has been released with major features and improvements. Here are some of the highlights:

### Global Enhancements

![The New TruLens Dashboard](../assets/dashboard_global_features.gif)

#### Global app selector

TruLens 1.0 introduced app versioning, allowing performance of their LLM apps to be tracked across different versions. On multi-app tables, the dashboard sidebar now includes an app selector to quickly navigate to the desired application.

#### App version and Record search and filtering

All pages in the dashboard now include relevant search and filter options to identify app versions and records quickly. The search bar allows filtering records and app versions by name or by other metadata fields. This makes it easy to find specific records or applications and compare their performance over time.

#### Performance enhancements

TruLens 1.1.0 includes several performance enhancements to improve the scalability and speed of the dashboard. The dashboard now queries only the most recent records unless specified otherwise. This helps prevent out-of-memory errors and improves the overall performance of the dashboard.

Furthermore, all record and app data is now cached locally, reducing network latency on refreshes. This results in faster load times and a more responsive user experience. The cache is cleared automatically every 15 minutes or manually with the new `Refresh Data` button.

### Leaderboard

![Leaderboard enhancements](../assets/leaderboard_metadata.gif)

The leaderboard is now displayed in a tabular format, with each row representing a different application version. The grid data can be sorted and filtered.

#### App Version Pinning

App versions can now be pinned to the top of the leaderboard for easy access. This makes it easy to track the performance of specific versions over time. Pinned versions are highlighted for easy identification and can be filtered to with a toggle.

#### Metadata Editing

To better identify and track application versions, app metadata visibility is a central part of this leaderboard update. In addition to being displayed on the leaderboard, metadata fields are now editable after ingestion by double-clicking the cell, or bulk selecting and choosing the `Add/Edit Metadata` option. In addition, new fields can be added with the `Add/Edit Metadata` button.

A selector at the top of the leaderboard allows toggling which app metadata fields are displayed to better customize the view.

#### Virtual App Creation

To bring in evaluation data from a non-TruLens app (e.g another runtime environment or benchmark by a third-party source), the `Add Virtual App` button has been added to the leaderboard! This creates a virtual app with user-defined metadata fields and evaluation data that can be used in the leaderboard and comparison view.

### Comparison View

This update introduces a brand-new comparison page that enables the comparison of up to 5 different app versions side by side.

#### App-level comparison

![App-level comparison](../assets/compare_app.png)

The comparison view allows performance comparisons across different app versions side by side. The aggregate feedback function results for each app version is plotted across each of the shared feedback functions, making it easy to see how the performance  has changed.

#### Record-level comparison

![Record-level comparison](../assets/compare_record.png)

To deep dive into the performance of individual records, the comparison view also allows comparison of overlapping records side by side. The dashboard computes a diff or variance score (depending on the number of apps compared against) to identify interesting or anomalous records which have the most significant performance differences. In addition to viewing the distribution of feedback scores, this page also displays the trace data of each record side by side.

### Records Page

![Records Page Flow](../assets/record_page.gif)

The records page has been updated to include a more intuitive flow for viewing and comparing records. The page now includes a search bar to quickly find specific records as well as matching app metadata filters.

#### Additional features

- URL serialization of key dashboard states
- Dark mode
- Improved error handling
- Fragmented rendering


#### Try it out!

We hope you enjoy the new features and improvements in TruLens 1.1.0! To get started, use [`run_dashboard`][trulens.dashboard.run.run_dashboard] with a TruSession object:


```python
from trulens.core import TruSession
from trulens.dashboard import run_dashboard

session = TruSession(...)
run_dashboard(session)
```
