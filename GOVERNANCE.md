# Governance

## Overview

TruLens is an open source project maintained by Snowflake's AI Observability
team. Development is led by a small core team, with contributions welcome from
the community.

Review and merge happen in this repository, in public. There is no internal
pipeline that mirrors changes in from elsewhere, which is what makes it possible
for contributors outside Snowflake to hold real responsibility here.

## Roles

Roles, the requirements for each, and the process for moving between them are
defined in the [Contributor Ladder](./CONTRIBUTOR_LADDER.md):

| Rung | Write access | Scope |
| ---- | ------------ | ----- |
| Contributor | None | — |
| Area Triager | None (GitHub Triage role) | Issues and PRs in one area |
| Area Reviewer | Merge, scoped to paths | One named area |
| Maintainer | Merge, project-wide | Whole project |

The current roster is in [MAINTAINERS.md](./MAINTAINERS.md). Area Reviewer
scopes are authoritative in [`.github/CODEOWNERS`](./.github/CODEOWNERS).

Roles are held by individuals, not by or through their employers.

## Project Lead

Josh Reini ([@joshreini1](https://github.com/joshreini1)) serves as the project
lead and primary decision-maker for TruLens. This includes decisions on roadmap,
architecture, release cadence, and accepting contributions.

## Decision Making

Day-to-day decisions are made by the project lead. For larger changes (new
features, breaking API changes, architectural shifts), the process is:

1. Open a GitHub issue or discussion describing the proposal.
2. Gather feedback from maintainers and the community.
3. The project lead makes the final call, documenting the rationale in the issue.

Nominations to the contributor ladder follow the process in
[CONTRIBUTOR_LADDER.md](./CONTRIBUTOR_LADDER.md) rather than this section.

## Releases

Release authority, and the credentials it requires — PyPI publish rights,
signing keys, CI secrets — belong to Maintainers only. Area Reviewers have merge
rights within their area and no release credentials. Release policy is in
[POLICIES.md](./POLICIES.md).

## Contributing

Contributions are welcome via pull requests. All PRs are reviewed by one or more
maintainers, or by the Area Reviewer for the paths they touch, before merging.
See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and guidelines.

## Evolving Governance

This model reflects the current state of the project. As the contributor
community grows we expect to adopt more formal processes, such as shared
decision-making among maintainers or a steering committee. Changes to this
document are proposed via pull request and approved by the project lead.

---

Adapted from [GitHub's Minimum Viable
Governance](https://github.com/github/MVG). Licensed under [CC-BY
4.0](https://creativecommons.org/licenses/by/4.0/).
