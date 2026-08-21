# Contributor Ladder

TruLens is maintained by Snowflake's AI Observability team. This document
describes how contributors outside Snowflake take on responsibility in the
project.

The criteria are published so you can see what the path is instead of guessing.
They are not a formula:

- **Meeting the requirements makes you eligible for nomination, not entitled to
  promotion.** The numbers are a floor. Judgment, care, and how you work with
  other people count for more than any of them.
- **We are biased toward yes.** Nominations are read as "do I have a reason to
  say no," not "give me a reason to say yes." An objection should come with
  specific next steps the candidate can act on.

## The rungs

| Rung | Write access | Scope |
| ---- | ------------ | ----- |
| Contributor | None | — |
| [Area Triager](#area-triager) | None (GitHub Triage role) | Issues and PRs in one area |
| [Area Reviewer](#area-reviewer) | Merge, scoped to paths | One named area |
| [Maintainer](#maintainer) | Merge, project-wide | Whole project |

There is no requirement to move up. Most contributors won't, and that's fine.

## Sponsors

Promotions need sponsors. A sponsor is someone already at or above the rung in
question who vouches for you publicly, by name, on the nomination.

Sponsorship means having worked with you directly — reviewed your code, argued
about a design, coordinated on an issue. Someone who hasn't done that can't
sponsor you, however well regarded you are. A sponsor is accountable for the
judgment.

You don't arrange your own sponsors. A Maintainer or Area Reviewer opens the
nomination and finds them.

Sponsors for Area Reviewer and above must not all share an employer with the
candidate, so that no single company can promote its own staff unopposed. Every
maintainer is currently a Snowflake employee, so this rule has no effect yet; it
starts to bind once someone outside Snowflake reaches Area Reviewer.

---

## Area Triager

Area Triagers keep the issue tracker and PR queue in order, and are the first
reviewers for pull requests in their area. No write access to the codebase.

Your area is named when you're nominated and recorded in
[`.github/area-reviewers.yml`](./.github/area-reviewers.yml). It determines what
gets routed to you. GitHub's Triage permission is repository-wide and cannot be
restricted to paths, so you can label and close issues anywhere; the area is
where you're asked to concentrate.

### Responsibilities

- Triage incoming issues in your area: reproduce what you can, ask for missing
  detail, apply labels, close duplicates and resolved issues.
- Review pull requests in your area. Comment reviews are enough: ask questions,
  point out what looks wrong, say what you checked.
- Point contributors at the right context: a related issue, the relevant docs, a
  prior PR.
- Follow the [Code of Conduct](./CONTRIBUTING.md) and help others meet it.

### Requirements

- A few merged pull requests — **about three** — or a comparable record of triage,
  documentation, or community support, over **at least a month**. Different shapes
  of history qualify: a run of small PRs, one or two that took real back-and-forth,
  or sustained help in issues. Maintenance and documentation count the same as
  features.
- **One sponsor** who is an Area Reviewer or Maintainer.

Someone reads the work itself. The counts only decide who gets looked at.

### Privileges

- GitHub **Triage** role: label, assign, close and reopen issues and PRs.
- Listed with your area under Area Triagers in
  [MAINTAINERS.md](./MAINTAINERS.md).
- Automatic review requests on PRs touching your area, routed by
  `area-reviewers.yml`.

On reviews specifically: you can leave comment reviews, and because this is a
public repository GitHub will also let you press Approve or Request changes. Those
buttons don't carry merge authority here — `main` requires an approving review
from a code owner, and GitHub ignores CODEOWNERS entries for accounts without
write access. So an Area Triager's review informs the decision without gating it,
whichever button you press.

Reviewing is also how you reach the next rung, which asks for substantive
reviews on five pull requests written by other people.

`src/core` is routed to all Area Triagers rather than to one owner. Everything else
in the project depends on it, so reading core changes is the fastest way to build
the context the next rung asks for.

### Process

1. A Maintainer or Area Reviewer opens an issue titled
   `Area Triager nomination: <github-handle>`, naming the proposed area and
   linking representative work.
2. The nominee comments to confirm they want the role, accept the
   responsibilities above, agree the area is the right one, and state their
   affiliation for [MAINTAINERS.md](./MAINTAINERS.md).
3. Open at least **72 hours** for comment. It passes if no Maintainer objects.
4. A Maintainer grants the Triage role, then opens a PR adding the handle and
   area to `MAINTAINERS.md` and `area-reviewers.yml`.

---

## Area Reviewer

The first rung with merge rights, **scoped to specific paths** in
[`.github/CODEOWNERS`](./.github/CODEOWNERS) rather than the whole repository.

### Responsibilities

- Review PRs touching your area and merge the ones that are ready.
- Keep your area healthy: its tests, its documentation, its deprecations.
- Exercise judgment for the good of the project, independently of your employer.
- Review newer contributors' work in your area.
- Tell a Maintainer if you need to step back.

### Requirements

- Area Triager for at least **1 month**.
- **Substantive reviews on at least 5 pull requests written by other people.**
  Substantive means engaging with the change: asking what it does, finding a
  problem, or approving with a note on what you checked. A bare "LGTM" doesn't
  count, and neither do comments on your own PRs.
- Activity spread over **at least 3 months**, rather than concentrated in a single
  burst.
- Depth in **one named area**, with a track record that includes maintenance —
  fixing, deprecating, following up — and not only new features.
- **Two sponsors** who are Area Reviewers or Maintainers. See
  [Sponsors](#sponsors).

### Privileges

- Merge rights scoped to your area's paths in `CODEOWNERS`. Your approval
  satisfies the review requirement for PRs in that area.
- Automatic review requests for PRs touching your paths.
- A vote on nominations to Area Triager and Area Reviewer.
- Listed with your area in `MAINTAINERS.md`.

Scoped merge rights do not reach publishing. Releases are built and uploaded by
Azure Pipelines, triggered when a version bump in `src/core/pyproject.toml` lands
on `main`, and the PyPI token is a secret variable in that project rather than a
GitHub permission. `src/core/` is owned by maintainers in
[`CODEOWNERS`](./.github/CODEOWNERS), and the release pipeline never runs on a
pull request.

### Process

1. A Maintainer or Area Reviewer opens a PR adding the candidate to `CODEOWNERS`
   and `MAINTAINERS.md`, naming the area and linking evidence for each
   requirement.
2. The candidate comments to confirm they accept the responsibilities and to state
   their affiliation, along with any involvement in a competing project.
3. The second sponsor comments `+1`.
4. Open at least **7 days**. It passes on approval by a majority of Maintainers,
   with no sustained objection from a Maintainer or from an Area Reviewer whose
   area overlaps.
5. A Maintainer grants scoped write access and merges the PR.

If a nomination doesn't pass, a Maintainer other than the proposer works with the
candidate to supply the missing evidence or to mentor toward it. A stalled
nomination should produce a plan, not silence.

---

## Maintainer

Project-wide responsibility, including releases and direction.

### Responsibilities

- Review and merge across the project, including paths no Area Reviewer covers.
- Set roadmap and architectural direction; decide on breaking changes.
- Run releases.
- Nominate and mentor people toward the next rung.
- Keep this document accurate, and amend it when it stops matching reality.

### Requirements

- Area Reviewer for at least **6 months**.
- Breadth across several areas rather than depth in one.
- **Substantive reviews on at least 30 pull requests** written by other people,
  and named as a requested reviewer on at least **10** of them.
- Has sponsored or mentored at least one contributor up a rung.
- Can exercise judgment for the good of the project, independent of their
  employer, friends, or team.

### Privileges

- Project-wide merge rights.
- A vote on releases, breaking changes, roadmap, and governance changes.
- Release authority: the Azure Pipelines project that builds and uploads
  releases, and the PyPI token it holds.
- Authority to represent the project publicly.

### Process

1. A Maintainer opens a PR moving the candidate to Maintainers in
   `MAINTAINERS.md`, with the evidence.
2. The candidate comments to confirm they accept the responsibilities.
3. Open at least **7 days**. It passes on approval by a majority of Maintainers
   with no sustained objection.

Maintainers discuss nominations privately before opening the PR, so that a
candidate can decline, or be declined, without a public record.

---

## Holding a role

### Independence

Roles are held by **individuals, not by or through their employers**. Changing
jobs doesn't affect your standing. Leaving Snowflake doesn't remove a role, and
joining Snowflake doesn't confer one.

Snowflake currently employs every Maintainer, and we want that to change. As a
goal rather than a hard rule at our size: we will avoid nominating Maintainers
from an organization that already employs half or more of them, and we will grow
the non-Snowflake share of Area Reviewers deliberately.

### Inactivity and Emeritus

After **12 months** with no contribution, an Area Triager or Area Reviewer moves to
Emeritus and their access is removed. Unused permissions are a liability, and an
inaccurate roster misleads contributors about who is available.

Merit earned doesn't expire. To come back, open an issue or email a Maintainer.
No re-application.

Two exceptions:

- **Activity metrics miss non-code work.** If triage, documentation, or community
  work went uncounted and you were moved to Emeritus in error, say so and we'll
  reverse it.
- **Some areas are legitimately quiet.** A provider integration that needed no
  changes for a year is working, not unmaintained. Maintainers can grant
  exceptions on that basis.

### Stepping down

Tell a Maintainer, or open a PR against `MAINTAINERS.md`. You'll be listed as
Emeritus and you're welcome back at the same rung whenever you want it.

---

## Changing this document

Amendments are proposed by PR, open at least **7 days**, and pass on approval by
a majority of Maintainers.

---

Structure adapted from the [CNCF contributor ladder
template](https://github.com/cncf/project-template/blob/main/CONTRIBUTOR_LADDER.md),
with criteria drawn from
[Kubernetes](https://github.com/kubernetes/community/blob/main/community-membership.md),
[Prometheus](https://github.com/prometheus/governance/blob/main/ROLES.md),
[Node.js](https://github.com/nodejs/node/blob/main/GOVERNANCE.md), and the
[Apache Software Foundation](https://community.apache.org/newcommitter.html).
Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
