# Contributor Ladder

TruLens is maintained by Snowflake's AI Observability team, and we want a real,
documented path for contributors outside Snowflake to take on responsibility in
the project. This document describes that path.

We publish the criteria because we think you should be able to see what the path
is, rather than feeling it is at the discretion of a private conversation. Two
things follow from that:

- **Meeting the requirements makes you eligible for nomination, not entitled to
  promotion.** The requirements are a floor for legibility, never a sufficient
  condition. Judgment, care, and how you work with other people matter more than
  any count.
- **We are biased toward saying yes.** Nominations are reviewed in the spirit of
  "I have a reason to say no," not "give me a reason to say yes." Anyone
  objecting to a nomination is expected to pair the objection with concrete,
  unambiguous next steps the candidate can take.

Adding people to this ladder is an investment in the future of the project, not
a reward for past work. Almost everything here is reversible: we use revision
control, and roles can be handed back without stigma.

## The rungs

| Rung | Write access | Scope |
| ---- | ------------ | ----- |
| Contributor | None | — |
| [Triager](#triager) | None (GitHub Triage role) | Issues and pull requests |
| [Area Reviewer](#area-reviewer) | Merge, scoped to paths | One named area |
| [Maintainer](#maintainer) | Merge, project-wide | Whole project |

There is no requirement to move up. Contributing at any rung, indefinitely, is a
completely normal and valued way to be part of this project.

---

## Triager

The on-ramp. Triagers help keep the issue tracker and PR queue healthy. This
role carries no write access to the codebase.

### Responsibilities

- Triage incoming issues: reproduce where you can, ask for the missing detail,
  apply labels, close duplicates and resolved issues.
- Point contributors at the right context — a related issue, the relevant
  section of the docs, a prior PR.
- Follow the [Code of Conduct](./CONTRIBUTING.md) and help others meet it.

### Requirements

- Roughly **1 month** of contribution history, and about **5 merged pull
  requests** — or a comparable record of issue triage, documentation, or
  community support. Several shapes of history qualify: a run of small PRs, a
  couple of complex ones that took real back-and-forth, or sustained help in
  issues and discussions.
- Two-factor authentication enabled on your GitHub account.
- Signed CLA.
- **One sponsor** who is an Area Reviewer or Maintainer.

### Privileges

- GitHub **Triage** role: label, assign, close and reopen issues and PRs.
- Listed under Triagers in [MAINTAINERS.md](./MAINTAINERS.md).
- Automatic review requests on pull requests touching your area, routed by
  [`.github/area-reviewers.yml`](./.github/area-reviewers.yml). Reviews from
  Triagers are advisory — a Maintainer or Area Reviewer still approves and merges
  — but this is where the review track record for the next rung comes from.

`src/core` is routed to all Triagers rather than to one area owner. It is the
largest area in the project and the one with the thinnest review coverage, and
reviewing it is the most direct way to build the experience the next rung asks
for.

### Process

1. A Maintainer or Area Reviewer opens an issue titled
   `Triager nomination: <github-handle>`, linking representative work.
2. The nominee comments confirming they want the role and accept the
   responsibilities above.
3. Open at least **72 hours** for comment. It passes if no Maintainer objects.
4. A Maintainer grants the Triage role and opens a PR adding the handle to
   `MAINTAINERS.md`.

---

## Area Reviewer

The first rung with merge rights. Those rights are **scoped to specific paths**,
listed in [`.github/CODEOWNERS`](./.github/CODEOWNERS) — not to the whole
repository. Scoping is deliberate: it lets us extend real authority in the place
where you have real depth, without asking either side to take on more than the
evidence supports.

### Responsibilities

- Review PRs touching your area, and merge the ones that are ready.
- Own the health of your area: its tests, its documentation, its deprecations.
- Exercise judgment for the good of the project, independently of your employer.
- Mentor contributors working in your area, including reviewing generously.
- Say so if you need to step back. That is expected and fine.

### Requirements

- Triager for at least **1 month**.
- At least **20 substantive contributions** in the past 12 months. Substantive
  contributions include authoring PRs, reviewing PRs, triaging issues, writing
  or restructuring documentation, testing release candidates, and answering
  questions from other users.
- Primary reviewer on at least **5 pull requests**.
- Demonstrated depth in **one named area**, with a track record that includes
  maintenance — fixing, deprecating, and following up — and not only new
  features.
- **Two sponsors** who are Area Reviewers or Maintainers. Once the project has
  any Area Reviewer or Maintainer not employed by Snowflake, at least one
  sponsor must not share an employer with the candidate.

### Privileges

- Merge rights scoped to your area's paths in `CODEOWNERS`. Your approval
  satisfies the review requirement for PRs in that area.
- Automatic review requests for PRs touching your paths.
- A vote on nominations to Triager and Area Reviewer.
- Listed with your area in `MAINTAINERS.md`.

Area Reviewers do **not** hold PyPI publish rights, package signing keys, or CI
secrets. Those attach only to Maintainer.

### Process

1. A Maintainer or Area Reviewer opens a PR adding the candidate to
   `CODEOWNERS` and `MAINTAINERS.md`, naming the area and linking the evidence
   for each requirement.
2. The candidate comments confirming they accept the responsibilities.
3. The second sponsor comments `+1`.
4. Open at least **7 days**. It passes on approval by a majority of Maintainers
   with no sustained objection from a Maintainer or from an Area Reviewer whose
   area overlaps.
5. A Maintainer grants scoped write access and merges the PR.

If a nomination does not pass, a Maintainer who was not the proposer is assigned
to work with the candidate: their job is to supply the missing evidence or to
mentor toward it. A nomination that stalls should produce a development plan,
not silence.

---

## Maintainer

Project-wide responsibility, including releases and direction.

### Responsibilities

- Review and merge across the project; be a fallback reviewer where no Area
  Reviewer covers the path.
- Set roadmap and architectural direction; decide on breaking changes.
- Run releases and hold the credentials that requires.
- Grow the contributor base: nominate people, and mentor them toward the next
  rung.
- Uphold this document, and amend it when it stops matching reality.

### Requirements

- Area Reviewer for at least **6 months**.
- Breadth across multiple areas of the project, not depth in only one.
- At least **10 pull requests as primary reviewer** and **30 reviewed or
  merged**.
- Has mentored at least one contributor up a rung.
- Can exercise judgment for the good of the project, independent of their
  employer, friends, or team.

### Privileges

- Project-wide merge rights.
- A vote on releases, breaking changes, roadmap, and governance changes.
- Release credentials: PyPI publish rights, signing keys, CI secrets.
- Authority to represent the project publicly.

### Process

1. A Maintainer opens a PR moving the candidate to Maintainers in
   `MAINTAINERS.md`, with the evidence.
2. The candidate comments confirming they accept the responsibilities.
3. Open at least **7 days**. It passes on approval by a majority of Maintainers
   with no sustained objection.

Nominations are discussed privately among Maintainers before the public PR, so
that a candidate can decline — or be declined — without a public record.

---

## Holding a role

### Independence

Roles on this ladder are held by **individuals, not by or through their
employers**. Changing jobs does not affect your standing. Leaving Snowflake does
not remove a role, and joining Snowflake does not confer one.

Snowflake employs most current Maintainers. We would like that to be less true
over time. As a stated goal rather than a hard rule at our current size, we will
avoid nominating new Maintainers from an organization that already employs 50%
or more of existing Maintainers, and we will grow the non-Snowflake share of
Area Reviewers deliberately.

### Inactivity and Emeritus

After **12 months** with no contribution, a Triager or Area Reviewer moves to
Emeritus and their access is removed. This is administrative, not a judgment:
elevated permissions shouldn't sit unused, and an accurate roster tells
contributors who is actually available.

Merit earned does not expire. To come back, open an issue or email a Maintainer
and ask — no re-application, no re-nomination.

Two exceptions:

- **Non-code contributions don't always show up in activity metrics.** If triage,
  documentation, or community work was missed and you were moved to Emeritus in
  error, say so and we will reverse it immediately.
- **Some areas are legitimately stable.** A provider integration that needs no
  changes for a year is a sign it works, not that its reviewer is inactive.
  Maintainers can grant exceptions on that basis.

### Stepping down

Tell a Maintainer, or open a PR against `MAINTAINERS.md`. You will be listed as
Emeritus, and you are welcome back at the same rung whenever you want it.

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
