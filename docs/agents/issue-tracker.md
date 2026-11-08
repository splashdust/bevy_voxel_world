# Issue tracker: Local Markdown

Specs, tickets and design work for this repo live as markdown files in `.scratch/`, which is git-ignored and never committed.

## Lifetime

`.scratch/` holds planning, not documentation. Once a feature is implemented, its directory is deleted, and from then on the code documents how the feature works. So:

- Before a feature's directory goes, anything a reader of the finished code needs moves into the code (module docs, doc comments) or `GLOSSARY.md`. See `domain.md`.
- Never reference a `.scratch/` path from code, commits, PR descriptions, `CHANGELOG.md` or any committed doc.
- A missing `.scratch/<feature>/` usually means the feature shipped, not that work was lost.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Tickets are one file each at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, two-digit numbers in order, never one combined tickets file
- Each ticket opens with YAML frontmatter:

  ```yaml
  ---
  title: <ticket name>
  labels: []
  status: open        # open | closed
  assignee:           # set when claimed
  blocked_by: []      # ticket numbers, e.g. [03, 07]
  ---
  ```
- Comments and conversation history go at the bottom of the file under a `## Comments` heading
- Supporting material (research notes, probe output, baselines, screenshots) goes in subfolders of the feature directory, such as `research/` and `baseline/`
- `.scratch/probes/` is not a feature: it holds throwaway probe code and scripts shared across efforts

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/` (creating the directory if needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the ticket number directly.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md`, with frontmatter `labels: [wayfinder:map]` and the Destination / Notes / Decisions-so-far / Fog body.
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md` with the question under `## Question`. `labels: [wayfinder:<type>]` records the type (`research`/`prototype`/`grilling`/`task`).
- **Blocking**: `blocked_by` in the frontmatter. A ticket is unblocked when every ticket it lists is `status: closed`.
- **Frontier**: scan `.scratch/<effort>/issues/` for tickets that are `status: open`, unblocked, and have no `assignee`; lowest number wins.
- **Claim**: set `assignee` and save before any work.
- **Resolve**: write the answer under a `## Resolution` heading, set `status: closed`, then add a gist and link to the map's Decisions-so-far.
