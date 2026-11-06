# Domain Docs

How the engineering skills should use this repo's domain documentation when exploring the codebase.

## Layout

Single-context:

```
/
├── GLOSSARY.md
├── docs/adr/        ← rare; usually absent
└── src/
```

## Before exploring, read these

- **`GLOSSARY.md`** at the repo root.
- **`docs/adr/`**, if it exists: read ADRs that touch the area you're about to work in.

If either is missing, **carry on without mentioning it**. Don't flag that it's missing, and don't suggest creating it upfront.

## Where knowledge lives

The code is the documentation. How a feature works belongs in the code itself: module docs, doc comments, and well-named types (follow the `comment-writing` skill). Domain vocabulary belongs in `GLOSSARY.md`. Planning material in `.scratch/` is deleted once a feature ships (see `issue-tracker.md`), so anything worth keeping moves into one of those two places first.

## ADRs: very sparingly

By default, don't write an ADR. Write one only when a decision meets both of these:

- It can't reasonably be documented next to the code it governs, because it spans many modules or rules out an approach and so leaves no code behind to comment on.
- Without it, the decision would get argued again: someone reading the code would reasonably try the rejected path.

If a decision fits in a module doc or a doc comment on the type it shapes, it goes there instead. When `/domain-modeling` or another skill offers to record an ADR, check it against this bar first. If it fails, propose a place in the code instead.

## Use the glossary's vocabulary

When your output names a domain concept (in a ticket title, a refactor proposal, a hypothesis, a test name), use the term as defined in `GLOSSARY.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, say so explicitly rather than quietly overriding it:

> _Contradicts ADR-0003 (…), but worth reopening because…_
