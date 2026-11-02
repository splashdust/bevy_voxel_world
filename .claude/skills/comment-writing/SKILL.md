---
name: comment-writing
description: Writing rules for this crate's code comments. Read before writing or rewriting any `///`, `//!` or `//` comment in src, benches or examples, and when reviewing the comments in a diff.
---

# Comment writing

Comments must be clear, concise, and understandable to a stranger. Avoid unnecessary jargon and roundabout sentence structure.

Write every comment for a **stranger**: a Rust programmer who knows voxel worlds but has never seen this crate, this branch, or the conversation the code came out of.

Clarify what the purpose of the item is, and any important details about how it works or why it exists. But keep it short and let the code speak for itself.

## Shape

- **Summary first.** A doc comment opens with one short sentence saying what the item does. What it returns gets a sentence of its own.
- **One paragraph per aspect**, with a blank `///` line between them: how it works, and why, the edge cases, the effect on the whole.
- **Inline comments** follow the same rules at the scale of one or two sentences.

## Words

- **Plain verbs and nouns**: passes through, stops at, queues, returns; chunk, camera, air.
- **The code's own names.** Where you would write a shorthand for something the code names, write the name: `chunk_lod`, `RAY_LOOKAHEAD`, "spawned but not yet published". A phrase coined during a design session means nothing to the stranger; the identifier or a plain description does.
- **Crate vocabulary**: chunk, spawn, publish, generate, level, extent, fill hint, camera. Use the GLOSSARY.md. The test for any other word: the stranger can grep the crate and find it.
- **General voxel content.** The crate assumes nothing about terrain, so a comment says "behind solid chunks", and terrain appears only as an example of what a world might hint.

## Done when

Read the finished comment as the stranger, sentence by sentence. The comment is done when every sentence can be understood from the code's identifiers and the crate's own words alone, and nothing in it leans on what the author knew from elsewhere.
