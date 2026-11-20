---
name: comment-writing
description: Writing rules for the code comments of this crate. Read before writing or rewriting any `///`, `//!` or `//` comment in src, benches or examples, and when reviewing the comments in a diff.
---

# Comment writing

Comments must be clear, concise, and understandable to a stranger. Avoid unnecessary jargon and roundabout sentence structure.

## Two readers

Who reads a comment depends on the visibility of the item.

- **Public items** are the API docs, and their reader is a **game developer**: someone building a game or app with the crate, reading docs.rs. A public item is one a game can name from outside the crate. Teach that reader how to use the item: what it's for, when to reach for it, what to pass, what happens, and what it costs. Describe behaviour as the game sees it. Name only public items, and use plain language over the internal vocabulary of the crate.
- **Everything else**, `pub(crate)`, private and inline `//`, is for the **maintainer**: the author and the coding agents working on the crate. That reader knows Rust and voxel worlds but not this code. Say what the item is for, why it's built this way, and what it costs, and name the internals freely.

Neither reader has seen the planning the code came out of.

## What to say

The code already says what each line does. A comment helps the reader understand it, by saying:

- **what the item is for**: what it returns, holds or decides;
- **why it's built this way**, where the reader would otherwise wonder, or be tempted to simplify it. When the obvious alternative doesn't work, say why in the present tense;
- **what it costs**, where that matters: what the cost grows with, such as the chunks, the voxels or the light colours. Measured timings and benchmark results stay out, since the reader can't rerun them and they go stale;
- **the cases a caller has to know about**.

Keep a comment short by choosing fewer of these to say. Most items need a summary and one paragraph. Give another paragraph only to an aspect a reader would otherwise get wrong. Say each thing once, on the item a reader would look at for it. Each thing you do say gets a whole, plain sentence: two sentences that read at once beat one that has to be decoded.

Where a sentence describes a formula or a rule of arithmetic, give a small worked example with numbers after it, or put the formula in a code span and explain each term. Keep the example in the terms of its reader: for a public item, what a game developer passes in and sees, not the steps inside.

## Sentences you could say aloud

Write each sentence as you may read it in a book. Elegant, in a spoken tone. A sentence has a subject and a verb, keeps every word it needs, and makes clear what "it" and "its" point at. Each one makes sense on its own, without borrowing words from the sentence before.

- **One idea per sentence.** A sentence that carries two ideas, joined by "and", "or else" or a comma, becomes two sentences.
- **"Of", not "'s".** Instead of "a material's index is", write "the index of a material", "the slots of the chunk", or a compound noun such as "the material index". "Its" is fine.
- **The code does the doing.** A function returns, reads, tries and skips. Data holds, covers and is.
- **Cases go in a list.** A fallback chain, or cases a caller sees handled differently, read best numbered or bulleted. The steps the code takes in order are already in the code, so a list of them only retells it.
- **One name per thing.** Once a chunk "covers" a voxel, everything else that covers it covers it too.
- **The noun again, not "it".** Where "it" or "its" could point at two things in the sentence, or at something two sentences back, name the thing again.

## Adding to a comment

When a change adds a fact to an item that already has a comment, read the whole comment again, then rewrite it so it reads as if it was written in one go:

- The new fact goes in the paragraph it belongs to, in a sentence of its own. A paragraph that now lists several cases becomes a list.
- Every sentence of the comment, old or new, follows these rules. Rewrite each old sentence that doesn't, even where the change didn't touch it.

## Shape

- **Module docs** open with a title line of a few words and no colon, then a blank `//!` line, then a paragraph that starts "This module" and tells how its parts work together. After it come the reasons that span several items: why the module is built the way it is, where a reader would otherwise wonder. That includes the reason for a design its types exist to serve, even where the code that carries the design out lives in another module. A detail of one item belongs in the doc of that item.
- **Item docs** open with one sentence saying what the item does or is. The doc of a function starts with its verb: "Returns", "Spawns", "Marks".
- **One paragraph per aspect**, with a blank `///` line between them: how it works, why, the edge cases, the cost.
- **Sub-headings** go in a module doc, or in a doc comment long enough that a reader scans it for one topic: add one above each topic. A short item doc has none. A bullet under a heading is still a full sentence, with its own subject and verb.
- **Inline comments** follow the same rules in one or two sentences, and say why a line is there.

## Planning notes are source material

Specs, tickets and design notes are written for the people planning the work. They are compressed notes, full of decisions, rejected alternatives, ticket numbers and phrases coined for the plan. Read them for what is true of the code, then explain that afresh for the reader, in your own full sentences:

- Carry over each reason the code still depends on, in the present tense.
- Leave the history in the notes: who decided, what a probe found, which ticket did what, how the code worked before.
- Where a note uses a phrase coined for the plan, write the identifier from the code or a plain description instead.

## Words

- **Names from the code**: `chunk_lod`, `RAY_LOOKAHEAD`, "spawned but not yet published".
- **Crate vocabulary** from GLOSSARY.md: chunk, spawn, publish, generate, level, extent, fill hint, light entry, camera. For any other term, the test is whether the reader could grep the crate and find it.
- **General voxel content.** The crate assumes nothing about terrain, so a comment says "behind solid chunks", and terrain appears only as an example of what a world might hint.

## Example

Before:

```rust
/// The most chunks that have not been spawned yet that one spawning ray queues
/// before it stops.
///
/// Two covers the next chunk along the ray, and one behind it in case the first
/// turns out to be air.
```

After:

```rust
/// How many chunks one spawning ray queues for spawning before it stops.
///
/// Two lets a ray queue the first chunk it meets, and the one behind it in
/// case the first turns out to be air.
```

## Done when

Read the finished comment once, in order, as its reader would: the game developer for a public item, the maintainer for the rest. It is done when no sentence needed a second reading, no "'s" is left, every term is an identifier in the code or a word in GLOSSARY.md, and nothing depends on having read the planning notes.
