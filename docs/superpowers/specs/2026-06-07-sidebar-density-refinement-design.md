# Sidebar Density Refinement Design

## Summary

This refinement adjusts the desktop sidebar so the primary controls fit more comfortably without stealing more width from the results workspace.

The main issue is not that the split layout is fundamentally wrong. The issue is that the left rail is too tall, too padded, and gives too much permanent space to low-frequency content. That makes the sidebar feel cramped even though the right-side preview is acceptable.

## Goals

- make the left sidebar feel lighter and more task-focused
- reduce the need to scroll the sidebar during normal search use
- keep the right-side results and preview split largely unchanged
- preserve the existing search and image-search behavior

## Non-goals

- changing retrieval behavior
- redesigning the right-side results workspace
- widening the sidebar significantly
- replacing the split-view preview layout

## Proposed Changes

### Compact Hero

The sidebar intro should be shortened so it acts as a label rather than a content block.

Changes:

- reduce vertical padding
- shorten or remove the descriptive paragraph
- keep the title and product identity visible

### Compact Panels

The left rail panels should use tighter spacing and shorter visual rhythm.

Changes:

- reduce panel padding slightly
- reduce vertical gaps between stacked controls
- keep the same control order

### Conditional Job Progress

Job progress should not consume permanent sidebar space when nothing is running.

Behavior:

- when there is no active or recent job, collapse the progress section into a minimal placeholder or hide its inner content
- when a job starts, expand or reveal the progress content

### Text Search Priority

Text search is the primary interaction and should remain the most prominent section.

Behavior:

- keep `Search By Text` fully expanded and easy to reach
- keep the button and inputs comfortably visible without scrolling past multiple large blocks

### Image Search De-emphasis

Image search should stay available but take less permanent space.

Changes:

- reduce surrounding copy and spacing
- keep only the file input, action button, and compact preview area

## UX Result

After this refinement:

- the left rail reads more like a compact control sidebar
- the user can reach the main search controls faster
- the right-side split workspace remains the main visual focus

## Testing

Add or update lightweight web asset tests to lock the refined structure if class names or text blocks change.

No backend or retrieval tests are needed for this refinement because behavior is unchanged.
