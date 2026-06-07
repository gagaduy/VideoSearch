# UI Layout And Image Search Design

## Summary

This spec redesigns the web UI to use the full viewport, reduces noisy search output, and adds a dedicated image-to-frame retrieval flow.

Goals:

- keep the main controls visible without forcing the user to scroll the whole page
- limit result noise by showing only strong matches and capping visible frames
- add a separate "search by image" flow that uploads one image and finds the most visually similar frames
- prevent accidental repeated search submissions from piling up requests

Non-goals:

- mixed text+image search in one request
- pagination or infinite scroll
- multi-image search
- object crop search

## Current Problems

The current UI stacks ingest, search, results, and preview vertically. On desktop this wastes horizontal space and pushes results far down the page.

The search results render as a long grid with no strong relevance cut-off. This makes weak matches consume space and attention.

The product only supports text search. There is no direct way to upload an example image and retrieve similar frames.

The current search form can be submitted repeatedly while a request is still running, which risks duplicate concurrent requests and a frozen-feeling UI.

## Proposed UX

### Overall Layout

The page will switch to a full-viewport workspace layout:

- a fixed left rail for controls
- a large right content pane for results and preview

Left rail sections:

- video import
- indexing progress
- text search
- image search

Right pane sections:

- search results grid as the main content area
- selected result preview and timeline details in a secondary panel

Desktop behavior:

- the page itself should not grow into a long vertical document for normal use
- the left rail stays visible
- the results area and preview area scroll internally if needed

Small-screen behavior:

- collapse back into a stacked layout
- preserve all features without horizontal overflow

### Text Search Results

Text search will keep the current backend retrieval pipeline, but the final display set will be filtered:

- backend still gathers a wider candidate set
- final results are filtered by a minimum relevance threshold
- remaining results are capped to a hard maximum of 16 frames

Expected behavior:

- strong queries may show anywhere from a few results up to 16
- weaker queries may show fewer than 10
- if nothing clears the threshold, the UI shows an explicit empty-state message instead of weak filler results

### Image Search

Image search is a separate mode from text search.

Flow:

1. user uploads one image
2. backend embeds the uploaded image using the visual embedding model already used for frame retrieval
3. backend retrieves the nearest frame vectors
4. threshold and cap are applied to the ranked results
5. UI renders the same result cards and preview experience used by text search

The uploaded query image will be shown in the left rail while its results are active so the user can confirm what reference image is being used.

Text search and image search do not combine in this iteration. Each submit action runs its own retrieval mode.

## Backend Design

### New Image Search Endpoint

Add a dedicated API endpoint for image-based retrieval.

Responsibilities:

- accept one uploaded image file
- validate it as an image input
- embed it with the OpenCLIP adapter
- search frame or segment vectors using the same vector index family used for visual retrieval
- return a response shape that matches the existing search response closely enough for the web client to reuse rendering paths

The endpoint should return:

- retrieval mode metadata such as `mode: "image"`
- the result list
- the query image metadata needed by the UI if useful

### Result Filtering Policy

Both text and image retrieval flows will use a filtered display policy:

- relevance threshold first
- hard cap second

The threshold should be mode-aware:

- text search keeps its own calibrated threshold
- image search gets its own threshold because score scales may differ

This avoids forcing both retrieval modes onto one arbitrary score cut-off.

If filtering removes all results, the response should still succeed with an empty `results` array rather than treating it as an error.

### Search Submission Guard

The frontend will disable the active search submit button while its request is in flight. This is a UX-side guard against duplicate requests.

No backend queueing or deduplication changes are required in this iteration.

## Frontend Design

### Structure

The static web app will be reorganized into a two-pane shell:

- left sidebar for controls and query state
- right workspace for results and preview

The result grid should fit comfortably within one page view on desktop, with cards sized so 10-16 results remain scan-friendly.

### Shared Result Rendering

Text search and image search should reuse the same:

- result card layout
- retrieval score display
- preview panel
- timeline snippet behavior

This keeps the UI behavior consistent regardless of query mode.

### States

The UI should clearly represent:

- idle state
- loading state
- empty results state
- error state

For image search specifically:

- no file selected
- invalid file upload
- valid query image selected and active

### Buttons And Request Safety

While text search is running:

- disable the text search button
- leave image search available unless shared state makes that unsafe

While image search is running:

- disable the image search button
- keep the chosen image preview visible

This prevents accidental multi-submit from the same form while staying lightweight.

## Error Handling

Text search:

- if no result clears the threshold, return success with zero results and show `No strong matches found for this query.`

Image search:

- reject missing file submissions at the UI layer
- reject unsupported image uploads with a clear backend error
- return success with zero results if no frame is strong enough

The UI should distinguish:

- invalid request
- request failed
- valid request with no strong matches

## Testing

Add tests for:

- text result filtering with threshold plus hard cap
- image search API upload path
- image search empty-state behavior
- web asset coverage for the new layout and image search form
- search button disabling during in-flight requests where current test coverage allows

Regression coverage should ensure existing text search and preview rendering still work when image search is added.

## Rollout Notes

This is intentionally scoped as a moderate change:

- reuse existing retrieval and rendering structures where possible
- avoid introducing pagination, hybrid search, or richer gallery tooling
- keep score presentation as a retrieval score, not a percentage

The implementation should favor small backend additions and a more substantial but contained web UI rewrite.
