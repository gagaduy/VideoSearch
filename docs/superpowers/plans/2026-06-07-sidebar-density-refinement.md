# Sidebar Density Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the desktop sidebar more compact and usable without changing search behavior or shrinking the right-side workspace further.

**Architecture:** Keep the existing left-sidebar and right split-view structure, but reduce sidebar density through tighter spacing, a shorter hero, and conditional/minimized progress and image-search presentation. This is a web-only refinement with no backend changes.

**Tech Stack:** Static HTML, CSS, JavaScript, pytest

---

## File Map

- Modify: `web/index.html`
  - Reduce permanent sidebar content and simplify low-priority sections.
- Modify: `web/styles.css`
  - Tighten sidebar spacing, compact the hero, and reduce panel height pressure.
- Modify: `web/app.js`
  - Hide or minimize job progress when idle if needed by the chosen markup.
- Test: `tests/unit/test_web_assets.py`
  - Lock the new compact sidebar structure and idle-progress behavior hooks.

### Task 1: Lock Compact Sidebar Markup Expectations

**Files:**
- Modify: `tests/unit/test_web_assets.py`
- Test: `tests/unit/test_web_assets.py`

- [ ] **Step 1: Write the failing test**

```python
def test_index_uses_compact_sidebar_markup() -> None:
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert 'class="left-rail compact-sidebar"' in html
    assert 'class="hero panel hero-compact"' in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.conda/bin/python -m pytest tests/unit/test_web_assets.py::test_index_uses_compact_sidebar_markup -v`
Expected: FAIL because the compact sidebar classes are missing

- [ ] **Step 3: Write minimal implementation**

```html
<aside class="left-rail compact-sidebar">
  <header class="hero panel hero-compact">
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.conda/bin/python -m pytest tests/unit/test_web_assets.py::test_index_uses_compact_sidebar_markup -v`
Expected: PASS

### Task 2: Tighten Sidebar Density

**Files:**
- Modify: `web/index.html`
- Modify: `web/styles.css`
- Test: `tests/unit/test_web_assets.py`

- [ ] **Step 1: Write the failing test**

```python
def test_index_uses_compact_image_search_container() -> None:
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert 'class="query-image-preview-slot query-image-preview-compact"' in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.conda/bin/python -m pytest tests/unit/test_web_assets.py::test_index_uses_compact_image_search_container -v`
Expected: FAIL because the compact class is missing

- [ ] **Step 3: Write minimal implementation**

```html
<section id="query-image-preview" class="query-image-preview-slot query-image-preview-compact">
```

```css
.compact-sidebar {
  gap: 0.75rem;
}

.hero-compact {
  padding: 0.95rem 1rem;
}

.compact-sidebar .panel {
  padding: 1rem;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.conda/bin/python -m pytest tests/unit/test_web_assets.py::test_index_uses_compact_image_search_container -v`
Expected: PASS

### Task 3: Minimize Idle Job Progress Footprint

**Files:**
- Modify: `web/app.js`
- Test: `tests/unit/test_web_assets.py`

- [ ] **Step 1: Write the failing test**

```python
def test_web_app_contains_idle_progress_guard() -> None:
    contents = Path("web/app.js").read_text(encoding="utf-8")
    assert "function syncIdleJobStatus()" in contents
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.conda/bin/python -m pytest tests/unit/test_web_assets.py::test_web_app_contains_idle_progress_guard -v`
Expected: FAIL because the helper does not exist

- [ ] **Step 3: Write minimal implementation**

```javascript
function syncIdleJobStatus() {
  if (!jobStatus) {
    return;
  }
  if (jobStatus.hidden) {
    jobStatus.parentElement?.classList.add("panel-idle");
  } else {
    jobStatus.parentElement?.classList.remove("panel-idle");
  }
}
```

- [ ] **Step 4: Call the helper from the existing render flow**

```javascript
syncIdleJobStatus();
```

- [ ] **Step 5: Run test to verify it passes**

Run: `./.conda/bin/python -m pytest tests/unit/test_web_assets.py::test_web_app_contains_idle_progress_guard -v`
Expected: PASS

### Task 4: Verify Web Asset Coverage And Rebuild UI

**Files:**
- Modify: none
- Test: `tests/unit/test_web_assets.py`

- [ ] **Step 1: Run the full web asset suite**

Run: `./.conda/bin/python -m pytest tests/unit/test_web_assets.py -v`
Expected: PASS

- [ ] **Step 2: Rebuild the web container**

Run: `docker compose up -d --build web`
Expected: web image rebuilds and container starts successfully

- [ ] **Step 3: Verify the served HTML contains the compact sidebar classes**

Run: `curl -sf http://localhost:8080 | rg 'compact-sidebar|hero-compact|query-image-preview-compact'`
Expected: all three markers are present
