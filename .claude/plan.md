# Implementation Plan: ScopePack

## Approach
Build incrementally with the "biggest win first" strategy:
1. Phase 1: PreToolUse(Read) capping + SCOPE summaries (immediate token savings)
2. Phase 2: UserPromptSubmit ScopePack injection (git/diff + hot-file summaries)
3. Phase 3: Session persistence (SessionStart/SessionEnd handoff packs)

---

## Phase 1: Core Infrastructure + Read Hook 🔄 IN PROGRESS
**Goal**: Reduce tokens from large file reads by 80%+
**Status**: Started

### Tasks
- [x] Create project structure
- [x] Set up devenv.nix with Python + dependencies
- [ ] Create SQLite schema for caching
- [ ] Implement scope-daemon skeleton
- [ ] Implement SCOPE compression algorithm (semantic chunking + relevance scoring)
- [ ] Create PreToolUse(Read) hook
- [ ] Test with sample large files

### Technical Notes
- Daemon listens on localhost:18765
- Cache key: sha256(content) + sha256(query) + budget_tokens + model_version
- Initial compression: head+tail fallback if daemon unavailable

---

## Phase 2: UserPromptSubmit ScopePack
**Goal**: Inject query-aware context at prompt submission
**Status**: Not Started

### Tasks
- [ ] Define ScopePack format (v1)
- [ ] Implement git status/diff extraction
- [ ] Implement hot-file tracking (recently edited files)
- [ ] Create UserPromptSubmit hook
- [ ] Integrate with existing replay-buffer facts

### ScopePack Format
```
[SCOPEPACK v1]
Goal: <1 line>
Current task state: <3-8 bullets>
Repo state:
- branch, dirty/clean
- changed files (top N)
Relevant symbols:
- file:Symbol - 1-line meaning
Constraints:
- tests to keep green, style rules, runtime
Open questions:
- ...
[/SCOPEPACK]
```

---

## Phase 3: Session Persistence
**Goal**: Carry state across session clears without losing context
**Status**: Not Started

### Tasks
- [ ] Implement SessionStart hook (load state pack)
- [ ] Implement SessionEnd hook (save state pack)
- [ ] Define state pack schema
- [ ] Test session handoff workflow

---

## Phase 4: PostToolUse Cache Updates
**Goal**: Keep file summaries fresh after edits
**Status**: Not Started

### Tasks
- [ ] Implement PostToolUse(Write|Edit) hook
- [ ] Invalidate/update cache on file changes
- [ ] Track file edit frequency for "hot file" detection

---

## Technical Decisions

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| HTTP daemon vs subprocess | Model loading once vs per-hook startup cost | IPC, Unix sockets |
| bge-small-en embeddings | Fast, good quality, 384 dims | MiniLM, all-mpnet |
| distilbart-cnn summarizer | Balance of speed and quality | bart-large-cnn, LED |
| SQLite cache | Simple, fast, no external deps | Redis, LevelDB |
| Python for hooks | transformers/sentence-transformers ecosystem | Rust (faster but harder) |

---

## Risks & Mitigations

- **Risk**: Daemon startup time too slow
  - **Mitigation**: Lazy model loading, start daemon at session start

- **Risk**: Compression loses critical context
  - **Mitigation**: Always include exact text slice (first 200 lines), keyword preservation

- **Risk**: Cache grows unbounded
  - **Mitigation**: LRU eviction, configurable max size
