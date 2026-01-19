# ScopePack: Token-Efficient Context Management for Claude Code

## Primary Objective
Build a hook-driven context compression system that reduces token burn in Claude Code sessions by:
1. Preventing huge tool outputs from entering context
2. Maintaining session state via compressed "state packs"
3. Using SCOPE-style semantic chunking and relevance-based compression

## Success Criteria
- [ ] PreToolUse(Read) hook caps large file reads and injects SCOPE summaries
- [ ] PostToolUse(Write|Edit) hook updates file summaries in cache
- [ ] UserPromptSubmit hook injects query-aware ScopePacks
- [ ] SessionStart/SessionEnd hooks persist state across sessions
- [ ] Daemon process loads embedding + summarizer models once (no per-hook startup cost)
- [ ] SQLite cache stores compressed content, file summaries, and session state
- [ ] Measurable token reduction in typical workflows (target: 40-60% reduction on large files)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code Hooks                           │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│ PreToolUse   │ PostToolUse  │ UserPrompt   │ Session           │
│ (Read)       │ (Write/Edit) │ Submit       │ Start/End         │
└──────┬───────┴──────┬───────┴──────┬───────┴───────┬───────────┘
       │              │              │               │
       └──────────────┴──────────────┴───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   scope-daemon    │
                    │  (localhost:18765)│
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
       ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
       │  Embeddings │ │ Summarizer  │ │   SQLite    │
       │ (bge-small) │ │ (BART)      │ │   Cache     │
       └─────────────┘ └─────────────┘ └─────────────┘
```

## Key Design Decisions
1. **Daemon architecture**: Models loaded once, hooks communicate via HTTP
2. **Small models only**: bge-small-en for embeddings, distilbart for summaries (no LLM calls)
3. **Aggressive caching**: Content hash + query hash + budget = cache key
4. **SCOPE algorithm**: Semantic chunking -> relevance scoring -> dynamic compression ratios

## Constraints
- Hooks have timeout limits - daemon must respond quickly
- No external API calls (all local models)
- Must integrate with existing replay-buffer MCP pattern
- Python 3.11+ with sentence-transformers, transformers, sqlite3

## Open Questions
- Optimal chunk size for code vs prose?
- How to handle binary files gracefully?
- Should we integrate with existing memory-bank MCP?
