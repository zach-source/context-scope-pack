# Tasks: ScopePack Implementation

## In Progress
- [ ] **TASK-001**: Create project infrastructure
  - Status: Directory structure created, working on devenv.nix
  - Files: devenv.nix, pyproject.toml, .gitignore

## Pending
- [ ] **TASK-002**: Create SQLite schema
  - Tables: content_cache, file_summaries, session_state
  - File: src/scopepack/db.py

- [ ] **TASK-003**: Implement scope-daemon skeleton
  - HTTP server on localhost:18765
  - Endpoints: /compress, /summarize, /health
  - File: src/scopepack/daemon.py

- [ ] **TASK-004**: Implement SCOPE compression algorithm
  - Semantic chunking (code blocks, paragraphs)
  - Relevance scoring via embeddings
  - Dynamic compression ratios
  - File: src/scopepack/scope.py

- [ ] **TASK-005**: Create PreToolUse(Read) hook
  - Cap large file reads
  - Inject SCOPE summaries
  - File: .claude/hooks/scope_pretool_read.py

- [ ] **TASK-006**: Create PostToolUse(Write|Edit) hook
  - Update file summaries in cache
  - Track file edit frequency
  - File: .claude/hooks/scope_posttool_write_edit.py

- [ ] **TASK-007**: Create UserPromptSubmit hook
  - Build query-aware ScopePack
  - Inject git status, hot files, replay facts
  - File: .claude/hooks/scope_user_prompt_submit.py

- [ ] **TASK-008**: Create SessionStart hook
  - Load state pack from cache
  - File: .claude/hooks/scope_session_start.sh

- [ ] **TASK-009**: Create SessionEnd hook
  - Save compressed state pack
  - File: .claude/hooks/scope_session_end.py

- [ ] **TASK-010**: Create .claude/settings.json
  - Wire all hooks with proper matchers
  - File: .claude/settings.json

## Completed
_None yet_

## Notes
- Focus on Phase 1 first (biggest token savings)
- Test with actual Claude Code sessions to measure impact
- Consider integration with existing replay-buffer MCP
