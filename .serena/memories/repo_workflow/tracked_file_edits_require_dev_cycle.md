# Tracked-File Edit Rule

For `/home/sean/git/binary-algo`, any task that will create, edit, delete, move, stage, commit, or otherwise change a git-tracked file must invoke and follow the repo-local `dev-cycle` skill from the start. Pure read-only inspection does not require `dev-cycle`; once work transitions from inspection to tracked-file mutation, switch to `dev-cycle` before editing.

This rule is also recorded in `CLAUDE.md`; `AGENTS.md` is a symlink to `CLAUDE.md`.