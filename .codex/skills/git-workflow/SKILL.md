---
name: git-workflow
description: >-
  Git and GitHub mechanics for binary-algo: checkout inventory, other writers
  on main, staging and deletion audit, linear sync, push verification, book
  tags, issue reads and writes, other worktrees and branches, and updating the
  trading host's checkout. Use from feature-dev, plan-review, strategy-eval,
  or dev-cycle, or for "sync", "what changed on main", "state of #<n>", "tag
  the book", and branch or worktree questions. Makes no workflow decisions: a
  bare "commit this and push" still needs the workflow that owns the change.
---

# Git and GitHub mechanics

The calling workflow owns authority, gates, review, and completion. This skill supplies commands and
their failure handling. Delivery uses the primary checkout on `main` with linear history. There are
no pull requests and no hosted checks: local gates are the only gates, so never describe an absent
check as green.

- A read-only question ("what changed on main", "state of #<n>", "which branches exist") is answered
  with the read commands below and changes nothing.
- "Commit this and push" with no caller: establish which workflow owns the change and whether its
  final gate ran on these exact bytes. If not, hand the change to `dev-cycle`, or to
  `strategy-eval` for a result-only delivery, before anything is committed.

## Read the state

- **Inventory.** `git status --short --branch --untracked-files=all`, `git diff --stat`,
  `git diff --cached --stat`, `git worktree list`, `git branch -vv`, `git rev-parse HEAD`. Record
  every staged, unstaged, and untracked path. Touch only state this task created, or a pre-existing
  change the user names as the deliverable; ask when it is unclear which. While automation may be
  running git here, use `git --no-optional-locks` for read-only status.
- **Revision truth.** Before a revision-sensitive claim compare `HEAD`, local `origin/main`, and
  hosted `main` (`git ls-remote origin refs/heads/main`).
- **History.** `git log --oneline <base>..<tip> [-- <path>]`; say which tip was read.

## Other writers of this checkout

Automation can commit and push from this checkout. Before the first edit and before staging, find
out which of these can run now:

- **The lab's result recorder.** `systemctl --user show deriv-recorder.service -p ActiveState -p
  FreezerState`. When a new result file arrives, it refuses the batch, and its loop exits, if `HEAD`
  differs from `origin/main`, a tracked file is modified, or an untracked path is not a result file
  it recognizes. `deriv-lab-ops` owns how to stop and resume it. A recorder frozen under an owner
  pause is left exactly as it is: it cannot write, and stopping a frozen unit thaws it first.
- **The scheduled probe fold** (`scripts/fold_probe_g_capture.py`, run by
  `deriv-probe-g-nearclose.service` after its probe). Check
  `systemctl --user show <unit> -p ActiveState -p SubState -p Job` for both the timer and the
  service; a service that already fired, or a queued start, can still reach its commit step. Its
  commit takes whatever is staged, so nothing may be staged from the first `git add` to the commit
  while it can run.

Default listings leave out inactive units; ask for each writer by name. Never leave an uncommitted
tracked change or a stray untracked file in the checkout while a recorder can run. Keep scratch
files, issue bodies, and prompts outside the repository.

## Commit

- First check the index: `git diff --cached --name-only`. If anything else is already staged, stop;
  do not commit around it. Then stage this task's paths by name: `git add -- <path>...`.
- Read `git diff --cached` and run `git diff --cached --check`. The staged paths and bytes must be
  the ones that were reviewed and gated.
- **Deletion audit.** Account for every line printed by
  `git diff --cached --output-indicator-old='<' | grep '^<'` (no output means no line was removed)
  and for every deleted, renamed, or type-changed path in `git diff --cached --name-status`, and for
  every binary change (a `-` in `git diff --cached --numstat`).
- **Generated ledgers.** `scripts/update_records.py` rewrites only its marked blocks:
  `DERIV-FEED-NOTES` and `DERIV-FEED-TABLES` in a results file, `DERIV-FEED-LEDGER` in a sweep
  file, plus the section and markers it adds the first time. Everything else in those files is
  hand-written and must be unchanged.
- **Size.** Check the size of each staged result file. The hosting service rejects very large
  files; the recorder compacts a large result and archives the original instead of committing it
  whole.
- **Message.** Subject: one imperative sentence of at most 72 characters. Body: scope tag and why;
  a books/results commit also cites the result JSON and the verdict. Keep any attribution trailer
  the host runtime requires. Commit the index as gated, with no path arguments. Never amend or
  rewrite a pushed commit.

## Sync and push

1. `git fetch origin main` must succeed; a failed fetch leaves `origin/main` stale. When
   `git merge-base --is-ancestor origin/main HEAD` exits 1 (any other failure is an error to
   diagnose), `git rebase origin/main` (no merge commits). On a conflict, list the paths with
   `git diff --name-only --diff-filter=U`, integrate both sides against their contracts, `git add`
   them, check `git diff --cached --check`, and `git rebase --continue`. If it cannot be resolved
   safely, `git rebase --abort` and report. A rebase gives the commits new IDs and can drop one as
   already applied.
2. Read `git log --oneline origin/main..HEAD`: every outgoing commit must be one this delivery is
   authorized to ship. Record the candidate, `candidate=$(git rev-parse HEAD)`, and hand it back to
   the calling workflow for the gates, the review, and the authority check it requires before every
   push attempt, retries included.
3. Immediately before pushing, `git rev-parse HEAD` must still equal the candidate; if not, start
   again at step 1. Push exactly it: `git push origin "$candidate:refs/heads/main"`. Never force.
4. After every push result, including a rejection or an ambiguous exit, fetch again and test
   `git merge-base --is-ancestor "$candidate" origin/main`.
   - Status 0: the commit landed, even if the tip has since advanced. Record it once.
   - Status 1 and the remote advanced: repeat from step 1; the candidate will change.
   - Status 1 and the remote did not advance: read the rejection before any retry.
   - Any other status, or a fetch that failed: the outcome is unknown. Keep the commit, report it,
     and do not retry blindly.

## Book tags

`MODEL_REGISTRY.md` requires each frozen book to be tagged `book/<id>` at the commit that contains
it. Tag only after that commit exists on `main`:

1. Use the landed commit of the delivery that completed the book. For an older book, list the
   candidates with `git log --format='%h %s' -- "books/<id>" "books/<id>.manifest.json"` and take
   one to the owner if more than one fits. After a fetch, require
   `git merge-base --is-ancestor "$commit" origin/main`, and require
   `git ls-tree -r --name-only "$commit" -- "books/<id>" "books/<id>.manifest.json"` to list the
   manifest and every artifact it declares. An empty listing means that commit does not contain
   the book.
2. `git tag "book/<id>" "$commit"`. Publish only that ref, and only when publication is authorized:
   `git push origin "refs/tags/book/<id>"`. Verify with
   `git ls-remote --exit-code origin "refs/tags/book/<id>" "refs/tags/book/<id>^{}"`: the `^{}` line
   when one is printed (an annotated tag), otherwise the single line, must equal `$commit`. Exit
   status 2 means the tag is not published; any other failure leaves it unknown.
3. If the tag already exists and points elsewhere, leave it. A published tag is never moved or
   deleted without the owner's decision; report the mismatch with both commits.

## GitHub issues

- `#<n>` is `sppburke/binary-algo#<n>`; pass `-R sppburke/binary-algo` on every call.
- **Account order.** Use the current `gh` account first. Only if it cannot resolve the repository,
  scope the fallback token to one call and never print or store it:
  `(token=$(gh auth token --user sppburke) && [ -n "$token" ] && GH_TOKEN="$token" gh <call>)`.
  If no account can read the repository, report the blocker and continue only local read-only work.
- **Read.** `gh issue view <n> -R sppburke/binary-algo --json number,title,body,comments,state,labels,updatedAt,url`.
  The caller decides which text is the plan: the body by default, a comment only when the user
  names it. Other comments are history and review receipts.
- **Write**, only on the caller's authority, always from a file outside the checkout, never with
  plan text on a command line:
  `gh issue create -R sppburke/binary-algo --title <title> --body-file <file>`,
  `gh issue edit <n> -R sppburke/binary-algo --body-file <file>`,
  `gh issue comment <n> -R sppburke/binary-algo --body-file <file>`,
  `gh issue close <n> -R sppburke/binary-algo --reason completed`.
  Immediately before replacing a body, read the issue again and compare `updatedAt` and the body
  hash with what was inspected; a change in between goes back to the calling workflow. Read the
  result back after every write.
- **Ambiguous create.** Before any retry, list issues in every state
  (`gh issue list -R sppburke/binary-algo --state all --limit 200 --json number,title,body,createdAt,url`)
  and look for the same title and body hash created at the time of the attempt. Reuse a unique
  match; retry only when there is none; otherwise report the outcome as unknown.
- **Body hash.** The receipt hash is the SHA-256 of the body's exact UTF-8 text. Hash the JSON
  field, not printed output, which gains a trailing newline:
  - hosted:
    `gh issue view <n> -R sppburke/binary-algo --json body | ~/binary-algo-venv/bin/python -c "import hashlib,json,sys; print(hashlib.sha256(json.load(sys.stdin)['body'].encode()).hexdigest())"`
  - local file:
    `~/binary-algo-venv/bin/python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <file>`

  After a create or an edit, require the two to be equal before any receipt is posted.

## Other worktrees and branches

A branch checked out in another worktree belongs to that session. Never switch, reset, rebase, or
delete it. Report it instead: `git rev-list --left-right --count main...<branch>` prints the
number of commits only on `main`, then only on the branch, and `git branch -r --contains <branch>`
shows which remote-tracking refs, possibly stale, hold its tip. Bring another worktree's
commits onto `main` only on the owner's explicit request: read them from the pinned branch tip
without touching that checkout, and take them through the delivery workflow as a linear change.
Delete a local branch only when asked, when no worktree has it checked out, and when
`git merge-base --is-ancestor <branch> main` confirms its commits are on `main`. A branch with
commits of its own, and any remote branch, is the owner's decision.

## Trading host

A push is not a deployment. `deriv-runtime-ops` decides whether and when that host's checkout may
move; an observer running there can be bound to one exact revision. When it may: on that host,
`git fetch origin main`, confirm the approved commit is on `origin/main`
(`git merge-base --is-ancestor <approved> origin/main`) and that the checkout can fast-forward to it
(`git merge-base --is-ancestor HEAD <approved>`), `git merge --ff-only <approved>`, and require
`git rev-parse HEAD` to equal it. A bare `git pull` can move past the approved commit. Every
service action on that host belongs to `deriv-runtime-ops`.

## Report

State what was inspected, the exact revisions involved, what was changed, and anything left for
the calling workflow or the owner. Decisions about scope, review, closure, and deployment stay with
the caller.
