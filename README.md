# Hydrus Video Deduplicator

<div align="center">

![hydrus video deduplicator running in terminal](./docs/img/preview.png)

Hydrus Video Deduplicator finds potential duplicate videos through the Hydrus API and marks them as potential duplicates to allow manual filtering through the Hydrus Client GUI.

[![PyPI - Version](https://img.shields.io/pypi/v/hydrusvideodeduplicator.svg)](https://pypi.org/project/hydrusvideodeduplicator)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hydrusvideodeduplicator.svg)](https://pypi.org/project/hydrusvideodeduplicator)
[![PyPI downloads](https://img.shields.io/pypi/dm/hydrusvideodeduplicator.svg)](https://pypistats.org/packages/hydrusvideodeduplicator)
[![GitHub Repo stars](https://img.shields.io/github/stars/hydrusvideodeduplicator/hydrus-video-deduplicator)](https://github.com/hydrusvideodeduplicator/hydrus-video-deduplicator/stargazers)

</div>

---

Hydrus Video Deduplicator **does not modify your files**. It only marks videos as `potential duplicates` through the Hydrus API so that you can filter them manually in the duplicates processing page.

[See the Hydrus documentation for how duplicates are managed in Hydrus](https://hydrusnetwork.github.io/hydrus/duplicates.html).

This program contains no telemetry. It only makes requests to the Hydrus API URL which you specify.

## [Installation](./docs/installation.md)

For Windows and Linux, you can get the [latest release directly from the github releases page](https://github.com/hydrusvideodeduplicator/hydrus-video-deduplicator/releases).

For macos, see the [Python Install](#python-install).

> **Note**: Many instructions in this repo are written for the Python install instead of the Windows exe. If you're using the Windows exe, simply replace `python -m hydrusvideodeduplicator` with `hydrusvideodeduplicator-windows-x86_64.exe` and the instructions will work the same.

## [Usage](./docs/usage.md)

See [Usage](./docs/usage.md) for how to run the program, and [FAQ](./docs/faq.md) for more information.

## Speed optimizations (this fork)

This fork makes the program do the same thing, faster. Every optimization below gives identical results and leaves
your Hydrus and dedupe databases exactly as the original program would, except the opt-in `--skip-repeat-pairs`
and `--prune-deleted-files`.
Each one has an off-switch that restores the original upstream code for that part.

Environment variables are set before running, e.g. in a `.bat` file add a line like `set HVD_PIPELINE=0` before the
command, or in a Linux/macOS shell run `HVD_PIPELINE=0 python -m hydrusvideodeduplicator ...`.

| Switch | Default | What it does | What the off value restores |
|---|---|---|---|
| `HVD_FAST_SEARCH` | `1` (on) | Exact in-memory distance index for "Finding potential duplicates". Falls back automatically if it can't be used. | `0`: original distance calculation (also turns off `HVD_SCAN_SEARCH`). |
| `HVD_SCAN_SEARCH` | `1` (on) | Skips the part of the search tree walk that is the same for every file. | `0`: original tree walk. |
| `HVD_PIPELINE` | `1` (on) | Searches the next files while Hydrus handles the current file's pairs (same calls, same order). | `0`: original one-file-at-a-time loop. |
| `HVD_FAST_INSERT` | `1` (on) | Faster "Building search tree", identical tree. | `0`: original tree inserts. |
| `--skip-repeat-pairs` / `HVD_SKIP_REPEAT_PAIRS=1` | off | Sends each pair to Hydrus once per run instead of usually twice. See below. | Off: original behavior. |
| `--gpu-decode` / `GPU_DECODE=1` | off | Decodes videos on an NVIDIA GPU while hashing (identical hashes, falls back to CPU). | Off: original CPU decoding. |
| `--relink-search-tree` | off | One-time repair tool for a damaged search tree. See below. | Off: nothing runs. |
| `--backup` | off | Backs up the dedupe database before the run changes it. See below. | Off: nothing runs. |
| `--prune-deleted-files` | off | Removes videos that are no longer in Hydrus from the dedupe database. See below. | Off: nothing runs. |

Always on, with no switch (no effect on results or saved data): the next file is downloaded while the current one
is hashed, a faster check for already-hashed files, and a smoother progress bar speed/ETA (display only).

### Speed tips for "Finding potential duplicates"

This step is limited by Hydrus's database. The fewer jobs Hydrus has queued, the faster this phase runs:

- Close any Hydrus duplicates pages (they recount pairs as they arrive).
- Pause duplicate auto-resolution rules.
- Pause Hydrus idle/background maintenance.
- Pause active downloads and subscriptions.

The program prints these tips at the start of the step. It also prints this warning once at startup, after the database filesize line:

> ⚠️ **WARNING**: If you clear or resync potential duplicates in Hydrus (**"delete all potential duplicate pairs and re-search"** or **"resync potential duplicate pairs to storage"**), you will have to clear this tool's search cache (`--clear-search-cache`) and re-search every video from zero, which can take DAYS on a large library.

### `--skip-repeat-pairs`

Most potential duplicate pairs are found twice during the search, once from each file's side, and by default each
time is sent to Hydrus. Hydrus stores the pair the same way both times, so the second send changes nothing.
`--skip-repeat-pairs` (or the environment variable `HVD_SKIP_REPEAT_PAIRS=1`) sends each pair only once per run,
which cuts the number of Hydrus API calls while finding potential duplicates. It is off by default.

- A pair is skipped only if that exact pair was already sent successfully earlier in the same run.
- Nothing is saved differently: the dedupe database is not changed, and Hydrus ends up with the same pairs.
- Downloads, subscriptions and duplicate auto-resolution can keep running in Hydrus while it runs.

### `--relink-search-tree`

Repairs a damaged search tree: relinks every perceptual hash into one valid search tree (takes seconds, nothing is
re-hashed), then clears the search cache so the following duplicate search checks every file again (this search
can take a long time on big databases, and can be stopped and resumed). Back up your database first. Can't be
combined with `--clear-search-tree`.

### `--backup`

Copies the dedupe database before the run changes it, into the same folder, named like
`videohashes.backup-20261121-013000.sqlite`. The copy is made with SQLite's own backup method, so it is consistent.
The 3 newest backups are kept and older backups with that name pattern are deleted; no other files are touched.
Nothing happens if there is no database yet. Each backup is the size of the database. In the GUI this is the
**Back Up Database** button.

### `--prune-deleted-files`

The dedupe database keeps every video it ever hashed, even after you delete the video in Hydrus. Those videos stay
in the search tree, can still be found as matches for new videos (each one is a wasted Hydrus call), and are searched
if they had not been searched yet. `--prune-deleted-files` removes them:

- It always makes a backup first (it turns on `--backup`).
- It asks Hydrus once for every video, gif and apng in "all local files" (read only; not limited by `--query` or
  `--file-service-key`). Anything still in Hydrus, including the trash, is kept.
- Every hashed video Hydrus no longer has is removed from the database, the search cache and the queue of videos
  waiting to be added to the search tree. Perceptual hashes that no remaining video uses are removed from the search
  tree, and the tree is relinked around them so it stays valid. Nothing is re-hashed and remaining videos keep their
  search progress.
- If Hydrus can't be reached or returns no videos, nothing is pruned and the run continues.
- It is all-or-nothing: on an error or CTRL+C the database is left unchanged.
- It prints a summary like `Pruned 5912 deleted videos from the database and 5612 entries from the search tree.`
- The database format does not change, so the pruned database still works with older versions of the program.

It runs at startup, before perceptual hashing. In the GUI this is the **Prune Deleted Files** button.

## Python Install

> **Note**: Windows users do not need the Python install. Download the exe as described in [Installation](#installation) instead.

### Dependencies

- [Python](https://www.python.org/downloads/) >=3.10

```sh
python -m pip install hydrusvideodeduplicator
```

Then continue to [Usage](./docs/usage.md).

---

## [Contact](./docs/contact.md)

Create an issue on GitHub for any problems/concerns. Provide as much detail as possible in your issue.

Email `hydrusvideodeduplicator@gmail.com` for other general questions/concerns.

Message `@applenanner` on the [Hydrus Discord](https://discord.gg/wPHPCUZ) for other general questions/concerns.
