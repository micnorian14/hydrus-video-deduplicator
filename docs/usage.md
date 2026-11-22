# Usage

## Creating a Hydrus API Key

1. [Enable the Hydrus Client API service](https://hydrusnetwork.github.io/hydrus/client_api.html#enabling_the_api). Ensure `use https` is enabled for the client api service.

    - If you need https instead of http, you must specify this in the Hydrus API URL.
        - Ex. For the CLI, use `python -m hydrusvideodeduplicator --api-url=https://localhost:45869`

    - Enable `allow non-local connections` in `manage services->client api` if you are running under [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/), or the connection will fail.

    - ⚠️ For https, SSL cert is **not verified** by default unless you enter the cert's file path.
      - Ex. For the CLI, use `--verify-cert`
      - For the GUI, this is only supported through the `REQUESTS_CA_BUNDLE` environment variable.

2. Create a client api service key with `permits everything` enabled.

    - Do NOT set blacklist/whitelist filters for the API token.

> **Example**: The API key should look something vaguely like `78d2fcc9fe1f43c5008959ed1abfe38ffedcfa127d4f051a1038e068d3e32656`

> **Note**: You will need this API key in the next step, so you should probably copy this API key to your clipboard.

After getting your API key, continue to [Running Video Dedupe](#running-video-dedupe).

## Running Video Dedupe

<details open>
<summary>Windows</summary>
<br>

Run hydrusvideodeduplicator-windows-x86_64.exe and enter the Hydrus API key you created previously when prompted.

</details>

<br>

<details>
<summary>Linux and macos</summary>

Run the program and enter the Hydrus API key you created previously.

<br>

Example:

```sh
python -m hydrusvideodeduplicator --api-key="78d2fcc9fe1f43c5008959ed1abfe38ffedcfa127d4f051a1038e068d3e32656"
```

To run the gui, add `--gui` to the command above. Ensure you have the GUI dependencies installed.

</details>

<br>

To cancel any stage of processing at any time, press CTRL+C.

You may want to cancel the first step, perceptually hashing, before it's finished in order to search for duplicates on that subset of videos that have been hashed. The next time you run the program it will continue where you left off. This may be useful for large databases that may take forever to perceptually hash.

See the full list of options with `--help`.

See the [FAQ](./faq.md) for more information.

### Advanced Usage

You can select certain files with queries just like Hydrus. e.g. `--query="character:batman"`

<details>
<summary>Example</summary>
<br>

```sh
python -m hydrusvideodeduplicator --api-key="..." --query="character:batman"
```

</details>

### Speed optimizations, speed tips and `--skip-repeat-pairs`

See [Speed optimizations (this fork)](../README.md#speed-optimizations-this-fork) in the README for every speed
switch and its default, the speed tips for "Finding potential duplicates", and the warning about clearing or
resyncing potential duplicates in Hydrus. See [New flags](../README.md#new-flags) for how each new flag works,
including `--skip-repeat-pairs`, `--gpu-decode`, `--relink-search-tree` and `--prune-orphans`.

### `--backup`, `--prune-deleted-files` and `--prune-orphans`

These flags are off by default. `--prune-deleted-files`, `--prune-orphans` and `--relink-search-tree` make a backup first.

- `--backup` copies the dedupe database next to itself before the run changes it, and keeps the 3 newest backups.
- `--prune-deleted-files` removes videos you have deleted in Hydrus from the dedupe database and search tree.
- `--prune-orphans` removes videos whose perceptual hash is missing, so the next run hashes them again.

Failed files are sent to a Hydrus page named `failed` if you have created that page. See [`--failed-page-name`](../README.md#--failed-page-name). Every run also prints two read-only [startup checks](../README.md#startup-checks).

```sh
python -m hydrusvideodeduplicator --api-key="..." --prune-deleted-files
```

See [`--backup`](../README.md#--backup), [`--prune-deleted-files`](../README.md#--prune-deleted-files) and [`--prune-orphans`](../README.md#--prune-orphans) in the
README for details.

> ⚠️ **WARNING**: If you clear or resync potential duplicates in Hydrus (**"delete all potential duplicate pairs and re-search"** or **"resync potential duplicate pairs to storage"**), you will have to clear this tool's search cache (`--clear-search-cache`) and re-search every video from zero. A large library can take several hours or days.
