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

### Faster duplicate search: `--skip-repeat-pairs`

Most potential duplicate pairs are found twice during the search, once from each file's side, and by default each
time is sent to Hydrus. Hydrus stores the pair the same way both times, so the second send changes nothing.
`--skip-repeat-pairs` (or the environment variable `HVD_SKIP_REPEAT_PAIRS=1`) sends each pair only once per run,
which roughly halves the number of Hydrus API calls while finding potential duplicates. It is off by default.

- A pair is skipped only if that exact pair was already sent successfully earlier in the same run.
- Nothing is saved differently: the dedupe database is not changed, and Hydrus ends up with the same pairs.
- Downloads, subscriptions and duplicate auto-resolution can keep running in Hydrus while it runs.

> ⚠️ **Warning**: do not use **"delete all potential duplicate pairs and re-search"** or **"resync potential duplicate pairs to storage"** in Hydrus while the program runs with this option. If you do, run the program again with `--clear-search-cache` so every pair is sent again.

<details>
<summary>Example</summary>
<br>

```sh
python -m hydrusvideodeduplicator --api-key="..." --skip-repeat-pairs
```

</details>
