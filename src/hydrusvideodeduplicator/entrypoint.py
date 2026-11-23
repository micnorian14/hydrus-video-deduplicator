from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, List, Optional

if TYPE_CHECKING:
    from typing import NoReturn

from pathlib import Path

import typer
from rich import print

from hydrusvideodeduplicator.__about__ import __version__
from hydrusvideodeduplicator.client import (
    ClientAPIException,
    FailedHVDClientConnection,
    HVDClient,
    create_client,
)
from hydrusvideodeduplicator.config import (
    Config,
    GpuDecodeArgError,
    normalize_gpu_decode_args,
    resolve_gpu_decode,
)
from hydrusvideodeduplicator.db import DedupeDB
from hydrusvideodeduplicator.db_tools import (
    check_database,
    print_queue_waiting,
    run_backup,
    run_prune,
    run_prune_orphans,
)
from hydrusvideodeduplicator.dedup import HydrusVideoDeduplicator, print_search_cache_warning
from hydrusvideodeduplicator.rehash import run_rehash_overhashed
from hydrusvideodeduplicator.dedup_util import (
    TqdmLoggingHandler,
    console_print,
    display_path,
    print_and_log,
    style_user_text,
)
from dotenv import find_dotenv, load_dotenv

import sys

"""
Parameters:
- api_key will be read from env var $HYDRUS_API_KEY or .env file
- api_url will be read from env var $HYDRUS_API_URL or .env file
- to add custom queries, do
  --custom-query="series:twilight" --custom-query="character:edward" ... etc for each query
- threshold is the min % matching to be considered similar. 100% is very very similar.
- verbose turns on logging
- debug turns on logging and sets the logging level to debug
"""

loaded_dotenv = load_dotenv()
config = Config.load_from_env()

# Shown at the bottom of --help, one paragraph per item.
CLI_HELP = "Find potential duplicate videos in Hydrus and mark them as potential duplicates for manual review."
CLI_EPILOG = "\n\n".join(
    [
        "Speed switches (environment variables; set to 0 to use the original upstream code; details in README.md):",
        "HVD_FAST_SEARCH=1  HVD_SCAN_SEARCH=1  HVD_PIPELINE=1  HVD_FAST_INSERT=1",
    ]
)


def main(
    api_key: Annotated[
        Optional[str],
        typer.Option(help="Hydrus API Key", show_default=False, prompt=False if config.hydrus_api_key else True),
    ] = config.hydrus_api_key,
    api_url: Annotated[Optional[str], typer.Option(help="Hydrus API URL")] = config.hydrus_api_url,
    overwrite: Annotated[Optional[Optional[bool]], typer.Option(hidden=True)] = None,  # deprecated
    query: Annotated[Optional[List[str]], typer.Option(help="Custom Hydrus tag query")] = config.hydrus_query,
    threshold: Annotated[
        Optional[float], typer.Option(help="Similarity threshold for a pair of videos where 100 is identical")
    ] = config.similarity_threshold,
    skip_hashing: Annotated[
        Optional[bool], typer.Option(help="Skip perceptual hashing and just search for duplicates")
    ] = False,
    file_service_key: Annotated[
        Optional[List[str]], typer.Option(help="Local file service key")
    ] = config.hydrus_local_file_service_keys,
    verify_cert: Annotated[
        Optional[str], typer.Option(help="Path to TLS cert. This forces verification.")
    ] = config.requests_ca_bundle,
    clear_search_tree: Annotated[
        Optional[bool], typer.Option(help="Clear the search tree that tracks what files have already been compared.")
    ] = False,
    relink_search_tree: Annotated[
        Optional[bool],
        typer.Option(help="Repair a damaged search tree (one-time; see README). Turns on --backup."),
    ] = False,
    clear_search_cache: Annotated[
        Optional[bool],
        typer.Option(
            help="Clear the search cache that tracks what files have been compared with a given similarity threshold."
        ),
    ] = False,
    backup: Annotated[
        Optional[bool],
        typer.Option(
            help="Back up the dedupe database before this run changes it. Keeps the 3 newest backups next to the database."  # noqa: E501
        ),
    ] = False,
    prune_deleted_files: Annotated[
        Optional[bool],
        typer.Option(
            help="Remove videos that are no longer in Hydrus from the dedupe database and search tree. Turns on --backup."  # noqa: E501
        ),
    ] = False,
    prune_orphans: Annotated[
        Optional[bool],
        typer.Option(
            help="Remove broken entries (videos whose perceptual hash is missing) so they are re-hashed. Turns on --backup."  # noqa: E501
        ),
    ] = False,
    rehash_overhashed: Annotated[
        Optional[bool],
        typer.Option(
            help="Once: re-hash videos hashed with far too many frames (see README). Turns on --backup."
        ),
    ] = False,
    failed_page_name: Annotated[
        Optional[str],
        typer.Option(help="The name of the Hydrus page to add failed files to. Defaults to 'failed'."),
    ] = config.failed_page_name,
    job_count: Annotated[
        Optional[int],
        typer.Option(help="Number of CPU threads to use for perceptual hashing. Default is all but one core."),
    ] = config.job_count,
    # Checked and normalized by normalize_gpu_decode_args; the default comes from GPU_DECODE.
    gpu_decode: Annotated[
        Optional[str],
        typer.Option(
            metavar="[auto|all|off]",
            help="Decode videos on an NVIDIA GPU (CUDA). 'auto' (recommended) uses the GPU and the CPU together and "
            "learns which is faster for each kind of video. 'all' tries to force all files to decode on the GPU. "
            "'off' uses only the CPU.",
        ),
    ] = config.gpu_decode_mode,
    reset_gpu_profile: Annotated[
        Optional[bool],
        typer.Option(
            help="Make '--gpu-decode auto' forget the CPU and GPU speeds it learned (gpu_profile.json) and start over."
        ),
    ] = False,
    skip_repeat_pairs: Annotated[
        Optional[bool],
        typer.Option(help="Send each pair to Hydrus once per run instead of twice (faster)."),
    ] = False,
    dry_run: Annotated[
        Optional[bool],
        typer.Option(help="Don't send anything to Hydrus; write found pairs to a file in the dedupe database folder."),
    ] = False,
    dedup_database_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="The directory to store the database used for dedupe.",
            # Show a generic label instead of the resolved path, which contains the Windows user name.
            show_default=r"%USERPROFILE%\AppData\Local",
        ),
    ] = config.dedupe_database_dir,
    verbose: Annotated[Optional[bool], typer.Option(help="Verbose logging")] = False,
    debug: Annotated[Optional[bool], typer.Option(hidden=True)] = False,
    gui: Annotated[  # This gui arg is unused, it's only here to show on --help. See run_main() for more info.
        Optional[bool], typer.Option(help="Launch the GUI. This ignores all other CLI arguments.")
    ] = config.hvd_gui,
):
    # Fix mypy errors from optional parameters
    assert threshold is not None and skip_hashing is not None and job_count is not None

    # CLI debug parameter sets log level to info or debug
    loglevel = logging.INFO
    if debug:
        loglevel = logging.DEBUG
        verbose = True

    logging.basicConfig(
        format=" %(asctime)s - %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=loglevel,
        handlers=[TqdmLoggingHandler()],
    )
    logger = logging.getLogger("main")
    logger.debug("Starting Hydrus Video Deduplicator.")

    def exit_from_failure() -> NoReturn:
        print_and_log(logger, "Exiting due to failure...")
        raise typer.Exit(code=1)

    # Verbose sets whether logs are shown to the user at all.
    # Logs are separate from printing in this program.
    if not verbose:
        logging.disable()

    DedupeDB.set_db_dir(dedup_database_dir)

    # Print a warning if the deprecated overwrite option is set
    if overwrite is not None:
        pretty_overwrite = "--" + ("" if overwrite is True else "no-") + "overwrite"
        print_and_log(
            logger,
            f"WARNING: '{pretty_overwrite}' option was deprecated and does nothing as of 0.7.0. Remove it from your args.",  # noqa: E501
        )

    # CLI overwrites env vars with no default value
    if not api_key:
        api_key = config.hydrus_api_key

    # Check for necessary variables
    if not api_key:
        print_and_log(logger, "Hydrus API key is not set. Please set with '--api-key'.")
        exit_from_failure()
    # This should not happen because there's a default val in config.py
    if not api_url:
        print_and_log(logger, "Hydrus API URL is not set. Please set with '--api-url'.")
        exit_from_failure()

    if clear_search_tree and relink_search_tree:
        print_and_log(logger, "Use either --clear-search-tree or --relink-search-tree, not both.", logging.ERROR)
        exit_from_failure()

    if threshold < 0.0 or threshold > 100.0:
        print_and_log(logger, f"Invalid similarity threshold: {threshold}. Must be between 0 and 100.", logging.ERROR)
        exit_from_failure()

    if config.gpu_decode_invalid is not None:
        console_print(
            style_user_text(
                f"WARNING: Unknown GPU_DECODE value '{config.gpu_decode_invalid}', so GPU decode is off. "
                "Use auto, all or off."
            ),
            highlight=False,
        )
        print()

    # Client connection
    print_and_log(logger, f"Connecting to Hydrus at {api_url}")
    try:
        hvdclient = create_client(
            file_service_key,
            api_url,
            api_key,
            verify_cert,
        )
        api_version = hvdclient.get_api_version()
        hydrus_api_version = hvdclient.get_hydrus_api_version()
        print_and_log(logger, f"Dedupe API version: 'v{api_version}'")
        print_and_log(logger, f"Hydrus API version: 'v{hydrus_api_version}'")
        hvdclient.verify_permissions()
    except (FailedHVDClientConnection, ClientAPIException) as exc:
        print_and_log(logger, f"{exc.pretty_msg} \nException: {exc}", logging.FATAL)
        exit_from_failure()

    if debug:
        HVDClient._log.setLevel(logging.DEBUG)

    # Deduplication

    if prune_deleted_files or prune_orphans or relink_search_tree or rehash_overhashed:
        backup = True  # pruning, relinking and re-hashing always back up first

    if DedupeDB.does_db_exist():
        print_and_log(logger, f"Found existing database at: '{display_path(DedupeDB.get_db_file_path())}'")
        if backup:
            # Before anything (including a database upgrade) changes the database.
            try:
                run_backup()
            except Exception as exc:
                print_and_log(logger, f"ERROR: Backing up the database failed: {exc}", logging.ERROR, highlight=False)
                exit_from_failure()
        db = DedupeDB.DedupeDb(DedupeDB.get_db_dir(), DedupeDB.get_db_name())
        db.init_connection()
        # Upgrade the database before doing anything.
        db.begin_transaction()
        db_upgraded = False
        with db.conn:
            db_upgraded = db.upgrade_db()
        # Vacuum DB after a successful database upgrade. This can reduce space by 1/2 in cases of large
        # db migrations.
        if db_upgraded:
            print_and_log(
                logger,
                "Database upgraded, vacuuming to save space.",
            )
            db_stats = DedupeDB.get_db_stats(db)
            print_and_log(
                logger,
                f"Database filesize before vacuum: {db_stats.file_size} bytes.",
            )
            db.vacuum()
            db_stats = DedupeDB.get_db_stats(db)
            print_and_log(
                logger,
                f"Database filesize after vacuum: {db_stats.file_size} bytes.",
            )
        db_stats = DedupeDB.get_db_stats(db)

        print_and_log(
            logger,
            f"Database has {db_stats.num_videos} videos already perceptually hashed.",
        )
        print_and_log(
            logger,
            f"Database filesize: {db_stats.file_size} bytes.",
        )
        # Read only. Each check is skipped when the flag that fixes it is given.
        check_database(db, check_orphans=not prune_orphans, check_tree=not (relink_search_tree or clear_search_tree))
        print_queue_waiting(db)
        print_search_cache_warning()

        if prune_orphans:
            # First, so the steps below only see valid rows.
            run_prune_orphans(db)

        if prune_deleted_files:
            run_prune(db, hvdclient)

        if rehash_overhashed:
            run_rehash_overhashed(db, hvdclient, job_count)

        if clear_search_tree:
            db.begin_transaction()
            with db.conn:
                db.clear_search_tree()
            print("[green] Cleared the search tree.")

        if relink_search_tree:
            print("[blue] Relinking the search tree. This only takes a moment...")
            db.begin_transaction()
            with db.conn:  # all-or-nothing: if anything fails or is interrupted, the DB is left unchanged
                relink = db.relink_search_tree()
            print(
                f"[green] Relinked the search tree: {relink['tree_nodes']} perceptual hashes, "
                f"{relink['files_to_search']} files will be searched for duplicates again."
            )
            skipped = relink["skipped_missing_phash"] + relink["skipped_missing_file"]
            if skipped:
                print(
                    f"[yellow] Skipped {skipped} file entries whose perceptual hash or file record is missing "
                    "(they were already unusable; nothing was deleted)."
                )

        if clear_search_cache:
            db.begin_transaction()
            with db.conn:
                db.clear_search_cache()
            print("[green] Cleared the search cache.")

    else:
        print_and_log(
            logger, f"Database not found. Creating one at '{display_path(DedupeDB.get_db_file_path())}'", logging.INFO
        )
        if not DedupeDB.get_db_dir().exists():
            DedupeDB.create_db_dir()
        db = DedupeDB.DedupeDb(DedupeDB.get_db_dir(), DedupeDB.get_db_name())
        db.init_connection()
        db.begin_transaction()
        with db.conn:
            db.create_tables()
        db_stats = DedupeDB.get_db_stats(db)
        print_search_cache_warning()

    deduper = HydrusVideoDeduplicator(
        db,
        client=hvdclient,
        similarity_threshold=threshold,
        job_count=job_count,
        failed_page_name=failed_page_name,
        custom_query=query,
        gpu_decode=resolve_gpu_decode(gpu_decode),
        skip_repeat_pairs=bool(skip_repeat_pairs),
        reset_gpu_profile=bool(reset_gpu_profile),
        dry_run=bool(dry_run),
    )

    if debug:
        deduper.hydlog.setLevel(logging.DEBUG)
        deduper._DEBUG = True

    num_similar_pairs = deduper.deduplicate(
        skip_hashing=skip_hashing,
    )

    db.close()

    return num_similar_pairs


def run_main(gui: bool, is_executable: bool):
    print(f"[blue]Hydrus Video Deduplicator {__version__} (optimizations fork) [/]\n")
    if loaded_dotenv:
        # Note: This doesn't mean any values were actually loaded from the .env file.
        print(f"Config file found: '{display_path(find_dotenv())}'")
    if gui or bool(int(config.hvd_gui)):
        try:
            from hydrusvideodeduplicator.gui.gui import gui_main
        except ImportError as exc:
            print_and_log(
                logging.getLogger("main"),
                f"Failed to import GUI dependencies. Did you install the GUI dependencies? Error: {exc}",  # noqa: E501
                logging.FATAL,
            )
            raise typer.Exit(code=1) from exc

        gui_main(config)
    else:
        # Allow launching the GUI via CLI using --gui. This is required before the typer.run()
        # because the typer checks if arguments exist, but we don't need any of the args for the GUI
        # to run.
        if len(sys.argv) > 1 and sys.argv[1] == "--gui":
            try:
                from hydrusvideodeduplicator.gui.gui import gui_main
            except ImportError as exc:
                print_and_log(
                    logging.getLogger("main"),
                    f"Failed to import GUI dependencies. Did you install the GUI dependencies? Error: {exc}",  # noqa: E501
                    logging.FATAL,
                )
                raise typer.Exit(code=1) from exc

            gui_main(config)
        else:
            try:
                # Same as typer.run(main), plus the help text and the optimization switches at the end of --help.
                app = typer.Typer(add_completion=False)
                app.command(help=CLI_HELP, epilog=CLI_EPILOG)(main)
                try:
                    args = normalize_gpu_decode_args(sys.argv[1:])
                except GpuDecodeArgError as exc:
                    console_print(style_user_text(str(exc)), highlight=False)
                    raise SystemExit(2) from exc
                app(args=args)
            except KeyboardInterrupt as exc:
                raise typer.Exit(-1) from exc
            finally:
                if is_executable:
                    # Hang the console window for the CLI exe, because 99% of users will be running this
                    # interactively and will want this pause to see errors/the final results. The other 1% should file
                    # a Github issue if this causes them issues and they want some --no-interactive option, or they should just # noqa: E501
                    # run the Python install.
                    input("Press ENTER to exit...")


if __name__ == "__main__":
    run_main(gui=False)
