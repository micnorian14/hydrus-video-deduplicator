from __future__ import annotations

import json
import os
from pathlib import Path
from platform import uname

from platformdirs import PlatformDirs


def logical_cpu_count() -> int:
    """CPUs this process may use (respects affinity and containers where Python can tell)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def default_job_count(full_speed: bool = False, logical_cpus: int | None = None) -> int:
    """The --job-count used when it isn't set (CPU threads for decoding and hashing a video).

    Default mode: half the CPU threads, at least 1 (e.g. 12 -> 6, 4 -> 2, 1 -> 1).
    Full speed (--full-speed): 0, all CPU threads, like before.
    """
    if full_speed:
        return 0
    cpus = max(1, logical_cpus if logical_cpus is not None else logical_cpu_count())
    return max(1, cpus // 2)


def effective_job_count(job_count, full_speed: bool = False, logical_cpus: int | None = None) -> int:
    """--job-count / JOB_COUNT if set (in either mode), else the mode's default (default_job_count)."""
    if job_count is None or str(job_count).strip() == "":
        return default_job_count(full_speed, logical_cpus)
    return int(job_count)


def resolve_job_count(job_count: int, logical_cpus: int | None = None) -> int:
    """The number of threads a --job-count means: a positive number as is, 0 = all CPU threads, -n = all but n
    (at least 1). Same rules as vpdq's hashing threads (hvdaccelerators)."""
    cpus = max(1, logical_cpus if logical_cpus is not None else logical_cpu_count())
    job_count = int(job_count)
    if job_count > 0:
        return job_count
    if job_count == 0:
        return cpus
    return max(1, cpus + job_count)


class InvalidEnvironmentVariable(Exception):
    """Raise for when environment variables are invalid."""

    def __init__(self, msg):
        super().__init__(msg)
        print("Exiting due to invalid environment variable.")


# Validates that a env var is a valid JSON array
# Program will exit if invalid
# Returns the parsed json array as a list
def validate_json_array_env_var(env_var: str | None, err_msg: str) -> list | None:
    if env_var is None:
        return None

    try:
        env_var_list = json.loads(env_var)
        if not isinstance(env_var_list, list):
            raise InvalidEnvironmentVariable(f"ERROR: {err_msg}")
    except json.decoder.JSONDecodeError as exc:
        raise InvalidEnvironmentVariable(f"ERROR: {err_msg}") from exc

    return env_var_list


# --gpu-decode auto / GPU_DECODE=auto: each video is decoded on the GPU or the CPU, whichever is usually faster for it.
GPU_DECODE_AUTO = "auto"
# --gpu-decode all / GPU_DECODE=all: every video is decoded on the GPU. off: every video on the CPU.
GPU_DECODE_ALL = "all"
GPU_DECODE_OFF = "off"
# all/off plus older spellings that still work but aren't documented.
GPU_DECODE_ON_VALUES = (GPU_DECODE_ALL, "1", "true", "yes", "on")
GPU_DECODE_OFF_VALUES = (GPU_DECODE_OFF, "0", "false", "no", "")
GPU_DECODE_NEEDS_VALUE = (
    "--gpu-decode now needs a value. Use --gpu-decode auto (recommended) or --gpu-decode all.\n"
    "--gpu-decode all tries to force all files to decode on the GPU."
)


class GpuDecodeArgError(ValueError):
    """A --gpu-decode on the command line without a value, or with one it doesn't know."""


def gpu_decode_mode(value: str | None) -> str | None:
    """GPU_DECODE_AUTO, GPU_DECODE_ALL or GPU_DECODE_OFF for a --gpu-decode value (any case), or None if unknown."""
    value = str(value or "").strip().lower()
    if value == GPU_DECODE_AUTO:
        return GPU_DECODE_AUTO
    if value in GPU_DECODE_ON_VALUES:
        return GPU_DECODE_ALL
    if value in GPU_DECODE_OFF_VALUES and value:
        return GPU_DECODE_OFF
    return None


def _checked_gpu_decode_mode(value: str) -> str:
    mode = gpu_decode_mode(value)
    if mode is None:
        raise GpuDecodeArgError(f"Unknown --gpu-decode value '{value}'. Use auto (recommended), all, or off.")
    return mode


def normalize_gpu_decode_args(args: list[str]) -> list[str]:
    """
    Check --gpu-decode on the command line and turn its value into auto, all or off.

    '--gpu-decode VALUE' and '--gpu-decode=VALUE' become '--gpu-decode MODE' (any case; on/true/1/yes mean all and
    false/0/no mean off). The old '--no-gpu-decode' becomes '--gpu-decode off'. Raises GpuDecodeArgError for a
    --gpu-decode without a value or with an unknown one. The last --gpu-decode given wins, including over GPU_DECODE.
    Everything else is passed through unchanged.
    """
    out: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            out.extend(args[i:])
            break
        if arg.lower().startswith("--gpu-decode="):
            out.extend(["--gpu-decode", _checked_gpu_decode_mode(arg.split("=", 1)[1])])
            i += 1
            continue
        if arg == "--gpu-decode":
            if i + 1 >= len(args) or args[i + 1].startswith("-"):
                raise GpuDecodeArgError(GPU_DECODE_NEEDS_VALUE)
            out.extend(["--gpu-decode", _checked_gpu_decode_mode(args[i + 1])])
            i += 2
            continue
        if arg == "--no-gpu-decode":
            out.extend(["--gpu-decode", GPU_DECODE_OFF])
        else:
            out.append(arg)
        i += 1
    return out


def resolve_gpu_decode(mode: str | None) -> bool | str:
    """The gpu_decode value for HydrusVideoDeduplicator: GPU_DECODE_AUTO, True (all on the GPU) or False (CPU)."""
    mode = gpu_decode_mode(mode)
    if mode == GPU_DECODE_AUTO:
        return GPU_DECODE_AUTO
    return mode == GPU_DECODE_ALL


class Config:
    """Hydrus Video Deduplicator Config"""

    def __init__(self):
        self.hydrus_api_key = None
        self.hydrus_api_url = None
        self.dedupe_database_dir = None
        self.failed_page_name = "failed"
        self.requests_ca_bundle = None
        self.hydrus_query = None
        self.hydrus_local_file_service_keys = None
        self.similarity_threshold = None
        self.job_count = None
        self.hvd_gui = None
        self.gpu_decode = False
        self.gpu_decode_auto = False
        self.gpu_decode_invalid = None  # an unknown GPU_DECODE value, warned about once at startup

    @property
    def gpu_decode_mode(self) -> str:
        """The GPU_DECODE setting as a --gpu-decode value: auto, all or off."""
        if self.gpu_decode_auto:
            return GPU_DECODE_AUTO
        return GPU_DECODE_ALL if self.gpu_decode else GPU_DECODE_OFF

    @staticmethod
    def _load(config_map: dict[str, str | None]):
        config = Config()

        def _get_default_ip():
            default_ip = "localhost"

            # If you're in WSL you probably want to connect to your Windows Hydrus Client by default
            def in_wsl() -> bool:
                return "microsoft-standard" in uname().release

            if in_wsl():
                from socket import gethostname

                default_ip = f"{gethostname()}.local"
            return default_ip

        config.hydrus_api_url = config_map.get("HYDRUS_API_URL", f"http://{_get_default_ip()}:45869")
        config.hydrus_api_key = config_map.get("HYDRUS_API_KEY", None)

        # Default is:
        # Linux: ~/.local/share/hydrusvideodeduplicator/
        # Windows: C:\Users\%username%\AppData\Local\hydrusvideodeduplicator\hydrusvideodeduplicator
        config.dedupe_database_dir = Path(
            config_map.get("DEDUP_DATABASE_DIR", PlatformDirs("hydrusvideodeduplicator").user_data_dir)
        )

        config.failed_page_name = config_map.get("FAILED_PAGE_NAME", "failed")

        config.requests_ca_bundle = config_map.get("REQUESTS_CA_BUNDLE", None)

        # Optional query for selecting files to process
        # TODO: Should validation really be done here? What other config options can/should be validated?
        config.hydrus_query = validate_json_array_env_var(
            config_map.get("HYDRUS_QUERY", None), err_msg="Ensure HYDRUS_QUERY is a JSON formatted array."
        )

        # Optional service key of local file service/s to fetch files from
        config.hydrus_local_file_service_keys = validate_json_array_env_var(
            config_map.get("HYDRUS_LOCAL_FILE_SERVICE_KEYS"),
            err_msg="Ensure HYDRUS_LOCAL_FILE_SERVICE_KEYS is a JSON formatted array",
        )

        config.similarity_threshold = config_map.get("SIMILARITY_THRESHOLD", 50.0)

        # None: not set, the mode's default is used (see effective_job_count).
        config.job_count = config_map.get("JOB_COUNT", None)

        config.hvd_gui = config_map.get("HVD_GUI", False)

        # Use the NVIDIA GPU (CUDA/NVDEC) to decode videos. Hashes are identical for the codecs tested
        # (H.264, HEVC, VP8, VP9). MPEG-4 and WMV3 can differ by a few bits but still score as a full
        # match, so duplicate detection isn't affected.
        gpu_decode_setting = str(config_map.get("GPU_DECODE", "0")).strip().lower()
        config.gpu_decode = gpu_decode_setting in GPU_DECODE_ON_VALUES
        # GPU_DECODE=auto: decode each video on the GPU or the CPU, whichever is usually faster for it.
        config.gpu_decode_auto = gpu_decode_setting == GPU_DECODE_AUTO
        if gpu_decode_setting not in (*GPU_DECODE_ON_VALUES, *GPU_DECODE_OFF_VALUES, GPU_DECODE_AUTO):
            config.gpu_decode_invalid = str(config_map.get("GPU_DECODE")).strip()

        return config

    @staticmethod
    def load_from_env():
        """Load config options from environment variables."""
        config_map = {}
        config_options = [
            "HYDRUS_API_URL",
            "HYDRUS_API_KEY",
            "DEDUP_DATABASE_DIR",
            "FAILED_PAGE_NAME",
            "REQUESTS_CA_BUNDLE",
            "HYDRUS_QUERY",
            "HYDRUS_LOCAL_FILE_SERVICE_KEYS",
            "SIMILARITY_THRESHOLD",
            "JOB_COUNT",
            "HVD_GUI",
            "GPU_DECODE",
        ]
        for option in config_options:
            config_map[option] = os.getenv(option)

        # Remove all items so that _load() can populate them with defaults if they are missing.
        filtered_config_map = {k: v for k, v in config_map.items() if v is not None}
        return Config._load(filtered_config_map)
