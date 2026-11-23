from __future__ import annotations

import json
import os
from pathlib import Path
from platform import uname

from platformdirs import PlatformDirs


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
GPU_DECODE_ON_VALUES = ("1", "true", "yes", "on")
GPU_DECODE_OFF_VALUES = ("0", "false", "no", "off", "")


def _gpu_decode_value_args(value: str) -> list[str] | None:
    """The flags for '--gpu-decode VALUE', or None if VALUE isn't one."""
    value = value.lower()
    if value == GPU_DECODE_AUTO:
        return ["--gpu-decode-auto"]
    if value in GPU_DECODE_ON_VALUES:
        return ["--gpu-decode", "--no-gpu-decode-auto"]
    if value in GPU_DECODE_OFF_VALUES and value:
        return ["--no-gpu-decode", "--no-gpu-decode-auto"]
    return None


def normalize_gpu_decode_args(args: list[str]) -> list[str]:
    """
    Let --gpu-decode take an optional value without changing how the flag already works.

    '--gpu-decode auto' and '--gpu-decode=auto' become the hidden '--gpu-decode-auto' flag. on/true/1/yes mean
    '--gpu-decode' and off/false/0/no mean '--no-gpu-decode' (any case). A plain '--gpu-decode' or '--no-gpu-decode'
    also gets '--no-gpu-decode-auto', so on the command line the last one given wins, including over GPU_DECODE=auto.
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
            mapped = _gpu_decode_value_args(arg.split("=", 1)[1])
            if mapped is not None:
                out.extend(mapped)
                i += 1
                continue
        if arg == "--gpu-decode" and i + 1 < len(args):
            mapped = _gpu_decode_value_args(args[i + 1])
            if mapped is not None:
                out.extend(mapped)
                i += 2
                continue
        out.append(arg)
        if arg in ("--gpu-decode", "--no-gpu-decode"):
            out.append("--no-gpu-decode-auto")
        i += 1
    return out


def resolve_gpu_decode(gpu_decode: bool | None, gpu_decode_auto: bool | None) -> bool | str:
    """The gpu_decode value for HydrusVideoDeduplicator: GPU_DECODE_AUTO, True (always GPU) or False (CPU)."""
    if gpu_decode_auto:
        return GPU_DECODE_AUTO
    return bool(gpu_decode)


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

        config.job_count = config_map.get("JOB_COUNT", -2)

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
