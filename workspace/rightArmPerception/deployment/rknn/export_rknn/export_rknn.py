
from __future__ import annotations

__version__ = "1.2.0"

import argparse
import copy
import inspect
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml



RKNN_CONFIG_KEYS = [
    "mean_values",
    "std_values",
    "quantized_dtype",
    "quantized_algorithm",
    "quantized_method",
    "quantized_hybrid_level",
    "target_platform",
    "quant_img_RGB2BGR",
    "float_dtype",
    "optimization_level",
    "custom_string",
    "remove_weight",
    "compress_weight",
    "inputs_yuv_fmt",
    "single_core_mode",
    "dynamic_input",
    "model_pruning",
    "op_target",
    "quantize_weight",
    "remove_reshape",
    "sparse_infer",
    "enable_flash_attention",
    "auto_hybrid_cos_thresh",
    "auto_hybrid_euc_thresh",
]




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a source model to RKNN format using a YAML config file. "
            "Run from the conda/virtualenv that has rknn-toolkit2 installed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python export_rknn.py --config my_model.yaml\n"
            "  python export_rknn.py --config my_model.yaml --print-config\n"
            "  python export_rknn.py --config my_model.yaml "
            "--set build.do_quantization=false\n"
        ),
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help=(
            "Override config values using dotted keys. "
            "e.g. --set model.model_path=/new/model.onnx "
            "--set build.do_quantization=false"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and validate config plus input paths without calling RKNN.",
    )
    parser.add_argument(
        "--print-config", action="store_true",
        help="Print the fully resolved config before running the export.",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser.parse_args()



def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return the top-level mapping."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(data).__name__}: {path}")
    return data


def resolve_path(path_value: str | None, base_dir: Path) -> str | None:
    """Resolve a possibly-relative path against *base_dir*."""
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def ensure_parent_dir(path_value: str) -> None:
    """Create parent directories for *path_value* if they don't exist."""
    Path(path_value).parent.mkdir(parents=True, exist_ok=True)


def as_list(value: Any) -> list[Any]:
    """Wrap a scalar in a list; pass through lists and empty values."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def resolve_existing_path(path_value: str | None, base_dir: Path, label: str) -> str:
    """Resolve a path and ensure it exists."""
    resolved = resolve_path(path_value, base_dir)
    if not resolved:
        raise ValueError(f"{label} is required.")
    if not Path(resolved).exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def resolve_input_paths(paths: Any, base_dir: Path, label: str) -> list[str]:
    """Resolve a config list/scalar of file paths and ensure every file exists."""
    resolved_paths = []
    for path_value in as_list(paths):
        resolved_paths.append(resolve_existing_path(str(path_value), base_dir, label))
    return resolved_paths


# ===== Config override helpers =============================================

def parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Invalid override, expected KEY=VALUE: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override key cannot be empty: {raw}")
    return key, yaml.safe_load(value)


def set_nested_value(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if next_value is None:
            next_value = {}
            cursor[part] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(
                f"Cannot set nested key '{dotted_key}': '{part}' is not a mapping."
            )
        cursor = next_value
    cursor[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    for raw in overrides:
        key, value = parse_override(raw)
        set_nested_value(resolved, key, value)
    return resolved


# ===== Dataset manifest ====================================================

def _load_dataset_entries(manifest_path: Path) -> list[tuple[int, list[str]]]:
    """Read non-empty dataset rows, preserving source line numbers."""
    entries: list[tuple[int, list[str]]] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        raw_entry = line.strip()
        if not raw_entry or raw_entry.startswith("#"):
            continue
        entry_parts = raw_entry.split()
        if entry_parts:
            entries.append((line_number, entry_parts))
    return entries


def declared_model_input_count(model_cfg: dict[str, Any]) -> int | None:
    """Return the configured model input count when the YAML declares it."""
    declared_counts = {}
    for key in ("inputs", "input_size_list"):
        value = model_cfg.get(key)
        if isinstance(value, list) and value:
            declared_counts[key] = len(value)
    if len(set(declared_counts.values())) > 1:
        details = ", ".join(
            f"{key}={count}" for key, count in sorted(declared_counts.items())
        )
        raise ValueError(f"Model input declaration count mismatch: {details}.")
    if declared_counts:
        return next(iter(declared_counts.values()))
    return None


def resolve_dataset(
    dataset_path: str | None,
    config_dir: Path,
    *,
    materialize: bool = False,
    expected_input_count: int | None = None,
) -> tuple[str | None, int, str | None]:
    """
    Resolve the calibration-dataset manifest file (.txt).

    The manifest is a plain-text file with one sample per line. Single-input
    models use one file per line; multi-input models use whitespace-separated
    files per line, ordered exactly as model.inputs/input_size_list.

    Returns (manifest_path_for_rknn, sample_count, source_manifest_path).
    """
    if not dataset_path:
        return None, 0, None

    manifest_path = Path(
        resolve_existing_path(str(dataset_path), config_dir, "Dataset manifest")
    )
    resolved_rows: list[list[str]] = []
    for line_number, row_parts in _load_dataset_entries(manifest_path):
        if (
            expected_input_count is not None
            and len(row_parts) != expected_input_count
        ):
            raise ValueError(
                f"Dataset row input count mismatch at {manifest_path}:{line_number}: "
                f"expected {expected_input_count}, got {len(row_parts)}."
            )

        resolved_row = []
        for raw_part in row_parts:
            sample_path = Path(raw_part).expanduser()
            if not sample_path.is_absolute():
                sample_path = (manifest_path.parent / sample_path).resolve()
            if not sample_path.exists():
                raise FileNotFoundError(
                    f"Dataset sample not found at {manifest_path}:{line_number}: {sample_path}"
                )
            resolved_row.append(str(sample_path))
        resolved_rows.append(resolved_row)

    if not materialize:
        return str(manifest_path), len(resolved_rows), str(manifest_path)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"{manifest_path.stem}_resolved_",
        suffix=".txt",
        delete=False,
    ) as temp_manifest:
        if resolved_rows:
            temp_manifest.write("\n".join(" ".join(row) for row in resolved_rows))
            temp_manifest.write("\n")
        return temp_manifest.name, len(resolved_rows), str(manifest_path)


# ===== Model loading =======================================================

def load_initial_values(
    files: Any, config_dir: Path
) -> list[Any] | None:
    """Load numpy arrays listed in ``model.input_initial_val_files``."""
    resolved_files = resolve_input_paths(files, config_dir, "Input initial value file")
    if not resolved_files:
        return None
    import numpy as np

    values = []
    for resolved_path in resolved_files:
        values.append(np.load(resolved_path))
    return values


def call_with_supported_kwargs(func: Any, kwargs: dict[str, Any], call_name: str) -> Any:
    """
    Call an RKNN API while ignoring kwargs unsupported by the installed toolkit.

    This keeps configs portable across RKNN versions whose Python signatures
    differ slightly.
    """
    supported_kwargs = set(inspect.signature(func).parameters)
    dropped_keys = sorted(
        key for key, value in kwargs.items()
        if key not in supported_kwargs and value is not None
    )
    filtered_kwargs = {
        key: value for key, value in kwargs.items()
        if key in supported_kwargs
    }
    if dropped_keys:
        print(
            f"[warn] Ignoring unsupported {call_name} kwargs for this "
            f"rknn-toolkit2 build: {', '.join(dropped_keys)}"
        )
    return func(**filtered_kwargs)


def collect_rknn_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract only the keys recognised by ``rknn.config()``, skipping None values."""
    kwargs: dict[str, Any] = {}
    for key in RKNN_CONFIG_KEYS:
        if key in config and config[key] is not None:
            kwargs[key] = config[key]
    return kwargs


def load_model_into_rknn(
    rknn: Any, model_cfg: dict[str, Any], config_dir: Path
) -> int:
    """Dispatch to the correct ``rknn.load_*`` method based on framework."""
    framework = str(model_cfg.get("framework", "onnx")).lower()
    model_path = resolve_path(model_cfg.get("model_path"), config_dir)
    if not model_path:
        raise ValueError("model.model_path is required.")

    inputs = model_cfg.get("inputs")
    outputs = model_cfg.get("outputs")
    input_size_list = model_cfg.get("input_size_list")
    input_is_nchw = bool(model_cfg.get("input_is_nchw", False))
    input_initial_val = load_initial_values(
        model_cfg.get("input_initial_val_files"), config_dir
    )

    if framework == "onnx":
        return call_with_supported_kwargs(
            rknn.load_onnx,
            {
                "model": model_path,
                "inputs": inputs,
                "input_size_list": input_size_list,
                "input_initial_val": input_initial_val,
                "input_is_nchw": input_is_nchw,
                "outputs": outputs,
            },
            "load_onnx",
        )
    if framework == "pytorch":
        if not input_size_list:
            raise ValueError("model.input_size_list is required for pytorch.")
        return rknn.load_pytorch(model=model_path, input_size_list=input_size_list)

    raise ValueError(f"Unsupported model.framework: {framework} (only onnx and pytorch are supported)")


# ===== Validation & summary ================================================

def validate_source_paths(config: dict[str, Any], config_dir: Path) -> None:
    """Ensure that all referenced source files actually exist on disk."""
    model_cfg = config.get("model", {})
    export_cfg = config.get("export", {})
    resolve_existing_path(model_cfg.get("model_path"), config_dir, "Model file")
    resolve_input_paths(model_cfg.get("input_initial_val_files"), config_dir, "Input initial value file")

    accuracy_cfg = export_cfg.get("accuracy_analysis") or {}
    resolve_input_paths(accuracy_cfg.get("inputs"), config_dir, "Accuracy analysis input")


def summarize_config(
    config: dict[str, Any],
    config_path: Path,
    config_dir: Path,
) -> dict[str, Any]:
    """Build a summary dict that includes resolved absolute paths."""
    model_cfg = config.get("model", {})
    export_cfg = config.get("export", {})
    build_cfg = config.get("build", {})

    output_path = resolve_path(export_cfg.get("output_path"), config_dir)
    if not output_path:
        raise ValueError("export.output_path is required.")

    validate_source_paths(config, config_dir)
    dataset_manifest, dataset_count, source_dataset_manifest = resolve_dataset(
        build_cfg.get("dataset"),
        config_dir,
        expected_input_count=declared_model_input_count(model_cfg),
    )

    do_quantization = build_cfg.get("do_quantization")
    if do_quantization is None:
        do_quantization = dataset_manifest is not None
    if do_quantization and not dataset_manifest:
        raise ValueError(
            "build.do_quantization is true but no dataset .txt file was provided."
        )
    if do_quantization and dataset_count <= 0:
        raise ValueError(
            "build.do_quantization is true but the dataset manifest is empty."
        )

    summary = copy.deepcopy(config)
    summary["meta"] = summary.get("meta", {})
    summary["meta"]["config_path"] = str(config_path.resolve())
    summary["meta"]["config_dir"] = str(config_dir.resolve())

    summary.setdefault("model", {})
    summary["model"]["resolved_model_path"] = resolve_path(
        model_cfg.get("model_path"), config_dir
    )


    summary.setdefault("build", {})
    summary["build"]["resolved_dataset_manifest"] = source_dataset_manifest
    summary["build"]["resolved_dataset_count"] = dataset_count
    summary["build"]["effective_do_quantization"] = bool(do_quantization)

    summary.setdefault("export", {})
    summary["export"]["resolved_output_path"] = output_path
    accuracy_cfg = export_cfg.get("accuracy_analysis") or {}
    resolved_accuracy_inputs = resolve_input_paths(
        accuracy_cfg.get("inputs"), config_dir, "Accuracy analysis input"
    )
    if resolved_accuracy_inputs:
        summary["export"]["resolved_accuracy_analysis_inputs"] = resolved_accuracy_inputs
    if accuracy_cfg.get("output_dir"):
        summary["export"]["resolved_accuracy_analysis_output_dir"] = resolve_path(
            accuracy_cfg.get("output_dir"), config_dir
        )

    return summary


# ===== Export pipeline =====================================================

def run_export(config: dict[str, Any], config_dir: Path) -> str:
    """Execute the full RKNN export pipeline: config → load → build → export."""
    from rknn.api import RKNN

    model_cfg = config.get("model", {})
    rknn_config = config.get("rknn_config", {})
    build_cfg = config.get("build", {})
    export_cfg = config.get("export", {})

    output_path = resolve_path(export_cfg.get("output_path"), config_dir)
    if not output_path:
        raise ValueError("export.output_path is required.")
    ensure_parent_dir(output_path)

    dataset_manifest, dataset_count, source_dataset_manifest = resolve_dataset(
        build_cfg.get("dataset"),
        config_dir,
        materialize=True,
        expected_input_count=declared_model_input_count(model_cfg),
    )
    temp_dataset_manifest = None
    if dataset_manifest and source_dataset_manifest and dataset_manifest != source_dataset_manifest:
        temp_dataset_manifest = dataset_manifest

    do_quantization = build_cfg.get("do_quantization")
    if do_quantization is None:
        do_quantization = dataset_manifest is not None
    if do_quantization and not dataset_manifest:
        raise ValueError(
            "build.do_quantization is true but no dataset .txt file was provided."
        )
    if do_quantization and dataset_count <= 0:
        raise ValueError(
            "build.do_quantization is true but the dataset manifest is empty."
        )

    verbose = config.get("meta", {}).get("verbose", False)
    verbose_file = config.get("meta", {}).get("verbose_file")
    if verbose_file:
        verbose_file = resolve_path(str(verbose_file), config_dir)
        assert verbose_file is not None
        ensure_parent_dir(verbose_file)

    print("[1/5] Creating RKNN object")
    rknn = RKNN(verbose=verbose, verbose_file=verbose_file)
    try:
        print("[2/5] Configuring RKNN")
        ret = rknn.config(**collect_rknn_config(rknn_config))
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: ret={ret}")

        print("[3/5] Loading source model")
        ret = load_model_into_rknn(rknn, model_cfg, config_dir)
        if ret != 0:
            raise RuntimeError(f"Model loading failed: ret={ret}")

        print(
            f"[4/5] Building RKNN "
            f"(quantization={bool(do_quantization)}, samples={dataset_count})"
        )
        ret = rknn.build(
            do_quantization=bool(do_quantization),
            dataset=dataset_manifest,
            rknn_batch_size=build_cfg.get("rknn_batch_size"),
            auto_hybrid=bool(build_cfg.get("auto_hybrid", False)),
        )
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: ret={ret}")

        print(f"[5/5] Exporting RKNN → {output_path}")
        ret = rknn.export_rknn(output_path)
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: ret={ret}")

        # Optional: accuracy analysis
        accuracy_cfg = export_cfg.get("accuracy_analysis")
        if accuracy_cfg:
            print("[+] Running accuracy analysis")
            acc_inputs_resolved = resolve_input_paths(
                accuracy_cfg.get("inputs"), config_dir, "Accuracy analysis input"
            )
            if acc_inputs_resolved:
                acc_out_dir = resolve_path(
                    accuracy_cfg.get("output_dir") or "./accuracy_analysis",
                    config_dir,
                )
                ensure_parent_dir(acc_out_dir + "/dummy")
                ret = rknn.accuracy_analysis(
                    inputs=acc_inputs_resolved, output_dir=acc_out_dir
                )
                if ret != 0:
                    raise RuntimeError(
                        f"rknn.accuracy_analysis failed: ret={ret}"
                    )
            else:
                print("    Skipped: no inputs provided for accuracy_analysis.")
    finally:
        try:
            rknn.release()
        except Exception:
            pass
        if temp_dataset_manifest:
            try:
                Path(temp_dataset_manifest).unlink(missing_ok=True)
            except Exception:
                pass
    return output_path


# ===== Entrypoint ==========================================================

def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config_dir = config_path.parent

    config = load_yaml(config_path)
    config = apply_overrides(config, args.set)

    summary = summarize_config(
        config=config,
        config_path=config_path,
        config_dir=config_dir,
    )

    if args.print_config or args.dry_run:
        print(yaml.safe_dump(summary, allow_unicode=True))

    if args.dry_run:
        print("Dry run finished. RKNN export was not executed.")
        return 0

    output_path = run_export(config, config_dir)
    print(json.dumps(
        {"status": "ok", "output_path": output_path},
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[export_rknn] ERROR: {exc}", file=sys.stderr)
        raise
