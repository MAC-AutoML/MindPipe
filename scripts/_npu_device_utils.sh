# shellcheck shell=bash

MINDPIPE_NPU_ID_MAP_INITIALIZED="${MINDPIPE_NPU_ID_MAP_INITIALIZED:-0}"
MINDPIPE_NPU_LOGICAL_COUNT="${MINDPIPE_NPU_LOGICAL_COUNT:-}"
MINDPIPE_NPU_VISIBLE_HARDWARE_IDS_CSV="${MINDPIPE_NPU_VISIBLE_HARDWARE_IDS_CSV:-}"
MINDPIPE_NPU_ID_MAP_ERROR="${MINDPIPE_NPU_ID_MAP_ERROR:-}"


mindpipe_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s' "$PYTHON_BIN"
    return 0
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    printf '%s' "${CONDA_PREFIX}/bin/python"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  printf 'python'
}


mindpipe_query_npu_logical_count() {
  local python_bin
  python_bin="$(mindpipe_python_bin)"
  "$python_bin" -c 'import importlib.util; assert importlib.util.find_spec("torch") and importlib.util.find_spec("torch_npu"); import torch, torch_npu; print(torch.npu.device_count())' 2>/dev/null
}


mindpipe_query_visible_hardware_ids() {
  if ! command -v npu-smi >/dev/null 2>&1; then
    return 0
  fi

  local line
  local hardware_id
  local -A seen_ids=()
  while IFS= read -r line; do
    if [[ "$line" =~ ^\|[[:space:]]*([0-9]+)[[:space:]]+[^|[:space:]][^|]*\| ]]; then
      hardware_id="${BASH_REMATCH[1]}"
      if [[ -z "${seen_ids[$hardware_id]+x}" ]]; then
        printf '%s\n' "$hardware_id"
        seen_ids["$hardware_id"]=1
      fi
    fi
  done < <(npu-smi info 2>/dev/null || true)
}


mindpipe_init_npu_id_map() {
  if [[ "$MINDPIPE_NPU_ID_MAP_INITIALIZED" == "1" ]]; then
    if [[ -z "$MINDPIPE_NPU_ID_MAP_ERROR" ]]; then
      return 0
    fi
    return 1
  fi

  MINDPIPE_NPU_ID_MAP_INITIALIZED=1
  MINDPIPE_NPU_LOGICAL_COUNT=""
  MINDPIPE_NPU_VISIBLE_HARDWARE_IDS_CSV=""
  MINDPIPE_NPU_ID_MAP_ERROR=""

  local logical_count
  logical_count="$(mindpipe_query_npu_logical_count | tr -d '[:space:]')"
  if [[ ! "$logical_count" =~ ^[0-9]+$ ]]; then
    MINDPIPE_NPU_ID_MAP_ERROR="failed to query torch_npu logical device count using $(mindpipe_python_bin)"
    return 1
  fi
  MINDPIPE_NPU_LOGICAL_COUNT="$logical_count"

  local -a hardware_ids=()
  local hardware_id
  while IFS= read -r hardware_id; do
    [[ -n "$hardware_id" ]] || continue
    hardware_ids+=("$hardware_id")
  done < <(mindpipe_query_visible_hardware_ids)

  if (( ${#hardware_ids[@]} > 0 )); then
    printf -v MINDPIPE_NPU_VISIBLE_HARDWARE_IDS_CSV '%s,' "${hardware_ids[@]}"
    MINDPIPE_NPU_VISIBLE_HARDWARE_IDS_CSV="${MINDPIPE_NPU_VISIBLE_HARDWARE_IDS_CSV%,}"
  fi

  return 0
}


mindpipe_resolve_visible_device_id() {
  local requested_id="$1"
  if [[ ! "$requested_id" =~ ^[0-9]+$ ]]; then
    MINDPIPE_NPU_ID_MAP_ERROR="unsupported NPU identifier \`$requested_id\`; expected an integer device id"
    return 1
  fi

  mindpipe_init_npu_id_map || return 1

  local logical_count="$MINDPIPE_NPU_LOGICAL_COUNT"
  if (( requested_id >= 0 && requested_id < logical_count )); then
    printf '%s' "$requested_id"
    return 0
  fi

  local -a hardware_ids=()
  if [[ -n "$MINDPIPE_NPU_VISIBLE_HARDWARE_IDS_CSV" ]]; then
    IFS=',' read -r -a hardware_ids <<< "$MINDPIPE_NPU_VISIBLE_HARDWARE_IDS_CSV"
  fi

  if (( ${#hardware_ids[@]} == logical_count )) && (( logical_count > 0 )); then
    local index
    for index in "${!hardware_ids[@]}"; do
      if [[ "${hardware_ids[$index]}" == "$requested_id" ]]; then
        printf '%s' "$index"
        return 0
      fi
    done
  fi

  if (( logical_count == 0 )); then
    MINDPIPE_NPU_ID_MAP_ERROR="torch_npu reports zero visible devices"
  elif [[ -n "$MINDPIPE_NPU_VISIBLE_HARDWARE_IDS_CSV" ]]; then
    MINDPIPE_NPU_ID_MAP_ERROR="requested NPU id \`$requested_id\` is neither a logical id in [0,$((logical_count - 1))] nor one of the visible hardware ids [${MINDPIPE_NPU_VISIBLE_HARDWARE_IDS_CSV}]"
  else
    MINDPIPE_NPU_ID_MAP_ERROR="requested NPU id \`$requested_id\` is not a logical id in [0,$((logical_count - 1))]"
  fi
  return 1
}
