#!/usr/bin/env bash
set -euo pipefail

CONNECTION_NAME="${UR3E_ETH_CONNECTION:-Wired connection 1}"
INTERFACE_NAME="${UR3E_ETH_INTERFACE:-enx00e04c3211b0}"
PC_ADDRESS="${UR3E_PC_ADDRESS:-192.168.0.3/24}"
ROBOT_IP="${UR3E_ROBOT_IP:-192.168.0.5}"

COMMANDS=(
  "nmcli connection modify '${CONNECTION_NAME}' connection.interface-name '${INTERFACE_NAME}'"
  "nmcli connection modify '${CONNECTION_NAME}' ipv4.method manual ipv4.addresses '${PC_ADDRESS}' ipv4.gateway '' ipv4.never-default yes ipv4.dns ''"
  "nmcli connection up '${CONNECTION_NAME}'"
  "ip route get '${ROBOT_IP}'"
  "ping -c 3 '${ROBOT_IP}'"
)

if [[ "${1:-}" != "--apply" ]]; then
  echo "Dry run. Re-run with --apply to configure the UR3e wired connection."
  echo
  printf '%s\n' "${COMMANDS[@]}"
  exit 0
fi

nmcli connection modify "${CONNECTION_NAME}" connection.interface-name "${INTERFACE_NAME}"
nmcli connection modify "${CONNECTION_NAME}" \
  ipv4.method manual \
  ipv4.addresses "${PC_ADDRESS}" \
  ipv4.gateway "" \
  ipv4.never-default yes \
  ipv4.dns ""
nmcli connection up "${CONNECTION_NAME}"
ip route get "${ROBOT_IP}"
ping -c 3 "${ROBOT_IP}"
