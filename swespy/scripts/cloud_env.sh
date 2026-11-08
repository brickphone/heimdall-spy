# Sourced by the cloud scripts: load the pod's environment so MAPILLARY_ACCESS_TOKEN
# (injected by RunPod from the account secret) is available in SSH/tmux shells.
# Never prints the token.
if [ -f /etc/rp_environment ]; then set -a; . /etc/rp_environment; set +a; fi
if [ -z "${MAPILLARY_ACCESS_TOKEN:-}" ] && [ -r /proc/1/environ ]; then
  MAPILLARY_ACCESS_TOKEN="$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^MAPILLARY_ACCESS_TOKEN=//p')"
  export MAPILLARY_ACCESS_TOKEN
fi
export UV_PYTHON_DOWNLOADS=never   # Use the image's Python 3.12; never fetch one from GitHub.
