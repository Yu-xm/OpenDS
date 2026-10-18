#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 1 && "$1" == "--version" ]]; then
  exec /usr/local/bin/yt-dlp-real --version
fi

readonly endpoint="http://127.0.0.1:${BGUTIL_POT_PORT:-4416}"
for attempt in $(seq 1 20); do
  if curl --noproxy '*' --fail --silent "${endpoint}/ping" >/dev/null; then
    break
  fi
  if [[ "${attempt}" -eq 20 ]]; then
    echo "PO token provider is not ready" >&2
    exit 70
  fi
  sleep 0.25
done

exec /usr/local/bin/yt-dlp-real \
  --js-runtimes node \
  --extractor-args "youtube:player_client=mweb" \
  --extractor-args "youtubepot-bgutilhttp:base_url=${endpoint}" \
  "$@"
