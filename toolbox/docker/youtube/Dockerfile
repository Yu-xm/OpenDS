ARG NODE_BASE_IMAGE=node:22.23.0-bookworm-slim
ARG SANDBOX_BASE_IMAGE=ghcr.io/agent-infra/sandbox:1.11.0
FROM ${NODE_BASE_IMAGE} AS pot-build

ARG BGUTIL_VERSION=1.3.2
WORKDIR /src
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl \
    && curl -fsSL "https://github.com/Brainicism/bgutil-ytdlp-pot-provider/archive/refs/tags/${BGUTIL_VERSION}.tar.gz" -o /tmp/provider.tar.gz \
    && tar -xzf /tmp/provider.tar.gz --strip-components=1 \
    && cd server \
    && sed -i 's/host: "::"/host: "127.0.0.1"/' src/main.ts \
    && sed -i 's/host: "0.0.0.0"/host: "127.0.0.1"/' src/main.ts \
    && npm ci --no-audit --no-fund \
    && npx tsc \
    && npm prune --omit=dev --no-audit --no-fund

FROM ${SANDBOX_BASE_IMAGE}

ENV AUTOSTART_BGUTIL_POT=true \
    BGUTIL_POT_PORT=4416

COPY --from=pot-build /src/server/build /opt/toolbox/bgutil/build
COPY --from=pot-build /src/server/node_modules /opt/toolbox/bgutil/node_modules
COPY --from=pot-build /src/server/package.json /opt/toolbox/bgutil/package.json
COPY docker/youtube/po_token_provider.conf /opt/gem/supervisord/po-token-provider.conf
COPY docker/youtube/po_token_wrapper.sh /usr/local/bin/po-token-wrapper
COPY docker/youtube/range_cache_runtime.py /opt/toolbox/range_cache_runtime.py

RUN python3 -m pip install --no-cache-dir "yt-dlp[default]==2026.8.19" \
    && python3 -m pip install --no-cache-dir --no-deps "bgutil-ytdlp-pot-provider==1.3.2" \
    && mv /usr/local/bin/yt-dlp /usr/local/bin/yt-dlp-real \
    && chmod 0755 /usr/local/bin/po-token-wrapper /opt/toolbox/range_cache_runtime.py \
    && ln -s /usr/local/bin/po-token-wrapper /usr/local/bin/yt-dlp

LABEL org.opencontainers.image.title="Toolbox YouTube seeker"
