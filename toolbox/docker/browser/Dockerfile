ARG SANDBOX_BASE_IMAGE=ghcr.io/agent-infra/sandbox:1.11.0
FROM ${SANDBOX_BASE_IMAGE}

LABEL org.opencontainers.image.title="Toolbox browser"

# Disable the inline PDF viewer in this disposable evaluation browser only.
# Cover both Google Chrome and distro Chromium policy locations.
RUN mkdir -p /etc/opt/chrome/policies/managed /etc/chromium/policies/managed \
    && printf '%s\n' '{"AlwaysOpenPdfExternally":true}' \
       > /etc/opt/chrome/policies/managed/toolbox-pdf.json \
    && cp /etc/opt/chrome/policies/managed/toolbox-pdf.json \
       /etc/chromium/policies/managed/toolbox-pdf.json
