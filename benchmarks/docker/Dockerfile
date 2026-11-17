# Image for benchmarks/compare.py on one Linux distribution, see run.py. The base image is
# swapped through BASE; Rust, uv and Python are installed the same way on all of them.
ARG BASE=debian:13
FROM ${BASE}
COPY install.sh /tmp/install.sh
RUN sh /tmp/install.sh
ENV PATH=/root/.cargo/bin:/root/.local/bin:$PATH \
    UV_PYTHON=3.14 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/cache/uv
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv python install 3.14
COPY bench.sh /usr/local/bin/bench
ENTRYPOINT ["sh", "/usr/local/bin/bench"]
