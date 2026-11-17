#!/bin/sh
# System packages of the benchmark image, for Debian/Ubuntu and Fedora/RHEL style bases: build
# tools for the extension, perf, strace and the allocators that bench.sh can preload.
set -eux
if command -v apt-get >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends build-essential ca-certificates curl git strace
  # Package names differ between releases: t64 suffixes since Debian 13 and Ubuntu 24.04.
  for package in libjemalloc2 libmimalloc3 libmimalloc2.0 libtcmalloc-minimal4t64 libtcmalloc-minimal4; do
    apt-get install -y --no-install-recommends "$package" || true
  done
  # perf: Debian ships it as linux-perf; Ubuntu per kernel, behind a wrapper that refuses to run
  # when the container's kernel tools do not match the host's kernel, so link the binary itself.
  if ! apt-get install -y --no-install-recommends linux-perf; then
    apt-get install -y --no-install-recommends linux-tools-generic
    ln -sf "$(ls /usr/lib/linux-tools/*/perf | head -n 1)" /usr/local/bin/perf
  fi
  rm -rf /var/lib/apt/lists/*
else
  dnf install -y gcc gcc-c++ make ca-certificates curl git strace perf findutils
  for package in jemalloc mimalloc gperftools-libs; do
    dnf install -y "$package" || true
  done
  dnf clean all
fi
