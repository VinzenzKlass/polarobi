#!/bin/sh
# Entry point of the benchmark image, started by run.py:
#
#     bench <base ref or -> <base allocator> <head allocator> <label>
#
# with further compare.py options in COMPARE_ARGS.
# The repository is mounted read-only at /repo and results go to /results. The head is the
# working tree as it is, uncommitted changes included; the base is a clone at <base ref>, or the
# head itself with -, to compare allocators alone. Allocators: system, jemalloc, mimalloc,
# tcmalloc, or a path to a shared library.
set -eu
base_ref=$1 base_allocator=$2 head_allocator=$3 label=$4

# Builds and downloads are kept in the /cache volume between runs.
mkdir -p /cache/registry /cache/target
ln -sfn /cache/registry /root/.cargo/registry
git config --global --add safe.directory '*'

rm -rf /work
mkdir -p /work/head
tar -C /repo --exclude=./.git --exclude=./.venv --exclude=./target -cf - . | tar -C /work/head -xf -
if [ "$base_ref" = "-" ]; then
  base=/work/head
else
  base=/work/base
  git clone -q /repo "$base"
  git -C "$base" checkout -q "$base_ref"
fi

sync() {
  echo "Building $1 in $2"
  (cd "$2" && UV_PROJECT_ENVIRONMENT=/venvs/$1 CARGO_TARGET_DIR=/cache/target/$1 \
    uv sync --group bench --locked --quiet)
}
sync head /work/head
if [ "$base" = /work/head ]; then
  ln -sfn /venvs/head /venvs/base
else
  sync base "$base"
fi

library() {
  case $1 in
    system) ;;
    jemalloc) ldconfig -p | awk '/libjemalloc\.so/ { print $NF; exit }' ;;
    mimalloc) ldconfig -p | awk '/libmimalloc\.so/ { print $NF; exit }' ;;
    tcmalloc) ldconfig -p | awk '/libtcmalloc_minimal\.so/ { print $NF; exit }' ;;
    *) echo "$1" ;;
  esac
}
set --
for side in base head; do
  eval "allocator=\$${side}_allocator"
  path=$(library "$allocator")
  if [ "$allocator" != system ] && [ -z "$path" ]; then
    echo "$allocator is not installed in this image" >&2
    exit 1
  fi
  [ -n "$path" ] && set -- "$@" "--$side-preload" "$path"
done

exec /venvs/head/bin/python /work/head/benchmarks/compare.py \
  --base "$base" --head /work/head \
  --base-python /venvs/base/bin/python --head-python /venvs/head/bin/python \
  --label "$label" --out /results/compare.json --summary /results/summary.md \
  "$@" ${COMPARE_ARGS:-}
