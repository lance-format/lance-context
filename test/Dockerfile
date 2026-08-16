# syntax=docker/dockerfile:1

# Single multi-stage build for the lance-context test environment.
#
# Stages:
#   ui        -> builds the master's React/Vite admin UI into static assets
#   builder   -> compiles the Rust workspace binaries (master + server)
#   master    -> runtime image for lance-context-master (control-plane), bundles UI
#   worker    -> runtime image for lance-context-server (data-plane)
#
# Select a target with `--target master` / `--target worker`, which is what the
# accompanying docker-compose.yml does. Built once, the shared `builder` stage
# is cached across both runtime images.

# ---------------------------------------------------------------------------
# UI assets (only needed by the master image)
# ---------------------------------------------------------------------------
FROM node:20-bookworm-slim AS ui
WORKDIR /ui
COPY crates/lance-context-master/ui/package.json crates/lance-context-master/ui/package-lock.json ./
RUN npm ci
COPY crates/lance-context-master/ui/ ./
RUN npm run build
# Vite emits to ./dist

# ---------------------------------------------------------------------------
# Rust workspace build
# ---------------------------------------------------------------------------
FROM rust:1-bookworm AS builder
# Native deps mirror what CI installs: protoc for the lance protobufs, openssl
# for etcd TLS + object-store, make/perl for the openssl-sys vendored build.
# libprotobuf-dev is not optional here: Debian's protobuf-compiler ships only
# the protoc binary, and the well-known types (google/protobuf/empty.proto and
# friends, which the lance-* protos import) live in libprotobuf-dev. It is a
# Recommends, so --no-install-recommends drops it and lance-encoding's build
# script fails with "google/protobuf/empty.proto: File not found".
RUN apt-get update && apt-get install -y --no-install-recommends \
        protobuf-compiler libprotobuf-dev libssl-dev pkg-config make perl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY . .
# Build only the two service binaries we ship (skip python/examples).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release \
        -p lance-context-master \
        -p lance-context-server \
    && mkdir -p /out \
    && cp target/release/lance-context-master /out/ \
    && cp target/release/lance-context-server /out/

# ---------------------------------------------------------------------------
# Master runtime (control-plane + UI)
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS master
RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl3 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /out/lance-context-master /usr/local/bin/lance-context-master
COPY --from=ui /ui/dist /app/ui
ENV UI_DIR=/app/ui
ENV MASTER_PORT=8090
EXPOSE 8090
ENTRYPOINT ["lance-context-master"]

# ---------------------------------------------------------------------------
# Worker runtime (data-plane)
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS worker
RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl3 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /out/lance-context-server /usr/local/bin/lance-context-server
EXPOSE 3000
ENTRYPOINT ["lance-context-server"]
