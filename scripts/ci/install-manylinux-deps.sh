#!/usr/bin/env bash
set -euo pipefail

if command -v yum >/dev/null 2>&1; then
  yum_options=(--setopt=timeout=30 --setopt=retries=2)
  if test -f /etc/almalinux-release; then
    # These build dependencies are all in BaseOS/AppStream. Avoid querying EPEL
    # and third-party mirrorlists; keep the distribution's GPG and TLS checks.
    yum_options+=(
      '--disablerepo=*'
      --enablerepo=baseos
      --enablerepo=appstream
      '--setopt=baseos.mirrorlist='
      '--setopt=appstream.mirrorlist='
      '--setopt=baseos.baseurl=https://repo.almalinux.org/almalinux/$releasever/BaseOS/$basearch/os/'
      '--setopt=appstream.baseurl=https://repo.almalinux.org/almalinux/$releasever/AppStream/$basearch/os/'
    )
  fi
  yum "${yum_options[@]}" install -y curl make openssl-devel perl
elif command -v apt-get >/dev/null 2>&1; then
  apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update
  apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 \
    install -y curl libssl-dev ca-certificates make perl
else
  echo 'No supported package manager for manylinux build dependencies' >&2
  exit 1
fi
