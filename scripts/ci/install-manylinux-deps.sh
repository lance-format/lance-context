#!/usr/bin/env bash
set -euo pipefail

if command -v yum >/dev/null 2>&1; then
  yum_options=(--setopt=timeout=30 --setopt=retries=2)
  if test -f /etc/almalinux-release; then
    # These build dependencies are all in BaseOS/AppStream. Avoid querying EPEL
    # and third-party mirrorlists; keep the distribution's GPG and TLS checks.
    # yum expands $releasever/$basearch, not the invoking shell.
    # shellcheck disable=SC2016
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
  # The cross-build images also contain unrelated PPAs. These dependencies
  # need only official distro repositories; a PPA outage must not block them.
  # Keep both metadata and packages authenticated by the distro keyring.
  # shellcheck source=/dev/null
  source /etc/os-release
  [[ ${VERSION_CODENAME:-} =~ ^[a-z]+$ ]]
  case "$ID" in
    ubuntu)
      keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg
      case "$(dpkg --print-architecture)" in
        amd64|i386) archive=https://archive.ubuntu.com/ubuntu; security=https://security.ubuntu.com/ubuntu ;;
        *) archive=https://ports.ubuntu.com/ubuntu-ports; security=$archive ;;
      esac
      security_suite=$VERSION_CODENAME-security
      ;;
    debian)
      keyring=/usr/share/keyrings/debian-archive-keyring.gpg
      archive=https://deb.debian.org/debian
      security=https://security.debian.org/debian-security
      security_suite=$VERSION_CODENAME-security
      ;;
    *) echo "Unsupported apt build distribution: $ID" >&2; exit 1 ;;
  esac
  test -r "$keyring"
  apt_workspace=$(mktemp -d)
  trap 'rm -rf -- "$apt_workspace"' EXIT
  chmod 755 "$apt_workspace"
  mkdir "$apt_workspace/lists"
  cat > "$apt_workspace/sources.list" <<EOF
deb [signed-by=$keyring] $archive $VERSION_CODENAME main
deb [signed-by=$keyring] $archive $VERSION_CODENAME-updates main
deb [signed-by=$keyring] $security $security_suite main
EOF
  apt_options=(
    -o Acquire::Retries=2
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
    -o "Dir::Etc::sourcelist=$apt_workspace/sources.list"
    -o Dir::Etc::sourceparts=-
    -o "Dir::State::lists=$apt_workspace/lists"
    -o APT::Update::Error-Mode=any
  )
  apt-get "${apt_options[@]}" update
  apt-get "${apt_options[@]}" install -y curl libssl-dev ca-certificates make perl
else
  echo 'No supported package manager for manylinux build dependencies' >&2
  exit 1
fi
