//! Hold the test's peer endpoints across child SIGKILL/restart. The parent
//! never accepts connections or relays TLS traffic on these sockets.
use super::*;
use std::{
    net::TcpListener as Listener,
    os::{fd::AsRawFd, fd::FromRawFd, unix::process::CommandExt},
};

pub(super) struct Listeners {
    peer: Listener,
    image: Listener,
}

impl Listeners {
    pub(super) fn bind() -> Result<Self> {
        Ok(Self {
            peer: Listener::bind("127.0.0.1:0")?,
            image: Listener::bind("127.0.0.1:0")?,
        })
    }

    pub(super) fn addresses(&self) -> Result<(SocketAddr, SocketAddr)> {
        Ok((self.peer.local_addr()?, self.image.local_addr()?))
    }

    pub(super) fn inherit(&self, command: &mut ProcessCommand) {
        let fds = [self.peer.as_raw_fd(), self.image.as_raw_fd()];
        command
            .env("CONTEXT_TLS_PEER_FD", fds[0].to_string())
            .env("CONTEXT_TLS_IMAGE_FD", fds[1].to_string());
        // SAFETY: only async-signal-safe fcntl calls and allocation-free error
        // handling run after fork. self remains alive through Command::spawn.
        // Change CLOEXEC in the child only so concurrent test children cannot
        // inherit unrelated listeners and extend their lifetime.
        unsafe {
            command.pre_exec(move || {
                for fd in fds {
                    if libc::fcntl(fd, libc::F_SETFD, 0) == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
    }
}

pub(super) fn receive() -> Result<(TcpListener, TcpListener)> {
    fn listener(name: &str) -> Result<TcpListener> {
        let fd: i32 = std::env::var(name)?.parse()?;
        ensure!(fd >= 3, "test listener must not use a standard stream");
        // SAFETY: the parent passed this open listener to this exact fixture
        // through exec, and this helper takes ownership once per descriptor.
        let listener = unsafe { Listener::from_raw_fd(fd) };
        // SAFETY: fd is owned by listener. Restore close-on-exec before the
        // fixture starts its services; parent descriptors were never changed.
        ensure!(
            unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } != -1,
            "restore test listener close-on-exec: {}",
            std::io::Error::last_os_error()
        );
        listener.set_nonblocking(true)?;
        Ok(TcpListener::from_std(listener)?)
    }
    Ok((
        listener("CONTEXT_TLS_PEER_FD")?,
        listener("CONTEXT_TLS_IMAGE_FD")?,
    ))
}
