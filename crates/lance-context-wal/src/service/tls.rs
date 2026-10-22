use super::*;
use rustls::client::danger::ServerCertVerifier;
use tokio_rustls::rustls::{
    self,
    pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer, ServerName, UnixTime},
    RootCertStore,
};

pub(super) struct Identity {
    pub fingerprint: [u8; 32],
    client: rustls::ClientConfig,
    server: rustls::ServerConfig,
}
impl Identity {
    pub fn load(files: &config::TlsFiles, name: &str) -> Result<Self> {
        let certificates =
            CertificateDer::pem_slice_iter(&config::read_bounded(&files.certificate, 1 << 20)?)
                .collect::<Result<Vec<_>, _>>()?;
        ensure!(
            !certificates.is_empty() && certificates.len() <= 16,
            "invalid certificate chain length"
        );
        let key =
            PrivateKeyDer::from_pem_slice(&config::read_bounded(&files.private_key, 64 << 10)?)?;
        let roots = CertificateDer::pem_slice_iter(&config::read_bounded(&files.roots, 1 << 20)?)
            .collect::<Result<Vec<_>, _>>()?;
        ensure!(
            !roots.is_empty() && roots.len() <= 128,
            "invalid root certificate count"
        );
        let mut store = RootCertStore::empty();
        for certificate in roots {
            store.add(certificate)?;
        }
        let store = Arc::new(store);
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let verifier = rustls::client::WebPkiServerVerifier::builder_with_provider(
            store.clone(),
            provider.clone(),
        )
        .build()?;
        verifier.verify_server_cert(
            &certificates[0],
            &certificates[1..],
            &ServerName::try_from(name.to_owned())?,
            &[],
            UnixTime::now(),
        )?;
        let fingerprint = ring::digest::digest(&ring::digest::SHA256, certificates[0].as_ref())
            .as_ref()
            .try_into()?;
        let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
            store.clone(),
            provider.clone(),
        )
        .build()?;
        let server = rustls::ServerConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()?
            .with_client_cert_verifier(verifier)
            .with_single_cert(certificates.clone(), key.clone_key())?;
        let client = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()?
            .with_root_certificates(store)
            .with_client_auth_cert(certificates, key)?;
        Ok(Self {
            fingerprint,
            client,
            server,
        })
    }
    pub fn configs(&self, alpn: &[u8]) -> (Arc<rustls::ClientConfig>, Arc<rustls::ServerConfig>) {
        let mut client = self.client.clone();
        client.alpn_protocols = vec![alpn.to_vec()];
        let mut server = self.server.clone();
        server.alpn_protocols = vec![alpn.to_vec()];
        (Arc::new(client), Arc::new(server))
    }
}
