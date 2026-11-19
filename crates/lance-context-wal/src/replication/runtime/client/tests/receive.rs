use super::*;

#[tokio::test]
async fn fragmented_receive_preserves_the_next_frame() -> Result<()> {
    let expected: Vec<u8> = (0..4099).map(|i| (i % 251) as u8).collect();
    let payload = expected.clone();
    let (mut writer, mut reader) = tokio::io::duplex(7);
    let sending = tokio::spawn(async move {
        writer.write_all(&payload).await?;
        writer.write_all(b"next").await?;
        writer.shutdown().await
    });

    assert!(super::super::receive(&mut reader, 0).await?.is_empty());
    assert_eq!(
        super::super::receive(&mut reader, expected.len()).await?,
        expected
    );
    assert_eq!(super::super::receive(&mut reader, 4).await?, &b"next"[..]);
    sending.await??;
    Ok(())
}

#[tokio::test]
async fn truncated_receive_keeps_unexpected_eof() {
    for available in [0, 7] {
        let payload = [42; 8];
        let mut reader = &payload[..available];
        let error = super::super::receive(&mut reader, payload.len())
            .await
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::UnexpectedEof
        );
    }
}
