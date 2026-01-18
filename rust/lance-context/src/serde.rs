use arrow_array::RecordBatch;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::ArrowError;
use serde::{Deserialize, Serialize};

pub const CONTENT_TYPE_TEXT: &str = "text/plain";
pub const CONTENT_TYPE_ARROW_STREAM: &str = "application/vnd.apache.arrow.stream";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializedContent {
    pub content_type: String,
    pub text_payload: Option<String>,
    pub binary_payload: Option<Vec<u8>>,
}

impl SerializedContent {
    pub fn text(value: impl Into<String>, content_type: Option<&str>) -> Self {
        Self {
            content_type: content_type.unwrap_or(CONTENT_TYPE_TEXT).to_string(),
            text_payload: Some(value.into()),
            binary_payload: None,
        }
    }

    pub fn image(bytes: impl Into<Vec<u8>>, mime: impl Into<String>) -> Self {
        Self {
            content_type: mime.into(),
            text_payload: None,
            binary_payload: Some(bytes.into()),
        }
    }

    pub fn dataframe_batches(batches: &[RecordBatch]) -> Result<Self, ArrowError> {
        let ipc_bytes = record_batches_to_ipc(batches)?;
        Ok(Self::dataframe_ipc_bytes(ipc_bytes))
    }

    pub fn dataframe_ipc_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            content_type: CONTENT_TYPE_ARROW_STREAM.to_string(),
            text_payload: None,
            binary_payload: Some(bytes.into()),
        }
    }
}

pub fn serialize_image(bytes: impl Into<Vec<u8>>, mime: impl Into<String>) -> SerializedContent {
    SerializedContent::image(bytes, mime)
}

pub fn serialize_dataframe(batches: &[RecordBatch]) -> Result<SerializedContent, ArrowError> {
    SerializedContent::dataframe_batches(batches)
}

pub fn serialize_dataframe_ipc(bytes: impl Into<Vec<u8>>) -> SerializedContent {
    SerializedContent::dataframe_ipc_bytes(bytes)
}

fn record_batches_to_ipc(batches: &[RecordBatch]) -> Result<Vec<u8>, ArrowError> {
    if batches.is_empty() {
        return Err(ArrowError::InvalidArgumentError(
            "no record batches provided".to_string(),
        ));
    }

    let schema = batches[0].schema();
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema)?;
        for batch in batches {
            if batch.schema() != schema {
                return Err(ArrowError::SchemaError(
                    "record batch schema mismatch".to_string(),
                ));
            }
            writer.write(batch)?;
        }
        writer.finish()?;
    }
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_ipc::reader::StreamReader;
    use arrow_schema::{DataType, Field, Schema};
    use std::io::Cursor;
    use std::sync::Arc;

    fn make_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let id_array = Arc::new(Int32Array::from(vec![1, 2]));
        let name_array = Arc::new(StringArray::from(vec!["alpha", "beta"]));
        RecordBatch::try_new(schema, vec![id_array, name_array]).unwrap()
    }

    #[test]
    fn image_serialization_sets_payloads() {
        let content = serialize_image(vec![1, 2, 3], "image/png");
        assert_eq!(content.content_type, "image/png");
        assert_eq!(content.text_payload, None);
        assert_eq!(content.binary_payload, Some(vec![1, 2, 3]));
    }

    #[test]
    fn dataframe_serialization_writes_ipc_stream() {
        let batch = make_batch();
        let content = serialize_dataframe(std::slice::from_ref(&batch)).unwrap();
        assert_eq!(content.content_type, CONTENT_TYPE_ARROW_STREAM);
        let bytes = content.binary_payload.expect("expected IPC payload");

        let reader = StreamReader::try_new(Cursor::new(bytes), None).unwrap();
        let batches: Vec<RecordBatch> = reader.map(|item| item.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].schema(), batch.schema());
        assert_eq!(batches[0].num_rows(), batch.num_rows());
    }

    #[test]
    fn dataframe_serialization_rejects_empty_batches() {
        let err = serialize_dataframe(&[]).unwrap_err();
        assert!(matches!(err, ArrowError::InvalidArgumentError(_)));
    }

    #[test]
    fn dataframe_serialization_rejects_mismatched_schema() {
        let batch = make_batch();
        let other_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let other_batch =
            RecordBatch::try_new(other_schema, vec![Arc::new(Int32Array::from(vec![1, 2]))])
                .unwrap();

        let err = serialize_dataframe(&[batch, other_batch]).unwrap_err();
        assert!(matches!(err, ArrowError::SchemaError(_)));
    }
}
