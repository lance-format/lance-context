# Storage backends

Point any store at cloud storage with `storage_options`:

```python
from lance_context import Context

ctx = Context.create(
    "s3://my-bucket/memory.lance",
    storage_options={
        "aws_access_key_id": "...",
        "aws_secret_access_key": "...",
        "aws_region": "us-east-1",
    },
)
```

GCS (`gs://`) and Azure (`az://`, `abfss://`) work the same way.

When `storage_options` is omitted, the standard environment variables
(`AWS_ACCESS_KEY_ID`, ...) are used instead.
