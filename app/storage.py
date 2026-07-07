from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import boto3

from app.config import Settings


def upload_image(settings: Settings, image_bytes: bytes) -> tuple[str, str]:
    date_prefix = datetime.now(timezone.utc).strftime("%Y%m%d")
    object_key = f"modal_pony_infs/{date_prefix}/{uuid4()}.png"

    client = boto3.client(
        service_name="s3",
        endpoint_url=settings.aws_endpoint,
        region_name=settings.aws_default_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
    client.put_object(
        Bucket=settings.aws_bucket,
        Key=object_key,
        Body=image_bytes,
        ContentType="image/png",
    )

    base_url = settings.aws_url.rstrip("/")
    public_url = f"{base_url}/{object_key}"
    return public_url, object_key
