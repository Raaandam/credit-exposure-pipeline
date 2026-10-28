"""Raw/curated storage locations for local disk, Azurite and Azure.

Three backends, selected by ``STORAGE_BACKEND``:

* ``local``   - plain paths under DATA_DIR (laptop / CI, no Azure dependency).
* ``azurite`` - Azure Blob emulator in docker-compose. Spark uses the WASB driver
  (``wasb://<container>@devstoreaccount1.blob.core.windows.net/...``). The compose file gives
  the Azurite container that DNS alias and Azurite parses the account from the host name
  ("production-style URL"), so the *same* URL shape works against real Azure Blob.
* ``azure``   - real Azure Storage. ``AZURE_STORAGE_SCHEME=abfss`` (ADLS Gen2, preferred:
  hierarchical namespace, atomic renames for Spark commits) or ``wasbs`` (classic Blob).
  Auth is an account key or a service principal (OAuth client credentials).

Spark itself only needs ``spark_hadoop_conf``; the azure-storage-blob SDK is used by the
upload helper and the Airflow sensor (imported lazily so tests don't require it).
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from exposure_pipeline import reference as ref
from exposure_pipeline.config import Settings, get_settings

log = logging.getLogger(__name__)


def _host(settings: Settings, container_kind: str = "blob") -> str:
    return f"{settings.azure_account}.{container_kind}.core.windows.net"


def container_uri(settings: Settings, container: str) -> str:
    if settings.storage_backend == "local":
        root = settings.raw_dir if container == settings.raw_container else settings.curated_dir
        return root.resolve().as_uri()
    if settings.storage_backend == "azurite":
        # Azurite speaks plain HTTP, hence wasb:// (not wasbs://)
        return f"wasb://{container}@{_host(settings)}"
    scheme = settings.azure_scheme
    kind = "dfs" if scheme.startswith("abfs") else "blob"
    return f"{scheme}://{container}@{_host(settings, kind)}"


def raw_uri(settings: Settings, business_date: date | str, filename: str) -> str:
    bd = business_date if isinstance(business_date, str) else business_date.isoformat()
    return f"{container_uri(settings, settings.raw_container)}/business_date={bd}/{filename}"


def curated_uri(settings: Settings, table: str) -> str:
    return f"{container_uri(settings, settings.curated_container)}/{table}"


def spark_hadoop_conf(settings: Settings) -> dict[str, str]:
    """Hadoop filesystem options, applied as ``spark.hadoop.*`` on the SparkSession."""
    if settings.storage_backend == "local":
        return {}
    if settings.storage_backend == "azurite":
        return {f"fs.azure.account.key.{_host(settings)}": settings.azure_key}
    if settings.azure_scheme.startswith("abfs"):
        host = _host(settings, "dfs")
        if settings.azure_client_id:
            return {
                f"fs.azure.account.auth.type.{host}": "OAuth",
                f"fs.azure.account.oauth.provider.type.{host}": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
                f"fs.azure.account.oauth2.client.id.{host}": settings.azure_client_id,
                f"fs.azure.account.oauth2.client.secret.{host}": settings.azure_client_secret,
                f"fs.azure.account.oauth2.client.endpoint.{host}": f"https://login.microsoftonline.com/{settings.azure_tenant_id}/oauth2/token",
            }
        return {f"fs.azure.account.key.{host}": settings.azure_key}
    return {f"fs.azure.account.key.{_host(settings)}": settings.azure_key}


# ---------------------------------------------------------------- SDK helpers (host side)
def _service_client(settings: Settings):
    from azure.storage.blob import BlobServiceClient  # lazy: optional dependency

    if settings.azure_connection_string:
        return BlobServiceClient.from_connection_string(settings.azure_connection_string)
    return BlobServiceClient(account_url=f"https://{_host(settings)}", credential=settings.azure_key)


def manifest_exists(business_date: date | str, settings: Settings | None = None) -> bool:
    """Airflow sensor callable: the source drops manifest.json LAST, so its presence means
    the partition is complete (classic "trigger/control file" pattern)."""
    settings = settings or get_settings()
    bd = business_date if isinstance(business_date, str) else business_date.isoformat()
    if settings.storage_backend == "local":
        return (ref.partition_dir(settings.raw_dir, bd) / ref.MANIFEST_FILE).exists()
    blob = _service_client(settings).get_blob_client(settings.raw_container, f"business_date={bd}/{ref.MANIFEST_FILE}")
    return blob.exists()


def upload_partition(business_date: str, settings: Settings | None = None) -> int:
    """Upload one local generator partition to the raw container (simulates the source drop)."""
    settings = settings or get_settings()
    local = ref.partition_dir(settings.raw_dir, business_date)
    svc = _service_client(settings)
    container = svc.get_container_client(settings.raw_container)
    if not container.exists():
        container.create_container()
    files = sorted(p for p in local.iterdir() if p.name != ref.MANIFEST_FILE) + [local / ref.MANIFEST_FILE]
    for path in files:  # manifest last
        with path.open("rb") as fh:
            container.upload_blob(f"business_date={business_date}/{path.name}", fh, overwrite=True)
    log.info("uploaded %d files for %s", len(files), business_date)
    return len(files)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Upload generated raw partitions to Azure Blob / Azurite")
    ap.add_argument("--business-date", help="YYYY-MM-DD; default = all local partitions")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    dates = [args.business_date] if args.business_date else sorted(
        p.name.split("=", 1)[1] for p in Path(settings.raw_dir).glob("business_date=*"))
    for bd in dates:
        upload_partition(bd, settings)


if __name__ == "__main__":
    main()
