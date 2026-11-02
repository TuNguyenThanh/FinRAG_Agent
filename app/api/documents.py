from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.rag.ingest import SUPPORTED_SUFFIXES, UnsupportedFileError, chunk_document
from app.security.auth import require_user
from app.services import Services, get_services

router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.post("", status_code=status.HTTP_201_CREATED, summary="Upload and index a document")
async def upload_document(
    file: UploadFile = File(...),
    user: str = Depends(require_user),
    services: Services = Depends(get_services),
):
    settings = services.settings
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(415, f"supported types: {sorted(SUPPORTED_SUFFIXES)}")
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"file larger than {settings.max_upload_mb} MB")
    if services.store is None:
        raise HTTPException(503, "document store unavailable")

    try:
        chunks = chunk_document(name, data, settings.chunk_size, settings.chunk_overlap)
    except UnsupportedFileError as exc:
        raise HTTPException(415, str(exc)) from exc
    if not chunks:
        raise HTTPException(422, "no extractable text (scanned PDF?)")

    embeddings = await services.embedder.embed_documents([c.content for c in chunks])
    doc_id, changed = await services.store.upsert_document(name, data, chunks, embeddings)
    if changed and services.cache:
        await services.cache.clear()  # cached answers may now be stale
    await services.audit.write(
        user, "document.upsert", None, name=name, chunks=len(chunks), changed=changed
    )
    return {"id": doc_id, "name": name, "chunks": len(chunks), "changed": changed}


@router.get("", summary="List indexed documents")
async def list_documents(
    _: str = Depends(require_user), services: Services = Depends(get_services)
):
    if services.store is None:
        raise HTTPException(503, "document store unavailable")
    return {"documents": await services.store.list_documents()}


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    name: str, user: str = Depends(require_user), services: Services = Depends(get_services)
):
    if services.store is None or not await services.store.delete_document(name):
        raise HTTPException(404, "document not found")
    if services.cache:
        await services.cache.clear()
    await services.audit.write(user, "document.delete", None, name=name)
