"""Versioned, persistent FAISS HNSW index over immutable SQLite chunks."""
import hashlib
import json
import os
import uuid
from pathlib import Path

from .errors import RagError


def _libraries():
    try:
        import faiss
        import numpy as np
        return faiss, np
    except ImportError as exc:
        raise RagError("FAISS is unavailable; install the ann extra") from exc


def build_ann(store, index_id, *, m=32, ef_construction=100, batch_size=512):
    faiss, np = _libraries()
    if not 4 <= m <= 128 or not 8 <= ef_construction <= 2000 or batch_size < 1:
        raise ValueError("Invalid HNSW build parameters")
    with store.connect() as db:
        row = db.execute("SELECT dimension FROM indexes WHERE id=?", (index_id,)).fetchone()
        if not row:
            raise RagError("Index version does not exist")
        dim = row[0]
        count = db.execute("SELECT COUNT(*) FROM chunks WHERE index_id=?", (index_id,)).fetchone()[0]
        if not count:
            raise RagError("Index has no chunks")
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        mapping, batch = [], []
        for chunk_id, vector in db.execute("SELECT chunk_id, vector FROM chunks WHERE index_id=? ORDER BY chunk_id", (index_id,)):
            mapping.append(chunk_id)
            batch.append(json.loads(vector))
            if len(batch) >= batch_size:
                index.add(np.asarray(batch, dtype=np.float32))
                batch.clear()
        if batch:
            index.add(np.asarray(batch, dtype=np.float32))
    # Use Python file I/O: FAISS's native Windows filename handling can fail on Unicode paths.
    payload = faiss.serialize_index(index).tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    revision = uuid.uuid4().hex
    directory = Path(str(store.path) + ".ann")
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{index_id}-{revision}.faiss"
    target = directory / filename
    with target.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    # Publish metadata only after the complete file is durable; a crash can leave an orphan file.
    with store.connect() as db:
        db.execute("CREATE TABLE IF NOT EXISTS ann_revisions (index_id TEXT PRIMARY KEY, revision TEXT NOT NULL, filename TEXT NOT NULL, sha256 TEXT NOT NULL, dimension INTEGER NOT NULL, count INTEGER NOT NULL, parameters TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS ann_items (revision TEXT NOT NULL, ordinal INTEGER NOT NULL, chunk_id TEXT NOT NULL, PRIMARY KEY(revision, ordinal))")
        db.executemany("INSERT INTO ann_items VALUES (?, ?, ?)", [(revision, i, chunk) for i, chunk in enumerate(mapping)])
        params = {"type": "faiss_hnsw_flat", "metric": "inner_product_normalized", "m": m, "ef_construction": ef_construction}
        db.execute("INSERT OR REPLACE INTO ann_revisions VALUES (?, ?, ?, ?, ?, ?, ?)", (index_id, revision, filename, digest, dim, count, json.dumps(params)))
    store._ann_cache.clear()
    return {"index_id": index_id, "revision": revision, "count": count, "dimension": dim,
            "sha256": digest, "filename": filename, "parameters": params}


def search_ann(store, db, index_id, vector, *, top_k, min_score, ef_search):
    from .pipeline import Hit
    faiss, np = _libraries()
    if not 1 <= ef_search <= 10000:
        raise ValueError("ef_search must be between 1 and 10000")
    try:
        row = db.execute("SELECT revision, filename, sha256, dimension, count FROM ann_revisions WHERE index_id=?", (index_id,)).fetchone()
    except Exception as exc:
        raise RagError("ANN index not built for this database") from exc
    if not row:
        raise RagError("ANN index not built for this index version")
    revision, filename, digest, dim, count = row
    if Path(filename).name != filename or dim != len(vector):
        raise RagError("Invalid ANN index metadata")
    index = store._ann_cache.get(revision)
    if index is None:
        path = Path(str(store.path) + ".ann") / filename
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise RagError("Cannot read ANN index file") from exc
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RagError("ANN index checksum mismatch")
        try:
            index = faiss.deserialize_index(np.frombuffer(payload, dtype=np.uint8).copy())
        except Exception as exc:
            raise RagError("Cannot decode ANN index") from exc
        if index.d != dim or index.ntotal != count:
            raise RagError("ANN index dimension/count mismatch")
        store._ann_cache = {revision: index}
    index.hnsw.efSearch = max(ef_search, top_k)
    scores, labels = index.search(np.asarray([vector], dtype=np.float32), min(top_k, count))
    positions = [int(pos) for pos in labels[0] if pos >= 0]
    if not positions:
        return []
    placeholders = ",".join("?" for _ in positions)
    rows = db.execute(f"SELECT a.ordinal, c.chunk_id, c.source, c.start, c.end, c.text FROM ann_items a JOIN chunks c ON c.index_id=? AND c.chunk_id=a.chunk_id WHERE a.revision=? AND a.ordinal IN ({placeholders})", (index_id, revision, *positions)).fetchall()
    by_pos = {r[0]: r[1:] for r in rows}
    if len(by_pos) != len(positions):
        raise RagError("ANN label mapping is incomplete")
    result = []
    for pos, score in zip(labels[0], scores[0]):
        if pos >= 0 and float(score) >= min_score:
            result.append(Hit(*by_pos[int(pos)], float(score)))
    return result
