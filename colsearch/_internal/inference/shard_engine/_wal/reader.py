"""Replay reader for the shard-engine WAL."""
from __future__ import annotations

from .common import *  # noqa: F401,F403
from .common import _REPLAY_CHUNK_SIZE
from .types import WalEntry, WalOp

class WalReader:
    """Reads and replays WAL entries, skipping corrupted ones."""

    def __init__(self, path: Path):
        self._path = Path(path)

    def replay(self) -> List[WalEntry]:
        if not self._path.exists():
            return []

        entries: List[WalEntry] = []
        corrupted = 0
        buf = b""

        with open(self._path, "rb") as fd:
            while True:
                chunk = fd.read(_REPLAY_CHUNK_SIZE)
                if not chunk and not buf:
                    break
                buf += chunk
                buf, new_entries, new_corrupt = self._process_buffer(buf, eof=not chunk)
                entries.extend(new_entries)
                corrupted += new_corrupt
                if not chunk:
                    break

        if corrupted > 0:
            logger.warning("WAL replay: %d corrupted entries skipped", corrupted)

        return entries

    @staticmethod
    def _process_buffer(data: bytes, eof: bool = False):
        """Parse as many complete entries as possible from *data*.

        Returns ``(remaining_bytes, entries, corrupt_count)``.
        """
        entries: List[WalEntry] = []
        corrupted = 0
        pos = 0

        while pos + HEADER_SIZE <= len(data):
            header = data[pos:pos + HEADER_SIZE]
            magic, version, entry_len = struct.unpack(HEADER_FMT, header)

            if magic != WAL_MAGIC or version != WAL_VERSION:
                corrupted += 1
                nxt = data.find(WAL_MAGIC, pos + 1)
                if nxt == -1:
                    if eof:
                        pos = len(data)
                    break
                pos = nxt
                continue

            payload_end = pos + HEADER_SIZE + entry_len
            crc_end = payload_end + CRC_SIZE
            if crc_end > len(data):
                if eof:
                    corrupted += 1
                    pos = len(data)
                break

            payload = data[pos + HEADER_SIZE:payload_end]
            expected_crc = struct.unpack("<I", data[payload_end:crc_end])[0]
            actual_crc = zlib.crc32(header + payload) & 0xFFFFFFFF

            if expected_crc != actual_crc:
                corrupted += 1
                logger.warning("WAL CRC mismatch at offset %d", pos)
                nxt = data.find(WAL_MAGIC, pos + 1)
                if nxt == -1:
                    if eof:
                        pos = len(data)
                    break
                pos = nxt
                continue

            entry = WalReader._parse_entry(payload)
            if entry is not None:
                entries.append(entry)
            pos = crc_end

        return data[pos:], entries, corrupted

    @staticmethod
    def _parse_entry(payload: bytes) -> Optional[WalEntry]:
        if len(payload) < 9:
            return None

        op_byte = payload[0]
        doc_id = struct.unpack_from("<q", payload, 1)[0]
        try:
            op = WalOp(op_byte)
        except ValueError:
            logger.warning("WAL: unknown op byte %d at doc_id %d", op_byte, doc_id)
            return None

        vectors = None
        entry_payload = None
        rest = payload[9:]

        if op == WalOp.UPDATE_PAYLOAD:
            if len(rest) >= 4:
                pld_len = struct.unpack_from("<I", rest, 0)[0]
                if len(rest) >= 4 + pld_len:
                    try:
                        entry_payload = json.loads(
                            rest[4:4 + pld_len].decode("utf-8"),
                        )
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
        elif op != WalOp.DELETE and len(rest) >= 8:
            n_vecs, dim = struct.unpack_from("<II", rest, 0)
            expected = n_vecs * dim * 4
            vec_data = rest[8:]
            if len(vec_data) >= expected:
                vectors = np.frombuffer(
                    vec_data[:expected], dtype=np.float32,
                ).reshape(n_vecs, dim).copy()
                trailing = vec_data[expected:]
                if len(trailing) >= 4:
                    pld_len = struct.unpack_from("<I", trailing, 0)[0]
                    if len(trailing) >= 4 + pld_len:
                        try:
                            entry_payload = json.loads(
                                trailing[4:4 + pld_len].decode("utf-8"),
                            )
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass

        return WalEntry(op=op, doc_id=doc_id, vectors=vectors, payload=entry_payload)

