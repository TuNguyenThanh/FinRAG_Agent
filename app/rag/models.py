from dataclasses import asdict, dataclass


@dataclass
class Chunk:
    doc_name: str
    content: str
    page: int | None = None
    section: str | None = None


@dataclass
class RetrievedChunk(Chunk):
    id: int = 0
    score: float = 0.0

    def citation(self) -> str:
        parts = [self.doc_name]
        if self.page:
            parts.append(f"tr.{self.page}")
        if self.section:
            parts.append(self.section)
        return ", ".join(parts)

    def to_source(self) -> dict:
        d = asdict(self)
        d["content"] = self.content[:300]
        d["citation"] = self.citation()
        return d
