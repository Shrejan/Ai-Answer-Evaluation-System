"""
stage3.candidates — Multi-source candidate generation and aggregation.
"""
from typing import Protocol, List
from stage3.schemas import Span


class CandidateSource(Protocol):
    """Protocol for candidate generation sources."""
    name: str

    def propose(self, span: Span) -> List[str]:
        """Propose a list of candidate replacements for the given span."""
        ...
