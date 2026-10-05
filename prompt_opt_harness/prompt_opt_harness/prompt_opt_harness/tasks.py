"""Immutable output schemas, independent of optimized instruction wording."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    labels: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, 'labels', tuple(self.labels))
        if not self.name or len(self.labels) < 2:
            raise ValueError('A task needs a name and at least two labels')
        if any(not label.isalpha() for label in self.labels):
            raise ValueError('Classification labels must be nonempty single words')
        if len({label.casefold() for label in self.labels}) != len(self.labels):
            raise ValueError('Task labels must be distinct ignoring case')

    @property
    def answer_format(self):
        return 'Answer with exactly one word: ' + ' or '.join(self.labels) + '.'


SENTIMENT = TaskSpec('sentiment', ('Positive', 'Negative'))
BINARY_NLI = TaskSpec('binary_nli', ('Entailment', 'Contradiction'))
