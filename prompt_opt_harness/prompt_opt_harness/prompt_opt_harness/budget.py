"""Search-scoped request limits; final evaluation uses the original backend."""
from contextlib import nullcontext


class CallBudgetExceeded(RuntimeError):
    pass


class BudgetedBackend:
    """Cap attempts relative to the start of a search, including native retries.

    Custom backends must increment stats.calls for every attempted request.
    Native LLMBackend also checks the limit inside its retry loop.
    """
    def __init__(self, backend, budget):
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
            raise ValueError('Search budget must be a nonnegative integer')
        self._backend = backend
        self.limit = backend.stats.calls + budget

    def __getattr__(self, name):
        return getattr(self._backend, name)

    @property
    def remaining(self):
        return max(0, self.limit - self.stats.calls)

    def require_calls(self, count):
        if count > self.remaining:
            raise CallBudgetExceeded(
                f'Complete operation needs {count} requests; {self.remaining} remain')

    def generate(self, prompt):
        self.require_calls(1)
        scope = (self._backend.limit_calls(self.limit)
                 if hasattr(self._backend, 'limit_calls') else nullcontext())
        with scope:
            return self._backend.generate(prompt)
