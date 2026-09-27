"""Domain errors. Entry points (API, CLI) decide how to present them; the core knows nothing about HTTP."""


class HarnessError(Exception):
    pass


class InvalidInputError(HarnessError):
    pass


class NotFoundError(HarnessError):
    pass


class ConflictError(HarnessError):
    """The request is valid but does not fit the run's current state (e.g. deciding an approval twice)."""
