class SimmsError(Exception):
    """Base for errors simms raises deliberately.

    The CLI prints these as a plain ``Error: <message>`` with no traceback: they describe
    something wrong with the *input*, so the stack that produced them tells the user nothing.
    Anything not descending from this is an unexpected failure and keeps its type in the
    message (and its traceback under ``--log-level DEBUG``).
    """


class InvalidInputError(SimmsError, ValueError):
    """An option value that cannot produce a valid result.

    Raised before any output is written, so a rejected run never leaves a half-built or
    physically meaningless MS behind.
    """


class ASCIISourceError(SimmsError):
    pass


class FITSSkymodelError(SimmsError):
    pass


class ASCIISkymodelError(SimmsError):
    pass


class SkymodelSchemaError(SimmsError, AttributeError):
    pass
