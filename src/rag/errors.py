"""Errors safe to surface without printing provider responses or credentials."""


class RagError(Exception):
    pass


class ConfigurationError(RagError):
    pass


class ProviderError(RagError):
    pass


class ProviderResponseError(ProviderError):
    pass
