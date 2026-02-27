class IdentificationError(Exception):
    """
    Raised when a confounder declared in the DAG is absent from the dataframe.

    Note: this only checks confounders the user explicitly modelled. There may
    be additional unobserved confounders not represented in the DAG at all —
    formative has no way to detect those.
    """
    pass


class GraphError(Exception):
    """Raised when the DAG is structurally invalid."""
    pass
