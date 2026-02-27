class IdentificationError(Exception):
    """Raised when causal identification fails given the provided DAG."""
    pass


class GraphError(Exception):
    """Raised when the DAG is structurally invalid."""
    pass
