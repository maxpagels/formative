import pytest

from formative import DAG


class TestDAG:
    def test_basic_edges(self):
        dag = DAG()
        dag.assume("A").causes("B")
        dag.assume("B").causes("C")
        assert dag.parents("B") == {"A"}
        assert dag.children("B") == {"C"}
        assert dag.ancestors("C") == {"A", "B"}
        assert dag.descendants("A") == {"B", "C"}

    def test_multiple_effects_in_one_call(self):
        dag = DAG()
        dag.assume("A").causes("B", "C")
        assert dag.children("A") == {"B", "C"}

    def test_cycle_detection(self):
        dag = DAG()
        dag.assume("A").causes("B")
        dag.assume("B").causes("C")
        with pytest.raises(Exception, match="cycle"):
            dag.assume("C").causes("A")
