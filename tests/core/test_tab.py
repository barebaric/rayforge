import pytest
import uuid
from rayforge.core.tab import Tab


class TestTab:
    def test_initialization(self):
        """Tests basic initialization of the Tab dataclass."""
        tab = Tab(width=3.0, segment_index=5, t=0.5)
        assert tab.width == 3.0
        assert tab.segment_index == 5
        assert tab.t == 0.5
        assert isinstance(tab.uid, str)
        assert len(tab.uid) > 0

    def test_uid_generation(self):
        """Tests that UIDs are unique and valid."""
        tab1 = Tab(width=1, segment_index=1, t=0.1)
        tab2 = Tab(width=2, segment_index=2, t=0.2)
        assert tab1.uid != tab2.uid
        # Check if the UID is a valid UUID string
        try:
            uuid.UUID(tab1.uid)
            uuid.UUID(tab2.uid)
        except ValueError:
            pytest.fail("Generated UID is not a valid UUID string")

    def test_custom_uid(self):
        """Tests initialization with a custom UID."""
        custom_id = "my-custom-tab-id"
        tab = Tab(width=5.0, segment_index=0, t=0.0, uid=custom_id)
        assert tab.uid == custom_id
