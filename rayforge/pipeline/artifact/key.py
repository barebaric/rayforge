from dataclasses import dataclass
import uuid


@dataclass(frozen=True)
class ArtifactKey:
    id: str
    group: str

    def __post_init__(self):
        # This is not a perfect check for UUID but ensures it's a string
        # that can be instantiated as a UUID if needed, which is good enough
        # for serialization boundaries.
        if self.group != "job":
            uuid.UUID(self.id)

    def __hash__(self):
        return hash((self.id, self.group))

    def __eq__(self, other):
        if not isinstance(other, ArtifactKey):
            return False
        return self.id == other.id and self.group == other.group

    @classmethod
    def for_workpiece(cls, workpiece_uid: str):
        return cls(id=workpiece_uid, group="workpiece")

    @classmethod
    def for_step(cls, step_uid: str):
        return cls(id=step_uid, group="step")

    @classmethod
    def for_job(cls):
        # The job key is a singleton logical concept, but we give it a UUID
        # to fit the generic key model.
        return cls(id="00000000-0000-0000-0000-000000000000", group="job")

    @classmethod
    def for_view(cls, workpiece_uid: str):
        return cls(id=workpiece_uid, group="view")
