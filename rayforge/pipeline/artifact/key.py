from dataclasses import dataclass
import uuid


@dataclass(frozen=True)
class ArtifactKey:
    id: str
    group: str

    def __post_init__(self):
        if self.group == "workpiece":
            return
        if self.group != "job":
            uuid.UUID(self.id)

    def __hash__(self):
        return hash((self.id, self.group))

    def __eq__(self, other):
        if not isinstance(other, ArtifactKey):
            return False
        return self.id == other.id and self.group == other.group

    @classmethod
    def for_workpiece(cls, workpiece_uid: str, step_uid: str = ""):
        if step_uid:
            return cls(id=f"{workpiece_uid}:{step_uid}", group="workpiece")
        return cls(id=workpiece_uid, group="workpiece")

    @classmethod
    def for_step(cls, step_uid: str):
        return cls(id=step_uid, group="step")

    @classmethod
    def for_job(cls):
        return cls(id=str(uuid.uuid4()), group="job")

    @classmethod
    def for_view(cls, workpiece_uid: str):
        return cls(id=workpiece_uid, group="view")
