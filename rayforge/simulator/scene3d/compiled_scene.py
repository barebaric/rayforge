from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type

import numpy as np

from ...pipeline.artifact.base import BaseArtifact
from ...pipeline.artifact.handle import BaseArtifactHandle


@dataclass
class VertexLayer:
    powered_verts: np.ndarray
    power_values: np.ndarray
    laser_indices: np.ndarray
    travel_verts: np.ndarray
    zero_power_verts: np.ndarray
    powered_cmd_offsets: list = field(default_factory=list)
    travel_cmd_offsets: list = field(default_factory=list)
    is_rotary: bool = False


@dataclass
class TextureLayer:
    power_texture: np.ndarray
    width_px: int
    height_px: int
    model_matrix: np.ndarray
    cylinder_vertices: Optional[np.ndarray] = None
    rotary_diameter: float = 0.0
    rotary_enabled: bool = False
    activation_cmd_idx: int = -1
    laser_uid: str = ""


@dataclass
class ScanlineOverlayLayer:
    positions: np.ndarray
    power_values: np.ndarray
    laser_indices: np.ndarray
    cmd_offsets: list
    is_rotary: bool = False


class CompiledSceneArtifactHandle(BaseArtifactHandle):
    def __init__(
        self,
        shm_name: str,
        handle_class_name: str,
        artifact_type_name: str,
        generation_id: int,
        array_metadata: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ):
        super().__init__(
            shm_name=shm_name,
            handle_class_name=handle_class_name,
            artifact_type_name=artifact_type_name,
            generation_id=generation_id,
            array_metadata=array_metadata,
        )


class CompiledSceneArtifact(BaseArtifact):
    def __init__(
        self,
        generation_id: int,
        vertex_layers: List[VertexLayer],
        texture_layers: List[TextureLayer],
        overlay_layers: List[ScanlineOverlayLayer],
        laser_uid_order: Optional[List[str]] = None,
    ):
        self.generation_id = generation_id
        self.vertex_layers = vertex_layers
        self.texture_layers = texture_layers
        self.overlay_layers = overlay_layers
        self.laser_uid_order = laser_uid_order or []

    def create_handle(self, shm_name, array_metadata):
        return CompiledSceneArtifactHandle(
            shm_name=shm_name,
            handle_class_name=CompiledSceneArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            generation_id=self.generation_id,
            array_metadata=array_metadata,
        )

    def get_arrays_for_storage(self) -> Dict[str, np.ndarray]:
        arrays: Dict[str, np.ndarray] = {}

        arrays["_meta"] = np.array(
            [
                len(self.vertex_layers),
                len(self.texture_layers),
                len(self.overlay_layers),
            ],
            dtype=np.int32,
        )

        if self.laser_uid_order:
            encoded = "\0".join(self.laser_uid_order).encode("utf-8")
            arrays["_luo"] = np.frombuffer(encoded, dtype=np.uint8).copy()

        for i, vl in enumerate(self.vertex_layers):
            arrays[f"vl{i}_pv"] = vl.powered_verts
            arrays[f"vl{i}_pvv"] = vl.power_values
            arrays[f"vl{i}_lid"] = vl.laser_indices
            arrays[f"vl{i}_tv"] = vl.travel_verts
            arrays[f"vl{i}_zpv"] = vl.zero_power_verts
            if vl.powered_cmd_offsets:
                arrays[f"vl{i}_pco"] = np.array(
                    vl.powered_cmd_offsets, dtype=np.int32
                )
            if vl.travel_cmd_offsets:
                arrays[f"vl{i}_tco"] = np.array(
                    vl.travel_cmd_offsets, dtype=np.int32
                )
            if vl.is_rotary:
                arrays[f"vl{i}_ir"] = np.array([1], dtype=np.int32)

        for i, tl in enumerate(self.texture_layers):
            arrays[f"tl{i}_tex"] = tl.power_texture
            arrays[f"tl{i}_mm"] = tl.model_matrix
            arrays[f"tl{i}_meta"] = np.array(
                [tl.width_px, tl.height_px], dtype=np.int32
            )
            if tl.rotary_diameter > 0:
                arrays[f"tl{i}_rd"] = np.array(
                    [tl.rotary_diameter], dtype=np.float32
                )
            if tl.rotary_enabled:
                arrays[f"tl{i}_re"] = np.array([1], dtype=np.int32)
            if tl.activation_cmd_idx >= 0:
                arrays[f"tl{i}_aci"] = np.array(
                    [tl.activation_cmd_idx], dtype=np.int32
                )
            if tl.cylinder_vertices is not None:
                arrays[f"tl{i}_cv"] = tl.cylinder_vertices
            if tl.laser_uid:
                encoded = tl.laser_uid.encode("utf-8")
                arrays[f"tl{i}_luid"] = np.frombuffer(
                    encoded, dtype=np.uint8
                ).copy()

        for i, ol in enumerate(self.overlay_layers):
            arrays[f"ol{i}_pos"] = ol.positions
            arrays[f"ol{i}_pow"] = ol.power_values
            arrays[f"ol{i}_lid"] = ol.laser_indices
            arrays[f"ol{i}_co"] = np.array(ol.cmd_offsets, dtype=np.int32)
            if ol.is_rotary:
                arrays[f"ol{i}_ir"] = np.array([1], dtype=np.int32)

        return arrays

    @classmethod
    def from_storage(
        cls: Type["CompiledSceneArtifact"],
        handle: BaseArtifactHandle,
        arrays: Dict[str, np.ndarray],
    ) -> "CompiledSceneArtifact":
        meta = arrays["_meta"]
        num_vl = int(meta[0])
        num_tl = int(meta[1])
        num_ol = int(meta[2])

        laser_uid_order = []
        if "_luo" in arrays:
            encoded = arrays["_luo"].tobytes()
            if encoded:
                laser_uid_order = encoded.decode("utf-8").split("\0")

        vertex_layers: List[VertexLayer] = []
        for i in range(num_vl):
            prefix = f"vl{i}_"
            pco_arr = arrays.get(f"{prefix}pco")
            tco_arr = arrays.get(f"{prefix}tco")
            is_rotary = False
            if f"{prefix}ir" in arrays:
                is_rotary = bool(arrays[f"{prefix}ir"][0])
            vl = VertexLayer(
                powered_verts=arrays[f"{prefix}pv"].copy(),
                power_values=arrays[f"{prefix}pvv"].copy(),
                laser_indices=arrays[f"{prefix}lid"].copy(),
                travel_verts=arrays[f"{prefix}tv"].copy(),
                zero_power_verts=arrays[f"{prefix}zpv"].copy(),
                powered_cmd_offsets=(
                    pco_arr.tolist() if pco_arr is not None else []
                ),
                travel_cmd_offsets=(
                    tco_arr.tolist() if tco_arr is not None else []
                ),
                is_rotary=is_rotary,
            )
            vertex_layers.append(vl)

        texture_layers: List[TextureLayer] = []
        for i in range(num_tl):
            prefix = f"tl{i}_"
            tl_meta = arrays[f"{prefix}meta"]
            width_px = int(tl_meta[0])
            height_px = int(tl_meta[1])
            cylinder_vertices = None
            if f"{prefix}cv" in arrays:
                cylinder_vertices = arrays[f"{prefix}cv"].copy()
            rotary_diameter = 0.0
            if f"{prefix}rd" in arrays:
                rotary_diameter = float(arrays[f"{prefix}rd"][0])
            rotary_enabled = False
            if f"{prefix}re" in arrays:
                rotary_enabled = bool(arrays[f"{prefix}re"][0])
            activation_cmd_idx = -1
            if f"{prefix}aci" in arrays:
                activation_cmd_idx = int(arrays[f"{prefix}aci"][0])
            laser_uid = ""
            if f"{prefix}luid" in arrays:
                laser_uid = arrays[f"{prefix}luid"].tobytes().decode("utf-8")
            tl = TextureLayer(
                power_texture=arrays[f"{prefix}tex"].copy(),
                width_px=width_px,
                height_px=height_px,
                model_matrix=arrays[f"{prefix}mm"].copy(),
                cylinder_vertices=cylinder_vertices,
                rotary_diameter=rotary_diameter,
                rotary_enabled=rotary_enabled,
                activation_cmd_idx=activation_cmd_idx,
                laser_uid=laser_uid,
            )
            texture_layers.append(tl)

        overlay_layers: List[ScanlineOverlayLayer] = []
        for i in range(num_ol):
            prefix = f"ol{i}_"
            is_rotary = False
            if f"{prefix}ir" in arrays:
                is_rotary = bool(arrays[f"{prefix}ir"][0])
            ol = ScanlineOverlayLayer(
                positions=arrays[f"{prefix}pos"].copy(),
                power_values=arrays[f"{prefix}pow"].copy(),
                laser_indices=arrays[f"{prefix}lid"].copy(),
                cmd_offsets=arrays[f"{prefix}co"].copy().tolist(),
                is_rotary=is_rotary,
            )
            overlay_layers.append(ol)

        return cls(
            generation_id=handle.generation_id,
            vertex_layers=vertex_layers,
            texture_layers=texture_layers,
            overlay_layers=overlay_layers,
            laser_uid_order=laser_uid_order,
        )
