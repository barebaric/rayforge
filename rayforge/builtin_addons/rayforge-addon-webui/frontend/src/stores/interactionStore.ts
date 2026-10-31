import { createStore } from "zustand/vanilla";
import type { Size2, Vec2 } from "../api/types";

export interface TransformPreview {
  uid: string;
  pos: Vec2;
  size: Size2;
  angleDeg: number;
}

interface InteractionState {
  /** Live transform values of the gizmo drag in progress, if any. */
  transformPreview: TransformPreview | null;
}

export const interactionStore = createStore<InteractionState>()(() => ({
  transformPreview: null,
}));

export function setTransformPreview(preview: TransformPreview): void {
  interactionStore.setState({ transformPreview: preview });
}

export function clearTransformPreview(): void {
  interactionStore.setState({ transformPreview: null });
}
