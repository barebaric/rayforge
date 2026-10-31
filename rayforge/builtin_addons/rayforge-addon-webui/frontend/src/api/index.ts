import { isMockMode } from "../env";
import { createMockBackend } from "../dev/mock";
import { ApiClient } from "./client";
import type { ApiBackend } from "./client";

export function createApi(): ApiBackend {
  return isMockMode() ? createMockBackend() : new ApiClient();
}

export { ApiError } from "./client";
export type { ApiBackend } from "./client";
