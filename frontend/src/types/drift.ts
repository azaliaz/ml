export type VideoSourceType = "file" | "rtsp";

export type DriftVideoStatus = "idle" | "processing" | "active" | "drift_detected" | "paused";

export type EvalLabelMode = "symmetric" | "post_only";

export interface DriftSchedule {
  /** Дни недели: 0 = пн, 6 = вс */
  activeDays: number[];
  startHour: number;
  endHour: number;
}

export interface DriftVideoConfig {
  id: string;
  name: string;
  sourceType: VideoSourceType;
  fileName?: string | null;
  rtspUrl?: string | null;
  previewUrl?: string | null;
  schedule: DriftSchedule;
  objectClasses: string;
  frameStride: number;
  driftWindowSec: number;
  onlyFramesWithDetections: boolean;
  videoId?: string;
  segmentsFile?: string;
  evalTransitionWindowSec?: number;
  evalFprTarget?: number;
  evalUpdateEveryN?: number;
  evalLabelMode?: EvalLabelMode;
  status: DriftVideoStatus;
  lastProcessedAt?: string | null;
  driftScore?: number | null;
  createdAt: string;
  updatedAt: string;
}

export const WEEKDAY_LABELS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"] as const;

export function createDefaultDriftVideo(name: string): DriftVideoConfig {
  const now = new Date().toISOString();
  return {
    id: crypto.randomUUID(),
    name,
    sourceType: "file",
    fileName: null,
    rtspUrl: null,
    previewUrl: null,
    schedule: {
      activeDays: [0, 1, 2, 3, 4],
      startHour: 8,
      endHour: 22,
    },
    objectClasses: "person,car",
    frameStride: 5,
    driftWindowSec: 10,
    onlyFramesWithDetections: false,
    evalLabelMode: "symmetric",
    status: "idle",
    createdAt: now,
    updatedAt: now,
  };
}
