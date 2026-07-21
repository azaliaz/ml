import { useCallback, useEffect, useState } from "react";
import type { DriftVideoConfig } from "../types/drift";
import { createDefaultDriftVideo } from "../types/drift";

const STORAGE_KEY = "cv-platform-drift-videos";

const STALE_STORAGE_KEYS = ["cv-platform-drift-videos-v2", "cv-platform-drift-videos-version"];

const REMOVED_DEMO_IDS = new Set(["demo-1", "demo-2", "demo-3", "demo-4", "demo-5", "demo-6", "demo-7", "demo-8"]);

function cleanupStaleStorage() {
  for (const key of STALE_STORAGE_KEYS) {
    localStorage.removeItem(key);
  }
}

function sanitizeVideos(videos: DriftVideoConfig[]): DriftVideoConfig[] {
  return videos.filter((v) => !REMOVED_DEMO_IDS.has(v.id));
}

function loadVideos(): DriftVideoConfig[] {
  cleanupStaleStorage();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as DriftVideoConfig[];
    return sanitizeVideos(parsed);
  } catch {
    return [];
  }
}

function saveVideos(videos: DriftVideoConfig[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(videos));
}

export function useDriftVideos() {
  const [videos, setVideos] = useState<DriftVideoConfig[]>(() => loadVideos());

  useEffect(() => {
    saveVideos(videos);
  }, [videos]);

  const getVideo = useCallback(
    (id: string) => videos.find((v) => v.id === id) ?? null,
    [videos],
  );

  const addVideo = useCallback((name?: string) => {
    const video = createDefaultDriftVideo(name ?? `Видео ${videos.length + 1}`);
    setVideos((prev) => [video, ...prev]);
    return video;
  }, [videos.length]);

  const updateVideo = useCallback((id: string, patch: Partial<DriftVideoConfig>) => {
    setVideos((prev) =>
      prev.map((v) =>
        v.id === id ? { ...v, ...patch, updatedAt: new Date().toISOString() } : v,
      ),
    );
  }, []);

  const removeVideo = useCallback((id: string) => {
    setVideos((prev) => prev.filter((v) => v.id !== id));
  }, []);

  return { videos, getVideo, addVideo, updateVideo, removeVideo };
}
