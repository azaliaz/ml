import { createContext, useContext, type ReactNode } from "react";
import { useDriftVideos } from "../hooks/useDriftVideos";

type DriftVideosContextValue = ReturnType<typeof useDriftVideos>;

const DriftVideosContext = createContext<DriftVideosContextValue | null>(null);

export function DriftVideosProvider({ children }: { children: ReactNode }) {
  const value = useDriftVideos();
  return <DriftVideosContext.Provider value={value}>{children}</DriftVideosContext.Provider>;
}

export function useDriftVideosContext() {
  const ctx = useContext(DriftVideosContext);
  if (!ctx) throw new Error("useDriftVideosContext must be used within DriftVideosProvider");
  return ctx;
}
