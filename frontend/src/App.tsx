import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { DriftVideosProvider } from "./context/DriftVideosContext";
import { DriftMonitorPage } from "./pages/DriftMonitorPage";
import { DriftVideoDetailPage } from "./pages/DriftVideoDetailPage";
import { DriftVideosPage } from "./pages/DriftVideosPage";
import { PreannotationPage } from "./pages/PreannotationPage";

export function App() {
  return (
    <DriftVideosProvider>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Navigate to="/preannotation" replace />} />
          <Route path="preannotation" element={<PreannotationPage />} />
          <Route path="drift" element={<DriftVideosPage />} />
          <Route path="drift/monitor" element={<DriftMonitorPage />} />
          <Route path="drift/:videoId" element={<DriftVideoDetailPage />} />
        </Route>
      </Routes>
    </DriftVideosProvider>
  );
}
