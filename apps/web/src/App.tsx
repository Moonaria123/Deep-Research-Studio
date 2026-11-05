import { Routes, Route, Navigate } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Toaster } from "./components/ui/toast";
import { Home } from "./pages/Home";
import { RunView } from "./pages/RunView";
import { History } from "./pages/History";
import { SessionThread } from "./pages/SessionThread";
import { ArchivedHistory } from "./pages/ArchivedHistory";
import { Library } from "./pages/Library";
import { Memory } from "./pages/Memory";
import { Graph } from "./pages/Graph";
import { Settings } from "./pages/Settings";

export default function App() {
  return (
    <>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Home />} />
          <Route path="run/:runId" element={<RunView />} />
          <Route path="session/:sessionId" element={<SessionThread />} />
          <Route path="history" element={<History />} />
          <Route path="history/archived" element={<ArchivedHistory />} />
          <Route path="library/sources/:documentId" element={<Library />} />
          <Route path="library/reports/:reportId" element={<Library />} />
          <Route path="library/notes/:noteId" element={<Library />} />
          <Route path="library/reports" element={<Library />} />
          <Route path="library/notes" element={<Library />} />
          <Route path="library" element={<Library />} />
          <Route path="memory" element={<Memory />} />
          <Route path="memory/:memoryId" element={<Memory />} />
          <Route path="graph" element={<Graph />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
      <Toaster />
    </>
  );
}
