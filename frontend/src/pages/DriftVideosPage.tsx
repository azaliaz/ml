import { Link, useNavigate } from "react-router-dom";
import { VideoSourcesEmptyIllustration } from "../components/illustrations/VideoSourcesEmptyIllustration";
import { driftStatusLabel, driftStatusVariant, StatusChip } from "../components/StatusChip";
import { PageHeader } from "../components/layout/PageHeader";
import { useDriftVideosContext } from "../context/DriftVideosContext";

function formatSourcesCount(count: number): string {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return `${count} источник`;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) return `${count} источника`;
  return `${count} источников`;
}

function formatSourceMeta(video: {
  sourceType: string;
  fileName?: string | null;
  rtspUrl?: string | null;
}): string {
  if (video.sourceType === "rtsp") {
    return video.rtspUrl?.trim() || "RTSP поток не задан";
  }
  return video.fileName?.trim() || "Файл не загружен";
}

export function DriftVideosPage() {
  const navigate = useNavigate();
  const { videos, addVideo } = useDriftVideosContext();
  const isEmpty = videos.length === 0;

  const handleAdd = () => {
    const video = addVideo();
    navigate(`/drift/${video.id}`);
  };

  return (
    <div className="page-shell">
      <PageHeader
        title="Модуль детекции дрейфа"
        subtitle="Добавьте видео или RTSP-поток, настройте параметры и запустите мониторинг дрейфа"
        actions={
          !isEmpty ? (
            <button type="button" className="btn-primary btn-compact" onClick={handleAdd}>
              Добавить видео
            </button>
          ) : null
        }
      />

      <section className="panel">
        <div className="panel-header panel-header-row">
          <span>Источники видео</span>
          {!isEmpty ? <span className="panel-header-meta">{formatSourcesCount(videos.length)}</span> : null}
        </div>

        {isEmpty ? (
          <div className="panel-body panel-body-empty">
            <div className="empty-state empty-state-centered">
              <VideoSourcesEmptyIllustration />
              <p className="empty-state-title">Источники видео ещё не добавлены</p>
              <p className="empty-state-text">
                Создайте первый источник: загрузите файл mp4/avi/mov/mkv или подключите RTSP-поток.
              </p>
              <button type="button" className="btn-primary btn-fit" onClick={handleAdd}>
                Добавить новое видео
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="panel-body">
              <div className="video-grid">
                {videos.map((video) => (
                  <Link key={video.id} to={`/drift/${video.id}`} className="video-card">
                    <div className="video-card-icon" aria-hidden>
                      {video.sourceType === "rtsp" ? "RTSP" : "MP4"}
                    </div>
                    <div className="video-card-body">
                      <div className="video-card-top">
                        <span className="video-card-title">{video.name}</span>
                        <StatusChip
                          label={driftStatusLabel(video.status)}
                          variant={driftStatusVariant(video.status)}
                        />
                      </div>
                      <p className="video-card-meta">{formatSourceMeta(video)}</p>
                      <div className="video-card-metric">
                        <span>Drift score</span>
                        <strong>{video.driftScore != null ? video.driftScore.toFixed(2) : "—"}</strong>
                      </div>
                    </div>
                    <span className="video-card-chevron" aria-hidden>
                      →
                    </span>
                  </Link>
                ))}
              </div>
            </div>
            <div className="panel-footer panel-footer-center">
              <button type="button" className="btn-secondary btn-fit" onClick={handleAdd}>
                Добавить ещё видео
              </button>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
