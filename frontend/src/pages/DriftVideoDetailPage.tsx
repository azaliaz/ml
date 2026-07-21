import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { PageHeader } from "../components/layout/PageHeader";
import { Toggle } from "../components/Toggle";
import { useDriftVideosContext } from "../context/DriftVideosContext";
import type { DriftVideoConfig, EvalLabelMode, VideoSourceType } from "../types/drift";
import { WEEKDAY_LABELS } from "../types/drift";

const VIDEO_ACCEPT = ".mp4,.mov,.avi,.mkv,.webm,.m4v";
const SHOW_DEBUG_FIELDS = import.meta.env.DEV;

function toggleDay(days: number[], day: number): number[] {
  return days.includes(day) ? days.filter((d) => d !== day) : [...days, day].sort();
}

export function DriftVideoDetailPage() {
  const { videoId } = useParams<{ videoId: string }>();
  const navigate = useNavigate();
  const { getVideo, updateVideo, removeVideo } = useDriftVideosContext();

  const [form, setForm] = useState<DriftVideoConfig | null>(() =>
    videoId ? getVideo(videoId) : null,
  );
  const [processing, setProcessing] = useState(false);
  const [message, setMessage] = useState<{ type: "ok" | "err"; text: string } | null>(null);
  const [previewObjectUrl, setPreviewObjectUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!videoId) {
      setForm(null);
      return;
    }
    const video = getVideo(videoId);
    setForm(video ? { ...video } : null);
  }, [videoId]);

  useEffect(() => {
    return () => {
      if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
    };
  }, [previewObjectUrl]);

  const patch = (p: Partial<DriftVideoConfig>) => {
    setForm((prev) => (prev ? { ...prev, ...p } : prev));
  };

  const handleNameChange = (name: string) => {
    patch({ name });
    if (videoId) updateVideo(videoId, { name });
  };

  const handleSave = () => {
    if (!form || !videoId) return;
    updateVideo(videoId, form);
    setMessage({ type: "ok", text: "Настройки сохранены" });
  };

  const handleFile = (file: File | null) => {
    if (!file || !form) return;
    if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
    const url = URL.createObjectURL(file);
    setPreviewObjectUrl(url);
    patch({
      sourceType: "file",
      fileName: file.name,
      rtspUrl: null,
      previewUrl: url,
    });
  };

  const handleRun = async () => {
    if (!form || !videoId) return;
    if (form.sourceType === "file" && !form.fileName && !form.previewUrl) {
      setMessage({ type: "err", text: "Загрузите видеофайл или укажите RTSP URL." });
      return;
    }
    if (form.sourceType === "rtsp" && !form.rtspUrl?.trim()) {
      setMessage({ type: "err", text: "Укажите RTSP URL." });
      return;
    }

    setProcessing(true);
    setMessage(null);
    updateVideo(videoId, { ...form, status: "processing" });

    await new Promise((r) => setTimeout(r, 1200));

    updateVideo(videoId, {
      ...form,
      status: "active",
      lastProcessedAt: new Date().toISOString(),
      driftScore: Math.random() * 0.35,
    });
    setProcessing(false);
    setMessage({ type: "ok", text: "Обработка запущена. Метрики появятся в панели мониторинга." });
  };

  const previewSrc = useMemo(() => {
    if (!form) return null;
    if (form.sourceType === "file") return form.previewUrl ?? previewObjectUrl;
    return null;
  }, [form, previewObjectUrl]);

  if (!videoId || !form) {
    return (
      <div className="page-shell">
        <div className="alert alert-error">Видео не найдено.</div>
        <Link to="/drift" className="btn-secondary btn-fit">
          К списку видео
        </Link>
      </div>
    );
  }

  return (
    <div className="page-shell">
      <PageHeader
        backTo="/drift"
        backLabel="К списку видео"
        title={form.name || "Настройка видео"}
        subtitle="Загрузка источника, параметры детекции дрейфа и запуск обработки"
        actions={
          <button
            type="button"
            className="btn-secondary btn-compact"
            onClick={() => navigate(`/drift/monitor?video=${videoId}`)}
          >
            Мониторинг
          </button>
        }
      />

      <div className="layout-grid layout-grid-wide">
        <aside className="panel panel-sticky">
          <div className="panel-header">Параметры дрейфа</div>
          <div className="panel-body">
            <div className="form-section">
              <p className="form-section-title">Основное</p>
              <div className="field">
                <label htmlFor="video-name">Название видео</label>
                <input
                  id="video-name"
                  className="input"
                  value={form.name}
                  onChange={(e) => handleNameChange(e.target.value)}
                />
              </div>
            </div>

            <div className="form-section">
              <p className="form-section-title">Расписание</p>
              <div className="field">
                <label>Дни недели</label>
                <div className="schedule-days">
                  {WEEKDAY_LABELS.map((label, idx) => (
                    <button
                      key={label}
                      type="button"
                      className={`schedule-day${form.schedule.activeDays.includes(idx) ? " schedule-day-active" : ""}`}
                      onClick={() =>
                        patch({
                          schedule: {
                            ...form.schedule,
                            activeDays: toggleDay(form.schedule.activeDays, idx),
                          },
                        })
                      }
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="row-2">
                <div className="field">
                  <label htmlFor="start-hour">С часа</label>
                  <input
                    id="start-hour"
                    className="input"
                    type="number"
                    min={0}
                    max={23}
                    value={form.schedule.startHour}
                    onChange={(e) =>
                      patch({
                        schedule: { ...form.schedule, startHour: Number(e.target.value) },
                      })
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="end-hour">До часа</label>
                  <input
                    id="end-hour"
                    className="input"
                    type="number"
                    min={0}
                    max={23}
                    value={form.schedule.endHour}
                    onChange={(e) =>
                      patch({
                        schedule: { ...form.schedule, endHour: Number(e.target.value) },
                      })
                    }
                  />
                </div>
              </div>
              <p className="field-hint">Можно отключить детекцию ночью и в выходные.</p>
            </div>

            <div className="form-section">
              <p className="form-section-title">Детекция</p>
              <div className="field">
                <label htmlFor="object-classes">Классы COCO</label>
                <input
                  id="object-classes"
                  className="input"
                  value={form.objectClasses}
                  onChange={(e) => patch({ objectClasses: e.target.value })}
                  placeholder="person,car"
                />
              </div>
              <div className="row-2">
                <div className="field">
                  <label htmlFor="frame-stride">Шаг кадров</label>
                  <input
                    id="frame-stride"
                    className="input"
                    type="number"
                    min={1}
                    value={form.frameStride}
                    onChange={(e) => patch({ frameStride: Number(e.target.value) || 1 })}
                  />
                </div>
                <div className="field">
                  <label htmlFor="drift-window">Окно дрейфа, сек</label>
                  <input
                    id="drift-window"
                    className="input"
                    type="number"
                    min={1}
                    step={0.1}
                    value={form.driftWindowSec}
                    onChange={(e) => patch({ driftWindowSec: Number(e.target.value) || 10 })}
                  />
                </div>
              </div>
              <Toggle
                checked={form.onlyFramesWithDetections}
                onChange={(v) => patch({ onlyFramesWithDetections: v })}
                label="Только кадры с детекциями"
              />
            </div>

            {SHOW_DEBUG_FIELDS ? (
              <div className="form-section debug-panel">
                <p className="form-section-title debug-panel-title">Отладка (dev)</p>
                <div className="field">
                  <label htmlFor="video-id">video_id</label>
                  <input
                    id="video-id"
                    className="input"
                    value={form.videoId ?? ""}
                    onChange={(e) => patch({ videoId: e.target.value || undefined })}
                  />
                </div>
                <div className="field">
                  <label htmlFor="segments-file">segments_file</label>
                  <input
                    id="segments-file"
                    className="input"
                    value={form.segmentsFile ?? ""}
                    onChange={(e) => patch({ segmentsFile: e.target.value || undefined })}
                    placeholder="segments.txt"
                  />
                </div>
                <div className="row-2">
                  <div className="field">
                    <label htmlFor="eval-window">eval_transition_window_sec</label>
                    <input
                      id="eval-window"
                      className="input"
                      type="number"
                      step={0.1}
                      value={form.evalTransitionWindowSec ?? ""}
                      onChange={(e) =>
                        patch({
                          evalTransitionWindowSec: e.target.value ? Number(e.target.value) : undefined,
                        })
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="eval-fpr">eval_fpr_target</label>
                    <input
                      id="eval-fpr"
                      className="input"
                      type="number"
                      step={0.001}
                      value={form.evalFprTarget ?? ""}
                      onChange={(e) =>
                        patch({ evalFprTarget: e.target.value ? Number(e.target.value) : undefined })
                      }
                    />
                  </div>
                </div>
                <div className="row-2">
                  <div className="field">
                    <label htmlFor="eval-n">eval_update_every_n</label>
                    <input
                      id="eval-n"
                      className="input"
                      type="number"
                      value={form.evalUpdateEveryN ?? ""}
                      onChange={(e) =>
                        patch({ evalUpdateEveryN: e.target.value ? Number(e.target.value) : undefined })
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="eval-mode">eval_label_mode</label>
                    <select
                      id="eval-mode"
                      className="select"
                      value={form.evalLabelMode ?? "symmetric"}
                      onChange={(e) => patch({ evalLabelMode: e.target.value as EvalLabelMode })}
                    >
                      <option value="symmetric">symmetric</option>
                      <option value="post_only">post_only</option>
                    </select>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
          <div className="panel-footer">
            <button type="button" className="btn-secondary btn-fit" onClick={handleSave}>
              Сохранить настройки
            </button>
          </div>
        </aside>

        <main className="panel">
          <div className="panel-header">Видео и запуск</div>
          <div className="panel-body">
            <div className="segmented-control">
              <button
                type="button"
                className={`segmented-item${form.sourceType === "file" ? " segmented-item-active" : ""}`}
                onClick={() => patch({ sourceType: "file" as VideoSourceType })}
              >
                Файл
              </button>
              <button
                type="button"
                className={`segmented-item${form.sourceType === "rtsp" ? " segmented-item-active" : ""}`}
                onClick={() => patch({ sourceType: "rtsp" as VideoSourceType })}
              >
                RTSP поток
              </button>
            </div>

            {form.sourceType === "file" ? (
              <div className="dropzone dropzone-compact">
                <p>
                  Загрузите видео (mp4, avi, mov, mkv) или{" "}
                  <strong>
                    <label className="file-picker-label">
                      выберите файл
                      <input
                        type="file"
                        accept={VIDEO_ACCEPT}
                        hidden
                        onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
                      />
                    </label>
                  </strong>
                </p>
                {form.fileName ? <p className="field-hint">Файл: {form.fileName}</p> : null}
              </div>
            ) : (
              <div className="field">
                <label htmlFor="rtsp-url">RTSP URL</label>
                <input
                  id="rtsp-url"
                  className="input"
                  value={form.rtspUrl ?? ""}
                  onChange={(e) => patch({ rtspUrl: e.target.value, sourceType: "rtsp" })}
                  placeholder="rtsp://..."
                />
              </div>
            )}

            <div className="video-preview-frame">
              {previewSrc ? (
                <video src={previewSrc} controls className="video-preview-player" />
              ) : form.sourceType === "rtsp" && form.rtspUrl ? (
                <div className="video-preview-placeholder">
                  <p className="empty-state-title">RTSP превью</p>
                  <p className="field-hint">Подключится после запуска обработки</p>
                  <p className="video-preview-url">{form.rtspUrl}</p>
                </div>
              ) : (
                <div className="video-preview-placeholder">
                  <p className="empty-state-title">Превью видео</p>
                  <p className="field-hint">Появится после загрузки файла</p>
                </div>
              )}
            </div>

            {message ? (
              <div className={`alert ${message.type === "ok" ? "alert-success" : "alert-error"}`}>{message.text}</div>
            ) : null}
          </div>

          <div className="panel-footer panel-footer-split">
            <button type="button" className="btn-primary btn-fit" disabled={processing} onClick={handleRun}>
              {processing ? (
                <>
                  <span className="spinner" aria-hidden />
                  Обработка…
                </>
              ) : (
                "Запустить обработку"
              )}
            </button>
            <button
              type="button"
              className="btn-secondary btn-fit"
              onClick={() => navigate(`/drift/monitor?video=${videoId}`)}
            >
              Открыть мониторинг
            </button>
            <button
              type="button"
              className="btn-secondary btn-fit btn-danger-text"
              onClick={() => {
                removeVideo(videoId);
                navigate("/drift");
              }}
            >
              Удалить
            </button>
          </div>
        </main>
      </div>
    </div>
  );
}
