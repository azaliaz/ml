/**
 * Превью предразметки после успешного ответа API — это data:image из preview_base64:
 * кадр из ZIP, который вернул ML (папка previews/), т.е. размечается то, что вы загрузили.
 * Статичные демо-кадры до запуска — опционально: SHOW_MARKETING_SAMPLES + файлы в public/marketing/.
 * Подробно: src/assets/DESIGN_ASSETS.md
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { PreviewLightbox } from "../components/PreviewLightbox";
import { PageHeader } from "../components/layout/PageHeader";
import { HeroIllustration } from "../components/illustrations/HeroIllustration";
import { UploadEmptyIllustration } from "../components/illustrations/UploadEmptyIllustration";
import { Toggle } from "../components/Toggle";

/** true — показать ряд демо-картинок; положите sample-1.jpg … sample-3.jpg в public/marketing/ */
const SHOW_MARKETING_SAMPLES = true;

type TaskType = "detection" | "segmentation" | "classification" | "other";

const TASK_LABELS: Record<TaskType, string> = {
  detection: "Детекция",
  segmentation: "Сегментация",
  classification: "Классификация",
  other: "Другое",
};

const EXPORT_FORMATS = ["COCO 1.0", "YOLO 1.1", "Pascal VOC 1.1"] as const;

interface PreannotationInfo {
  status?: string | null;
  archive_path?: string | null;
  archive_size?: number | null;
  preview_filename?: string | null;
  preview_base64?: string | null;
  import_request_id?: number | string | null;
  error?: string | null;
  archive_stats?: {
    images_count?: number;
    annotations_count?: number;
    categories_count?: number;
  } | null;
}

interface UploadResult {
  task_id: number;
  task_name: string;
  media_type?: "image" | "video";
  warnings: string[];
  video?: {
    frame_step?: number;
    start_frame?: number;
    stop_frame?: number | null;
    max_frames?: number | null;
    cvat_frame_count?: number;
  };
  preannotation: PreannotationInfo | null;
}

const VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"];

function isVideoFile(file: File): boolean {
  const lower = file.name.toLowerCase();
  return VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

interface MlHealth {
  ok: boolean;
  error?: string;
  data?: unknown;
}

async function readErrorDetail(res: Response): Promise<string> {
  try {
    const j = await res.json();
    if (typeof j?.detail === "string") return j.detail;
    if (Array.isArray(j?.detail)) return j.detail.map((x: { msg?: string }) => x.msg ?? JSON.stringify(x)).join("; ");
    return JSON.stringify(j);
  } catch {
    return await res.text();
  }
}

type ClassificationMode = "object" | "image";

export function PreannotationPage() {
  const [taskType, setTaskType] = useState<TaskType>("detection");
  const [classificationMode, setClassificationMode] = useState<ClassificationMode>("object");
  const [classesText, setClassesText] = useState("object\nperson");
  const [classDescription, setClassDescription] = useState("");
  const [scoreThreshold, setScoreThreshold] = useState(0.3);
  const [maxBoxes, setMaxBoxes] = useState(10);
  const [useClip, setUseClip] = useState(false);
  const [useQwen, setUseQwen] = useState(false);
  const [qwenInstruction, setQwenInstruction] = useState(
      "Look at the image and return ONLY valid JSON with no explanations, no markdown, and no extra text: {\"class_names\":[...],\"text_prompts\":[...]}. class_names must be short CVAT class names in English. text_prompts must be short descriptive prompts in English, strictly 1:1 aligned with class_names. Both arrays must have the same length. Use concise object-level labels. If no objects are detected, return {\"class_names\":[],\"text_prompts\":[]}."
  );
  const [runPreannot, setRunPreannot] = useState(true);
  const [taskName, setTaskName] = useState("ml_preannot_task");
  const [videoFrameStep, setVideoFrameStep] = useState(5);
  const [videoMaxFrames, setVideoMaxFrames] = useState(300);

  const [files, setFiles] = useState<File[]>([]);
  const [dragActive, setDragActive] = useState(false);

  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [preannStatus, setPreannStatus] = useState<string | null>(null);
  const [preannPhase, setPreannPhase] = useState<string | null>(null);

  const [mlHealth, setMlHealth] = useState<MlHealth | null>(null);
  const [mlHealthState, setMlHealthState] = useState<"checking" | "ok" | "degraded" | "down">("checking");
  const mlHealthFailuresRef = useRef(0);

  const [exportFormat, setExportFormat] = useState<string>("COCO 1.0");
  const [includeImages, setIncludeImages] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const [reviewerUser, setReviewerUser] = useState("");
  const [grantLoading, setGrantLoading] = useState(false);
  const [grantMessage, setGrantMessage] = useState<{ type: "ok" | "err"; text: string } | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);

  const refreshMlHealth = useCallback(async () => {
    try {
      const r = await fetch("/api/ml/health");
      const d = (await r.json()) as MlHealth;
      if (!r.ok || d.ok !== true) {
        throw new Error(d.error || `status ${r.status}`);
      }
      mlHealthFailuresRef.current = 0;
      setMlHealth(d);
      setMlHealthState("ok");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Сеть";
      mlHealthFailuresRef.current += 1;
      setMlHealth((prev) => prev ?? { ok: false, error: msg });
      setMlHealthState((prev) => {
        if (prev === "ok" && mlHealthFailuresRef.current < 3) return "degraded";
        if (prev === "checking") return "down";
        return mlHealthFailuresRef.current >= 3 ? "down" : prev;
      });
    }
  }, []);

  useEffect(() => {
    refreshMlHealth();
    const id = setInterval(refreshMlHealth, 30000);
    return () => clearInterval(id);
  }, [refreshMlHealth]);

  const onFiles = useCallback((list: FileList | File[]) => {
    const incoming = Array.from(list);
    if (!incoming.length) return;
    setFiles((prev) => {
      const hasVideo = prev.some(isVideoFile) || incoming.some(isVideoFile);
      const hasImage = prev.some((f) => !isVideoFile(f)) || incoming.some((f) => !isVideoFile(f));
      if (hasVideo && hasImage) {
        setUploadError("Нельзя смешивать видео и изображения в одной задаче.");
        return prev;
      }
      if (incoming.filter(isVideoFile).length > 1 || (prev.some(isVideoFile) && incoming.some(isVideoFile))) {
        setUploadError("В одной задаче допускается только одно видео.");
        return prev;
      }
      if (incoming.some(isVideoFile)) {
        return incoming.filter(isVideoFile).slice(0, 1);
      }
      setUploadError(null);
      return [...prev, ...incoming];
    });
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (preannotationInProgress) return;
    if (e.dataTransfer.files?.length) onFiles(e.dataTransfer.files);
  };

  const parseClasses = (): string[] =>
    classesText
      .split("\n")
      .map((c) => c.trim())
      .filter(Boolean);

  useEffect(() => {
    if (taskType === "classification") {
      setUseClip(classificationMode === "image");
    }
  }, [taskType, classificationMode]);

  const hasVideoSelected = files.some(isVideoFile);

  const handleUpload = async () => {
    setUploadError(null);
    setUploadResult(null);
    setPreannPhase(null);
    setPreviewOpen(false);
    if (!files.length) {
      setUploadError("Добавьте хотя бы одно изображение или видео.");
      return;
    }
    const form = new FormData();
    const payload: Record<string, unknown> = {
      task_name: taskName,
      task_type: taskType,
      classification_mode: classificationMode,
      classes: parseClasses(),
      class_description: classDescription,
      score_threshold: scoreThreshold,
      max_boxes: maxBoxes,
      use_clip: useClip,
      use_qwen: useQwen,
      qwen_instruction: qwenInstruction,
      run_preannot: runPreannot,
    };
    if (hasVideoSelected) {
      payload.video_frame_step = videoFrameStep;
      payload.video_max_frames = videoMaxFrames;
    }
    form.append("payload_json", JSON.stringify(payload));
    for (const f of files) form.append("files", f);

    setUploading(true);
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 4 * 60 * 1000);
    try {
      const res = await fetch("/api/tasks/upload", { method: "POST", body: form, signal: controller.signal });
      if (!res.ok) throw new Error(await readErrorDetail(res));
      const data = (await res.json()) as UploadResult;
      setUploadResult(data);
      setPreannStatus(data.preannotation?.status ?? null);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        setUploadError("Таймаут запроса: предразметка выполняется слишком долго. Проверьте логи ml_service.");
      } else {
        setUploadError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      window.clearTimeout(timeoutId);
      setUploading(false);
    }
  };

  const taskId = uploadResult?.task_id;
  const preannotationInProgress =
    uploading ||
    preannStatus === "processing" ||
    uploadResult?.preannotation?.status === "processing";

  useEffect(() => {
    if (!taskId) return;
    if (preannStatus !== "processing") return;
    const id = window.setInterval(async () => {
      try {
        const r = await fetch(`/api/tasks/${taskId}/preannotation-status`);
        if (!r.ok) return;
        const st = (await r.json()) as PreannotationInfo & {
          error?: string;
          phase?: string;
          frames_done?: number;
          frames_total?: number;
        };
        if (st.phase) setPreannPhase(st.phase);
        if (st.status === "done") {
          setUploadResult((prev) =>
            prev
              ? {
                  ...prev,
                  preannotation: {
                    status: "done",
                    archive_path: st.archive_path ?? null,
                    archive_size: st.archive_size ?? null,
                    preview_filename: st.preview_filename ?? null,
                    preview_base64: st.preview_base64 ?? null,
                    import_request_id: st.import_request_id ?? null,
                    archive_stats: st.archive_stats ?? null,
                  },
                }
              : prev,
          );
          setPreannStatus("done");
          window.clearInterval(id);
        } else if (st.status === "error") {
          setUploadError(st.error ?? "Ошибка фоновой предразметки.");
          setUploadResult((prev) =>
            prev
              ? {
                  ...prev,
                  preannotation: {
                    ...(prev.preannotation ?? {}),
                    status: "error",
                    error: st.error ?? "Ошибка фоновой предразметки.",
                  },
                }
              : prev,
          );
          setPreannStatus("error");
          window.clearInterval(id);
        }
      } catch {
        // ignore transient polling errors
      }
    }, 4000);
    return () => window.clearInterval(id);
  }, [taskId, preannStatus]);

  const handleExport = async () => {
    if (taskId == null) return;
    setExportError(null);
    setExporting(true);
    try {
      const res = await fetch(`/api/tasks/${taskId}/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          format_name: exportFormat,
          include_images: includeImages,
          task_name: taskName,
        }),
      });
      if (!res.ok) throw new Error(await readErrorDetail(res));
      const blob = await res.blob();
      const cd = res.headers.get("Content-Disposition");
      let filename = `${taskName}_annotations.zip`;
      const m = cd?.match(/filename="?([^";]+)"?/);
      if (m) filename = m[1];
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setExportError(e instanceof Error ? e.message : String(e));
    } finally {
      setExporting(false);
    }
  };

  const handleGrant = async (e: React.FormEvent) => {
    e.preventDefault();
    setGrantMessage(null);
    if (taskId == null || !reviewerUser.trim()) {
      setGrantMessage({ type: "err", text: "Укажите валидатора и создайте задачу." });
      return;
    }
    setGrantLoading(true);
    try {
      const res = await fetch(`/api/tasks/${taskId}/grant-validation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviewer_user: reviewerUser.trim() }),
      });
      let data: Record<string, unknown> = {};
      try {
        data = (await res.json()) as Record<string, unknown>;
      } catch {
        /* empty */
      }
      if (!res.ok) {
        const d = data.detail;
        throw new Error(typeof d === "string" ? d : JSON.stringify(data) || res.statusText);
      }

      const failed = (data.jobs_failed as unknown[])?.length ?? 0;
      const patched = (data.jobs_patched as unknown[])?.length ?? 0;
      if (failed || !patched) {
        const codes: number[] = [];
        for (const f of (data.jobs_failed as { last_response?: { status_code?: number } }[]) || []) {
          const sc = f?.last_response?.status_code;
          if (typeof sc === "number") codes.push(sc);
        }
        if (codes.includes(403)) {
          setGrantMessage({
            type: "err",
            text: "403: нет прав назначать пользователей на job. Используйте токен admin/owner.",
          });
        } else {
          setGrantMessage({ type: "err", text: "Не удалось назначить валидатора. Проверьте ответ сервера." });
        }
        return;
      }
      setGrantMessage({ type: "ok", text: `Валидатор назначен: ${reviewerUser.trim()}` });
    } catch (err) {
      setGrantMessage({ type: "err", text: err instanceof Error ? err.message : String(err) });
    } finally {
      setGrantLoading(false);
    }
  };

  // Живое превью: первый файл из previews/ внутри ZIP от ML (не фиксированный ассет из дизайна).
  const previewSrc =
    uploadResult?.preannotation?.preview_base64 != null
      ? `data:image/png;base64,${uploadResult.preannotation.preview_base64}`
      : null;
  const previewFilename = uploadResult?.preannotation?.preview_filename ?? null;
  const preannDone = uploadResult?.preannotation?.status === "done";
  const preannStats = uploadResult?.preannotation?.archive_stats;

  return (
    <div className="page-shell">
      {previewOpen && previewSrc ? (
        <PreviewLightbox
          src={previewSrc}
          alt={previewFilename ?? "Превью предразметки"}
          filename={previewFilename}
          onClose={() => setPreviewOpen(false)}
        />
      ) : null}
      <PageHeader
        title="Автоматизированная предразметка CVAT"
        subtitle="Меньше ручной работы — больше результата"
        actions={<HeroIllustration />}
      />

      <div className="layout-grid">
        <aside className="panel">
          <div className="panel-header">Параметры модели</div>
          <fieldset className="panel-body panel-fieldset" disabled={preannotationInProgress}>
            <div className="field">
              <label htmlFor="task-type">Тип задачи</label>
              <select
                id="task-type"
                className="select"
                value={taskType}
                onChange={(e) => setTaskType(e.target.value as TaskType)}
              >
                {(Object.keys(TASK_LABELS) as TaskType[]).map((k) => (
                  <option key={k} value={k}>
                    {TASK_LABELS[k]}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label htmlFor="classes">Классы (по одному в строке)</label>
              <textarea
                id="classes"
                className="textarea"
                value={classesText}
                onChange={(e) => setClassesText(e.target.value)}
                rows={5}
              />
            </div>

            <div className="field">
              <label htmlFor="desc">Описание меток (для всех классов)</label>
              <textarea
                id="desc"
                className="textarea"
                value={classDescription}
                onChange={(e) => setClassDescription(e.target.value)}
                rows={3}
                placeholder="Необязательно"
              />
            </div>

            <div className="field">
              <label>Порог score (детекция)</label>
              <div className="range-wrap">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={scoreThreshold}
                  onChange={(e) => setScoreThreshold(Number(e.target.value))}
                />
                <span className="range-value">{scoreThreshold.toFixed(2)}</span>
              </div>
            </div>

            <div className="row-2">
              <div className="field">
                <label htmlFor="max-boxes">Max боксов</label>
                <input
                  id="max-boxes"
                  className="input"
                  type="number"
                  min={1}
                  max={200}
                  value={maxBoxes}
                  onChange={(e) => setMaxBoxes(Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label htmlFor="tname">Имя задачи</label>
                <input
                  id="tname"
                  className="input"
                  value={taskName}
                  onChange={(e) => setTaskName(e.target.value)}
                />
              </div>
            </div>

            {hasVideoSelected ? (
              <>
                <div className="field">
                  <label htmlFor="video-step">Шаг кадров (каждый N-й кадр видео)</label>
                  <input
                    id="video-step"
                    className="input"
                    type="number"
                    min={1}
                    max={120}
                    value={videoFrameStep}
                    onChange={(e) => setVideoFrameStep(Math.max(1, Number(e.target.value) || 1))}
                  />
                </div>
                <div className="field">
                  <label htmlFor="video-max">Макс. кадров в задаче CVAT</label>
                  <input
                    id="video-max"
                    className="input"
                    type="number"
                    min={1}
                    max={5000}
                    value={videoMaxFrames}
                    onChange={(e) => setVideoMaxFrames(Math.max(1, Number(e.target.value) || 1))}
                  />
                  <p className="field-hint">Видео загружается в CVAT как video task; предразметка идёт по извлечённым кадрам.</p>
                </div>
              </>
            ) : null}

            <Toggle
              checked={useClip}
              onChange={setUseClip}
              label="SigLIP2 (уточнение класса по crop)"
              disabled={preannotationInProgress || (taskType === "classification" && classificationMode === "image")}
            />

            {taskType === "classification" ? (
              <div className="field">
                <label htmlFor="classification-mode">Режим классификации</label>
                <select
                  id="classification-mode"
                  className="input"
                  value={classificationMode}
                  onChange={(e) => setClassificationMode(e.target.value as ClassificationMode)}
                  disabled={preannotationInProgress}
                >
                  <option value="object">Объекты (GND bbox → SigLIP2)</option>
                  <option value="image">Весь кадр (SigLIP2 на изображении)</option>
                </select>
              </div>
            ) : null}
            <Toggle checked={useQwen} onChange={setUseQwen} label="Qwen — промпты" disabled={preannotationInProgress} />

            {useQwen ? (
              <div className="field">
                <label htmlFor="qwen">Инструкция для Qwen</label>
                <textarea
                  id="qwen"
                  className="textarea"
                  value={qwenInstruction}
                  onChange={(e) => setQwenInstruction(e.target.value)}
                  rows={4}
                />
              </div>
            ) : null}

            <Toggle
              checked={runPreannot}
              onChange={setRunPreannot}
              label="Предразметка перед импортом в CVAT"
              disabled={preannotationInProgress}
            />
          </fieldset>
        </aside>

        <main className="panel">
          <div className="panel-header">Файлы и результат</div>
          <div className="panel-body">
            <div
              className="dropzone"
              data-active={dragActive}
              onDragOver={(e) => {
                if (preannotationInProgress) return;
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => {
                if (preannotationInProgress) return;
                setDragActive(false);
              }}
              onDrop={onDrop}
            >
              {/* Пустое состояние: иллюстрация — править в components/illustrations/UploadEmptyIllustration.tsx */}
              {files.length === 0 ? <UploadEmptyIllustration /> : null}
              <p>
                Перетащите изображения или видео сюда или{" "}
                <strong>
                  <label style={{ cursor: "pointer", textDecoration: "underline" }}>
                    выберите файлы
                    <input
                      type="file"
                      accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.mp4,.mov,.avi,.mkv,.webm,.m4v"
                      multiple
                      disabled={preannotationInProgress}
                      hidden
                      onChange={(e) => {
                        if (preannotationInProgress) return;
                        if (e.target.files?.length) onFiles(e.target.files);
                        e.target.value = "";
                      }}
                    />
                  </label>
                </strong>
              </p>
              <p className="field-hint" style={{ marginTop: 12 }}>
                Изображения: JPG, PNG, BMP, TIFF · Видео: MP4, MOV, AVI (одно видео на задачу)
              </p>
            </div>

            {/* Статичные примеры разметки: public/marketing/sample-1.jpg … — см. DESIGN_ASSETS.md */}
            {SHOW_MARKETING_SAMPLES && files.length === 0 && !uploadResult ? (
              <div className="marketing-samples">
                <p className="marketing-samples-label">Пример результата (демо)</p>
                <div className="marketing-samples-row">
                  <img src="/marketing/sample-1.jpg" alt="Демо: пример разметки 1" />
                  <img src="/marketing/sample-2.jpg" alt="Демо: пример разметки 2" />
                  <img src="/marketing/sample-3.jpg" alt="Демо: пример разметки 3" />
                </div>
              </div>
            ) : null}

            {files.length > 0 ? (
              <ul className="file-list">
                {files.map((f, i) => (
                  <li key={`${f.name}-${i}`}>
                    <span>{f.name}</span>
                    <button
                      type="button"
                      disabled={preannotationInProgress}
                      onClick={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
                    >
                      Убрать
                    </button>
                  </li>
                ))}
              </ul>
            ) : null}

            <button
              type="button"
              className="btn-primary"
              disabled={uploading || preannotationInProgress}
              onClick={handleUpload}
              style={{ width: "100%", marginTop: 12 }}
            >
              {uploading ? (
                <>
                  <span className="spinner" aria-hidden />
                  {runPreannot ? "Загрузка и предразметка…" : "Загрузка в CVAT…"}
                </>
              ) : (
                "Загрузить в CVAT"
              )}
            </button>

            {uploadError ? <div className="alert alert-error">{uploadError}</div> : null}

            {uploadResult ? (
              <section className="result-card">
                <div className="result-card-header">
                  <h2 className="section-title">Задача создана</h2>
                  {preannDone ? <span className="status-chip status-chip-ok">Предразметка готова</span> : null}
                  {uploadResult.preannotation?.status === "processing" ? (
                    <span className="status-chip status-chip-warn">В процессе</span>
                  ) : null}
                </div>

                <div className="result-meta">
                  <div className="task-badge">
                    <span>CVAT task</span>
                    {uploadResult.task_id}
                  </div>
                  {uploadResult.media_type === "video" ? (
                    <div className="stat-pill">
                      Video
                      {uploadResult.video?.cvat_frame_count != null
                        ? ` · ${uploadResult.video.cvat_frame_count} кадров`
                        : ""}
                      {uploadResult.video?.frame_step != null ? ` · step ${uploadResult.video.frame_step}` : ""}
                    </div>
                  ) : (
                    <div className="stat-pill">Изображения</div>
                  )}
                </div>

                {uploadResult.warnings?.length ? (
                  <div className="alert alert-warn">{uploadResult.warnings.join(" ")}</div>
                ) : null}

                {uploadResult.preannotation ? (
                  <>
                    {uploadResult.preannotation.status === "processing" ? (
                      <div className="alert alert-warn">
                        <span className="alert-inline">
                          <span className="spinner spinner-inline" aria-hidden />
                          Предразметка выполняется
                          {preannPhase === "extracting_frames"
                            ? " — извлечение кадров"
                            : preannPhase === "calling_ml"
                              ? " — ML inference"
                              : preannPhase === "importing_to_cvat"
                                ? " — импорт в CVAT"
                                : ""}
                          . Статус обновляется автоматически.
                        </span>
                      </div>
                    ) : null}
                    {uploadResult.preannotation.status === "error" ? (
                      <div className="alert alert-error">
                        Предразметка завершилась с ошибкой.
                        {uploadResult.preannotation.error ? ` ${uploadResult.preannotation.error}` : ""}
                      </div>
                    ) : null}
                    {preannDone ? (
                      <div className="result-stats">
                        <div className="result-stat">
                          <span className="result-stat-label">Аннотации в CVAT</span>
                          <span className="result-stat-value">Импортированы</span>
                        </div>
                        {uploadResult.preannotation.archive_size != null ? (
                          <div className="result-stat">
                            <span className="result-stat-label">Архив</span>
                            <span className="result-stat-value">
                              {(uploadResult.preannotation.archive_size / 1024).toFixed(1)} KB
                            </span>
                          </div>
                        ) : null}
                        {preannStats?.annotations_count != null ? (
                          <div className="result-stat">
                            <span className="result-stat-label">Объектов</span>
                            <span className="result-stat-value">{preannStats.annotations_count}</span>
                          </div>
                        ) : null}
                        {preannStats?.images_count != null ? (
                          <div className="result-stat">
                            <span className="result-stat-label">Кадров</span>
                            <span className="result-stat-value">{preannStats.images_count}</span>
                          </div>
                        ) : null}
                        {preannStats?.categories_count != null ? (
                          <div className="result-stat">
                            <span className="result-stat-label">Классов</span>
                            <span className="result-stat-value">{preannStats.categories_count}</span>
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                    {previewSrc ? (
                      <div className="preview-card">
                        <p className="preview-card-label">Превью предразметки</p>
                        <button
                          type="button"
                          className="preview-thumb"
                          onClick={() => setPreviewOpen(true)}
                          aria-label="Открыть превью предразметки"
                        >
                          <img src={previewSrc} alt={previewFilename ?? "Превью предразметки"} />
                          <span className="preview-thumb-overlay">Нажмите, чтобы открыть</span>
                        </button>
                        <div className="preview-card-actions">
                          <button type="button" className="btn-secondary btn-compact" onClick={() => setPreviewOpen(true)}>
                            Открыть
                          </button>
                          {previewFilename ? (
                            <span className="preview-filename">{previewFilename}</span>
                          ) : null}
                        </div>
                      </div>
                    ) : null}
                  </>
                ) : (
                  <div className="alert alert-success">Файлы загружены без предразметки.</div>
                )}
              </section>
            ) : null}

            <div className="divider" />
            <h2 className="section-title">Экспорт аннотаций</h2>
            {taskId == null ? (
              <p style={{ color: "var(--text-muted)", fontSize: 14 }}>Сначала создайте задачу выше.</p>
            ) : (
              <>
                <div className="row-2">
                  <div className="field">
                    <label htmlFor="fmt">Формат</label>
                    <select
                      id="fmt"
                      className="select"
                      value={exportFormat}
                      onChange={(e) => setExportFormat(e.target.value)}
                    >
                      {EXPORT_FORMATS.map((f) => (
                        <option key={f} value={f}>
                          {f}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="field" style={{ display: "flex", alignItems: "flex-end" }}>
                    <Toggle checked={includeImages} onChange={setIncludeImages} label="Включить изображения" />
                  </div>
                </div>
                {exportError ? <div className="alert alert-error">{exportError}</div> : null}
                <div className="export-actions">
                  <button type="button" className="btn-secondary" disabled={exporting || preannotationInProgress} onClick={handleExport}>
                    {exporting ? "Экспорт…" : "Скачать ZIP"}
                  </button>
                </div>
              </>
            )}

            <div className="divider" />
            <h2 className="section-title">Валидация в CVAT</h2>
            {taskId == null ? (
              <p style={{ color: "var(--text-muted)", fontSize: 14 }}>Нужен task id после загрузки.</p>
            ) : (
              <form onSubmit={handleGrant}>
                <div className="field">
                  <label htmlFor="rev">Валидатор (username, email или id)</label>
                  <input
                    id="rev"
                    className="input"
                    value={reviewerUser}
                    onChange={(e) => setReviewerUser(e.target.value)}
                    placeholder="username"
                    disabled={preannotationInProgress}
                  />
                </div>
                <button type="submit" className="btn-secondary" disabled={grantLoading || preannotationInProgress}>
                  {grantLoading ? "Назначение…" : "Назначить валидатора"}
                </button>
                {grantMessage ? (
                  <div
                    className={`alert ${grantMessage.type === "ok" ? "alert-success" : "alert-error"}`}
                    style={{ marginTop: 12 }}
                  >
                    {grantMessage.text}
                  </div>
                ) : null}
              </form>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
