import { Link, useSearchParams } from "react-router-dom";
import { PageHeader } from "../components/layout/PageHeader";
import { StatusChip } from "../components/StatusChip";
import { useDriftVideosContext } from "../context/DriftVideosContext";

const MOCK_ALERTS = [
  { time: "12:08", text: "PSI > 0.15 для camera_07", level: "warn" as const },
  { time: "11:41", text: "mIoU упал до 0.81", level: "warn" as const },
  { time: "11:12", text: "Новые редкие классы в потоке", level: "info" as const },
  { time: "09:30", text: "Обработка завершена успешно", level: "ok" as const },
];

const RISK_SEGMENTS = [
  { label: "night_shift_frames", value: 32 },
  { label: "small_objects", value: 24 },
  { label: "rain_fog_weather", value: 19 },
  { label: "camera_07_angle", value: 14 },
  { label: "other", value: 11 },
];

const KPI_ITEMS = [
  { label: "Drift score", value: "0.27", chip: { label: "WARN", variant: "warn" as const } },
  { label: "PSI", value: "0.18" },
  { label: "KL divergence", value: "0.12" },
  { label: "Model mIoU", value: "0.81" },
];

export function DriftMonitorPage() {
  const [params] = useSearchParams();
  const videoFilter = params.get("video");
  const { videos, getVideo } = useDriftVideosContext();
  const selected = videoFilter ? getVideo(videoFilter) : null;

  const activeVideos = videos.filter((v) => v.status === "active" || v.status === "drift_detected");
  const driftScore = (selected?.driftScore ?? (
    activeVideos.length > 0
      ? activeVideos.reduce((s, v) => s + (v.driftScore ?? 0), 0) / activeVideos.length
      : 0.18
  )).toFixed(2);

  const kpi = KPI_ITEMS.map((item, idx) =>
    idx === 0 ? { ...item, value: driftScore } : item,
  );

  return (
    <div className="page-shell">
      <PageHeader
        title="Панель мониторинга"
        subtitle={
          selected
            ? `Метрики для «${selected.name}»`
            : "Сводные метрики дрейфа по всем видео-источникам"
        }
        actions={
          selected ? (
            <Link to={`/drift/${selected.id}`} className="btn-secondary btn-compact">
              Настройки видео
            </Link>
          ) : null
        }
      />

      <div className="kpi-grid">
        {kpi.map((item) => (
          <article key={item.label} className="kpi-card panel">
            <span className="kpi-label">{item.label}</span>
            <div className="kpi-row">
              <span className="kpi-value">{item.value}</span>
              {item.chip ? <StatusChip label={item.chip.label} variant={item.chip.variant} /> : null}
            </div>
          </article>
        ))}
      </div>

      <div className="layout-grid layout-grid-monitor">
        <section className="panel">
          <div className="panel-header">Динамика дрейфа</div>
          <div className="panel-body chart-panel">
            <div className="chart-placeholder">
              <svg viewBox="0 0 800 240" className="chart-svg" aria-hidden>
                <defs>
                  <linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#14b8a6" stopOpacity="0.25" />
                    <stop offset="100%" stopColor="#14b8a6" stopOpacity="0" />
                  </linearGradient>
                </defs>
                {[40, 90, 140, 190].map((y) => (
                  <line key={y} x1="48" y1={y} x2="760" y2={y} stroke="#e2e8f0" />
                ))}
                <line x1="48" y1="200" x2="760" y2="200" stroke="#cbd5e1" />
                <line x1="48" y1="88" x2="760" y2="88" stroke="#f59e0b" strokeDasharray="8 6" />
                <polygon
                  fill="url(#chartFill)"
                  points="60,170 140,158 220,150 300,142 380,128 460,112 540,98 620,84 700,72 740,68 740,200 60,200"
                />
                <polyline
                  fill="none"
                  stroke="#0d9488"
                  strokeWidth="3"
                  points="60,170 140,158 220,150 300,142 380,128 460,112 540,98 620,84 700,72 740,68"
                />
                <circle cx="740" cy="68" r="5" fill="#0d9488" />
              </svg>
            </div>
            <p className="field-hint chart-caption">Подключение Grafana или API метрик — на следующем этапе.</p>
          </div>
        </section>

        <aside className="panel">
          <div className="panel-header">Сегменты риска</div>
          <div className="panel-body">
            <ul className="risk-list">
              {RISK_SEGMENTS.map((item) => (
                <li key={item.label} className="risk-item">
                  <div className="risk-item-head">
                    <span>{item.label}</span>
                    <strong>{item.value}%</strong>
                  </div>
                  <div className="risk-bar">
                    <span style={{ width: `${item.value}%` }} />
                  </div>
                </li>
              ))}
            </ul>
            <button type="button" className="btn-primary btn-fit">
              Запустить дообучение
            </button>
          </div>
        </aside>
      </div>

      <section className="panel">
        <div className="panel-header">Лента алертов</div>
        <div className="panel-body alert-feed">
          {MOCK_ALERTS.map((a) => (
            <div key={`${a.time}-${a.text}`} className={`alert-feed-item alert-feed-${a.level}`}>
              <span className="alert-feed-time">{a.time}</span>
              <span className="alert-feed-text">{a.text}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">Grafana dashboards</div>
        <div className="panel-body">
          <div className="grafana-frame">
            <span>Область для встраивания дашбордов Grafana</span>
          </div>
        </div>
      </section>
    </div>
  );
}
