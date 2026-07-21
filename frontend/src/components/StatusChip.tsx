type ChipVariant = "ok" | "warn" | "danger" | "neutral";

const VARIANT_CLASS: Record<ChipVariant, string> = {
  ok: "status-chip-ok",
  warn: "status-chip-warn",
  danger: "status-chip-danger",
  neutral: "status-chip-neutral",
};

export function StatusChip({ label, variant = "neutral" }: { label: string; variant?: ChipVariant }) {
  return <span className={`status-chip ${VARIANT_CLASS[variant]}`}>{label}</span>;
}

export function driftStatusVariant(status: string): ChipVariant {
  switch (status) {
    case "active":
      return "ok";
    case "drift_detected":
      return "danger";
    case "processing":
      return "warn";
    default:
      return "neutral";
  }
}

export function driftStatusLabel(status: string): string {
  switch (status) {
    case "active":
      return "Активно";
    case "processing":
      return "Обработка";
    case "drift_detected":
      return "Дрейф";
    case "paused":
      return "Пауза";
    default:
      return "Ожидание";
  }
}
