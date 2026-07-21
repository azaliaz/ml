import { useEffect } from "react";

interface PreviewLightboxProps {
  src: string;
  alt: string;
  filename?: string | null;
  onClose: () => void;
}

export function PreviewLightbox({ src, alt, filename, onClose }: PreviewLightboxProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = src;
    a.download = filename ?? "preannotation_preview.png";
    a.click();
  };

  return (
    <div
      className="preview-lightbox"
      role="dialog"
      aria-modal="true"
      aria-label="Превью предразметки"
      onClick={onClose}
    >
      <div className="preview-lightbox-panel" onClick={(e) => e.stopPropagation()}>
        <div className="preview-lightbox-toolbar">
          <span className="preview-lightbox-title">{filename ?? "Превью предразметки"}</span>
          <div className="preview-lightbox-actions">
            <button type="button" className="btn-ghost" onClick={handleDownload}>
              Скачать PNG
            </button>
            <button type="button" className="btn-ghost btn-ghost-strong" onClick={onClose}>
              Закрыть
            </button>
          </div>
        </div>
        <div className="preview-lightbox-body">
          <img src={src} alt={alt} />
        </div>
      </div>
    </div>
  );
}
