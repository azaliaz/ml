export function UploadEmptyIllustration() {
  return (
    <svg
      className="upload-empty-illustration"
      width={200}
      height={120}
      viewBox="0 0 200 120"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      {/* «Облако» — намёк на загрузку */}
      <path
        d="M148 52h-8a18 18 0 10-35-6 16 16 0 10-30 10h-9a14 14 0 000 28h74a16 16 0 000-32z"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinejoin="round"
        opacity={0.35}
      />
      {/* Папка */}
      <path
        d="M38 78h88a6 6 0 006-6V48a6 6 0 00-6-6H84l-6-8H44a6 6 0 00-6 6v36a6 6 0 006 6z"
        stroke="currentColor"
        strokeWidth={1.75}
        strokeLinejoin="round"
      />
      {/* Мини-«кадры» */}
      <rect x={52} y={56} width={22} height={16} rx={2} stroke="currentColor" strokeWidth={1.25} opacity={0.5} />
      <rect x={78} y={52} width={26} height={20} rx={2} stroke="currentColor" strokeWidth={1.25} />
      {/* Стрелка вверх к облаку */}
      <path
        d="M100 34v14M94 40l6-8 6 8"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.7}
      />
    </svg>
  );
}
