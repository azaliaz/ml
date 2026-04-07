export function HeroIllustration() {
  return (
    <svg
      className="hero-illustration"
      width={200}
      height={72}
      viewBox="0 0 200 72"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      {/* Фото */}
      <rect x={8} y={38} width={28} height={22} rx={3} stroke="currentColor" strokeWidth={1.5} opacity={0.55} />
      <path d="M14 54l6-6 5 5 4-4 7 8H14z" fill="currentColor" opacity={0.12} />
      {/* Стрелка 1 */}
      <path d="M42 49h18" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" opacity={0.45} />
      <path d="M56 45l6 4-6 4" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" opacity={0.45} />
      {/* ML */}
      <rect x={68} y={36} width={36} height={26} rx={4} stroke="currentColor" strokeWidth={1.5} />
      <text x={86} y={53} textAnchor="middle" fill="currentColor" fontSize={10} fontFamily="system-ui,sans-serif" opacity={0.5}>
        ML
      </text>
      {/* Стрелка 2 */}
      <path d="M108 49h18" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" opacity={0.45} />
      <path d="M122 45l6 4-6 4" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" opacity={0.45} />
      {/* CVAT */}
      <rect x={132} y={36} width={60} height={26} rx={4} stroke="currentColor" strokeWidth={1.5} opacity={0.75} />
      <text x={162} y={53} textAnchor="middle" fill="currentColor" fontSize={9} fontFamily="system-ui,sans-serif" opacity={0.55}>
        CVAT
      </text>
    </svg>
  );
}
