import { NavLink, Outlet } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/preannotation", label: "Предразметка", hint: "CVAT + ML" },
  { to: "/drift", label: "Источники видео", hint: "Дрейф данных" },
  { to: "/drift/monitor", label: "Мониторинг", hint: "Метрики и алерты" },
] as const;

export function AppShell() {
  return (
    <div className="platform-shell">
      <aside className="platform-sidebar">
        <div className="platform-brand">
          <span className="platform-brand-title">Автодообучение CV-моделей</span>
          <span className="platform-brand-sub">Платформа компьютерного зрения</span>
        </div>
        <nav className="platform-nav" aria-label="Основная навигация">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to !== "/preannotation"}
              className={({ isActive }) =>
                `platform-nav-link${isActive ? " platform-nav-link-active" : ""}`
              }
            >
              <span className="platform-nav-label">{item.label}</span>
              <span className="platform-nav-hint">{item.hint}</span>
            </NavLink>
          ))}
        </nav>
      </aside>
      <div className="platform-main">
        <Outlet />
      </div>
    </div>
  );
}
