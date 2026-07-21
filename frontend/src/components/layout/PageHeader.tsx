import type { ReactNode } from "react";
import { Link } from "react-router-dom";

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  backTo?: string;
  backLabel?: string;
  actions?: ReactNode;
}

export function PageHeader({ title, subtitle, backTo, backLabel, actions }: PageHeaderProps) {
  return (
    <header className="page-header">
      <div className="page-header-main">
        {backTo ? (
          <Link to={backTo} className="back-link">
            ← {backLabel ?? "Назад"}
          </Link>
        ) : null}
        <h1 className="app-title">{title}</h1>
        {subtitle ? <p className="app-sub">{subtitle}</p> : null}
      </div>
      {actions ? <div className="page-header-actions">{actions}</div> : null}
    </header>
  );
}
