import { Navigate } from "react-router-dom";

type Props = {
  children: React.ReactNode;
  user: any; 
  requireAdmin?: boolean;
};

export default function ProtectedRoute({ children, user, requireAdmin }: Props) {
  if (!user) return <Navigate to="/login" replace />;

  if (requireAdmin && user.role !== "admin") {
    return <Navigate to="/analyze" replace />;
  }

  return <>{children}</>;
}