// import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
// import LoginPage from "@/pages/LoginPage";
// import AnalyzePage from "@/pages/AnalyzePage";

// export default function App() {
//   return (
//     <BrowserRouter>
//       <Routes>
//         <Route path="/login" element={<LoginPage />} />
//         <Route path="/analyze" element={<AnalyzePage />} />
//         <Route path="*" element={<Navigate to="/analyze" replace />} />
//       </Routes>
//     </BrowserRouter>
//   );
// }

import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import AnalyzePage from "@/pages/AnalyzePage";
import LoginPage from "@/pages/LoginPage";
import AdminPage from "@/pages/AdminPage";
import HistoryPage from "@/pages/HistoryPage";
import ChatPage from "@/pages/ChatPage";
import RegisterPage from "@/pages/RegisterPage";
import ForgotPasswordPage from "@/pages/ForgotPasswordPage";
import ResetPasswordPage from "@/pages/ResetPasswordPage";
import GoogleCallbackPage from "@/pages/GoogleCallbackPage";
import ProtectedRoute from "@/components/ui/auth/ProtectedRoute";
import LoadingScreen from "@/components/ui/LoadingScreen";
import { me } from "@/lib/auth";

export default function App() {
  const [user, setUser] = useState<any>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    (async () => {
      const token = localStorage.getItem("access_token");
      if (!token) {
        setUser(null);
        setReady(true);
        return;
      }
      try {
        const u = await me();
        setUser(u);
      } catch {
        setUser(null);
      } finally {
        setReady(true);
      }
    })();
  }, []);

  if (!ready) return <LoadingScreen />;

  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/login"
          element={
            user ? (
              <Navigate to={user.role === "admin" ? "/admin" : "/analyze"} replace />
            ) : (
              <LoginPage onLogin={(u: any) => setUser(u)} />
            )
          }
        />

        <Route
          path="/forgot-password"
          element={
            user ? (
              <Navigate to="/analyze" replace />
            ) : (
              <ForgotPasswordPage />
            )
          }
        />

        <Route
          path="/auth/google/callback"
          element={<GoogleCallbackPage />}
        />

        <Route
          path="/reset-password"
          element={
            user ? (
              <Navigate to="/analyze" replace />
            ) : (
              <ResetPasswordPage />
            )
          }
        />

        <Route
          path="/register"
          element={
            user && user.role !== "admin" ? (
              <Navigate to="/analyze" replace />
            ) : (
              <RegisterPage />
            )
          }
        />

        <Route
          path="/analyze"
          element={
            <ProtectedRoute user={user}>
              <AnalyzePage user={user} onLogout={() => setUser(null)} />
            </ProtectedRoute>
          }
        />

        <Route
          path="/history"
          element={
            <ProtectedRoute user={user}>
              <HistoryPage user={user} onLogout={() => setUser(null)} />
            </ProtectedRoute>
          }
        />

        <Route
          path="/chat"
          element={
            <ProtectedRoute user={user}>
              <ChatPage user={user} onLogout={() => setUser(null)} />
            </ProtectedRoute>
          }
        />

        <Route
          path="/admin"
          element={
            <ProtectedRoute user={user} requireAdmin>
              <AdminPage user={user} onLogout={() => setUser(null)} />
            </ProtectedRoute>
          }
        />

        <Route path="*" element={<Navigate to="/analyze" replace />} />
      </Routes>
    </BrowserRouter>
  );
}