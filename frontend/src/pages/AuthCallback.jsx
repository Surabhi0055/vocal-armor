import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuthStore } from "../store/authStore";
import Cookies from "js-cookie";

export default function AuthCallback() {
  const navigate = useNavigate();
  const { setTokens } = useAuthStore();

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const access = params.get("access_token") || params.get("token");
    const refresh = params.get("refresh_token") || params.get("refresh");

    if (access && refresh) {
      setTokens(access, refresh);

      // Fetch user info
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      fetch(`${apiUrl}/auth/me`, {
        headers: { Authorization: `Bearer ${access}` },
      })
        .then((r) => r.json())
        .then((user) => {
          useAuthStore.setState({ user, accessToken: access });
          navigate("/", { replace: true });
        })
        .catch(() => navigate("/login", { replace: true }));
    } else {
      navigate("/login", { replace: true });
    }
  }, []);

  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#050e10",
        color: "#00d4c8",
        fontFamily: "'Space Grotesk', sans-serif",
      }}
    >
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: "24px", marginBottom: "12px" }}>⏳</div>
        <p>Completing sign in...</p>
      </div>
    </div>
  );
}
