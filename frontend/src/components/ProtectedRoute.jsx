import { Navigate } from "react-router-dom";
import { useAuthStore } from "../store/authStore";

const ProtectedRoute = ({ children }) => {
  const { user, accessToken } = useAuthStore();
  if (!user || !accessToken) {
    return <Navigate to="/start" replace />;
  }
  return children;
};

export default ProtectedRoute;
