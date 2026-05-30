import axios from 'axios';
import { useAuthStore } from '../store/authStore';

const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
    withCredentials: true,
});

// Attach token to every request
api.interceptors.request.use((config) => {
    const token = useAuthStore.getState().accessToken;
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
});

// Auto refresh on 401
api.interceptors.response.use(
    (res) => res,
    async (error) => {
        const original = error.config;
        if (error.response?.status === 401 && !original._retry) {
            original._retry = true;
            const refreshed = await useAuthStore.getState().refreshAccessToken();
            if (refreshed) {
                const token = useAuthStore.getState().accessToken;
                original.headers.Authorization = `Bearer ${token}`;
                return api(original);
            }
        }
        return Promise.reject(error);
    }
);

export default api;