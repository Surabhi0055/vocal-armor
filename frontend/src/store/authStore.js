import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import Cookies from 'js-cookie';
import axios from 'axios';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const useAuthStore = create(
    persist(
        (set, get) => ({
            user: null,
            accessToken: null,
            refreshToken: null,
            isLoading: false,
            error: null,

            setTokens: (access, refresh) => {
                set({ accessToken: access, refreshToken: refresh });
                Cookies.set('va_refresh', refresh, { expires: 7, secure: false });
            },

            login: async (email, password, rememberMe) => {
                set({ isLoading: true, error: null });
                try {
                    const res = await axios.post(`${API}/auth/login`, {
                        email, password, remember_me: rememberMe
                    });
                    const { access_token, refresh_token, user } = res.data;
                    set({ user, accessToken: access_token, isLoading: false });
                    get().setTokens(access_token, refresh_token);
                    return { success: true };
                } catch (err) {
                    const msg = err.response?.data?.detail || 'Login failed';
                    set({ error: msg, isLoading: false });
                    return { success: false, error: msg };
                }
            },

            register: async (email, username, password, fullName) => {
                set({ isLoading: true, error: null });
                try {
                    const res = await axios.post(`${API}/auth/register`, {
                        email, username, password, full_name: fullName
                    });
                    const { access_token, refresh_token, user } = res.data;
                    set({ user, accessToken: access_token, isLoading: false });
                    get().setTokens(access_token, refresh_token);
                    return { success: true };
                } catch (err) {
                    const msg = err.response?.data?.detail || 'Registration failed';
                    set({ error: msg, isLoading: false });
                    return { success: false, error: msg };
                }
            },

            logout: async () => {
                const refresh = Cookies.get('va_refresh');
                if (refresh) {
                    try {
                        await axios.post(`${API}/auth/logout`, { refresh_token: refresh });
                    } catch { }
                }
                Cookies.remove('va_refresh');
                set({ user: null, accessToken: null, refreshToken: null });
                // Notify all components that history has changed (now empty for next user)
                window.dispatchEvent(new Event('va_history_updated'));
            },

            refreshAccessToken: async () => {
                const refresh = Cookies.get('va_refresh');
                if (!refresh) return false;
                try {
                    const res = await axios.post(`${API}/auth/refresh`, {
                        refresh_token: refresh
                    });
                    const { access_token, refresh_token } = res.data;
                    set({ accessToken: access_token });
                    get().setTokens(access_token, refresh_token);
                    return true;
                } catch {
                    get().logout();
                    return false;
                }
            },

            clearError: () => set({ error: null }),
            updateUser: (userData) => set({ user: { ...get().user, ...userData } }),
        }),
        {
            name: 'va_auth',
            partialize: (state) => ({
                user: state.user,
                accessToken: state.accessToken,
            }),
        }
    )
);