# VocalArmor — Frontend Client

A premium, dark-mode React + Vite dashboard for the VocalArmor deepfake voice detection engine.

## Stack

- **React 19** + **Vite 8** — Component-driven SPA with hot module replacement
- **React Router v7** — SPA routing and protected route management
- **Zustand** — Lightweight global auth and user state
- **Recharts** — Interactive confidence histogram and fake rate trend charts
- **Axios** — API communication with the FastAPI backend
- **Tabler Icons** — Icon system via CDN

## Pages

| Route | Page | Description |
|-------|------|-------------|
| `/start` | Landing Page | Public marketing page |
| `/login` | Auth Page | Login / Register / Google OAuth |
| `/` | Detector | Single-file deepfake upload and analysis |
| `/live` | Live Monitor | Real-time microphone stream via WebSocket |
| `/batch` | Batch Upload | Multi-file scanning with progress and results |
| `/history` | History | Scan history, charts, table, and CSV export |
| `/user` | Settings | User profile and detection preferences |

## Running Locally

```bash
npm install
npm run dev
```

Frontend will be available at `http://localhost:5173`.

## Building for Production

```bash
npm run build
```

Output goes to `dist/`.

## Environment

The frontend expects the FastAPI backend at `http://localhost:8000`.
Update `vite.config.js` to proxy API calls if needed.
