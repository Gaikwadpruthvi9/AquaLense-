# AQUORA

> **AI-Powered Automated Underwater Marine Debris & Sonar Anomaly Detection System**  
> Developed for Side-Scan Sonar Imagery intelligence, bathymetric mapping, and hazard detection.

---

## 🛠️ Architecture

- **Backend**: FastAPI (Python 3.11+), SQLite / SQLAlchemy, NumPy, Pillow, Uvicorn
- **Frontend**: Vite, React 18, TypeScript, Tailwind CSS, Leaflet / React-Leaflet, Recharts, Lucide Icons

---

## 🚀 Quick Start

### Option 1: One-Click Launch (Windows)

Double-click or run:
```cmd
start-all.bat
```
This will automatically set up virtual environments, install dependencies, and launch both servers in separate windows.

---

### Option 2: Manual Setup (Terminal)

#### 1. Backend Server (FastAPI)
Open a terminal in the project root:
```powershell
cd project\backend

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start backend server
python main.py
```
- Backend API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive API Docs (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

> **Note**: On startup, the backend automatically initializes the SQLite database (`oceanscan.db`) and seeds synthetic sonar imagery and survey detections.

#### 2. Frontend Application (React + Vite)
Open a second terminal:
```powershell
cd project\frontend

# Install dependencies
npm install

# Start Vite dev server
npm run dev
```
- Frontend Web App: [http://localhost:5173](http://localhost:5173)

---

## 🔑 Demo Login

The authentication system supports seamless demo login:
- **Email**: `oceanographer@oceanscan.marine.gov` (or any email)
- **Password**: Any password (e.g. `password123`)
