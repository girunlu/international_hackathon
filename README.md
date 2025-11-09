# Smart Sustainable Shopping Assistant (S3A)

## Overview
S3A helps households cut food waste and save money by tracking pantry activity, forecasting demand, and nudging smarter purchasing decisions. The platform combines lightweight AI services with a Vite/React front end and a FastAPI backend that emulates the cloud APIs locally.

## Key Features
- Tracks shopping history, waste logs, and saved items to surface actionable insights.
- Predicts future demand using time-series models and highlights likely overstocked products.
- Classifies produce images to speed up data entry while shopping.
- Presents dashboards, recommendations, and personalized alerts in a responsive web app.

## Quick Start

### Backend (FastAPI)
1. `cd backend`
2. Create and activate a virtual environment (optional but recommended).
3. Install the dependencies your environment needs, for example: `pip install fastapi uvicorn pandas numpy torch torchvision pillow`
4. Start the local API: `uvicorn main:app --reload --port 8000`

The backend loads CSV/JSON data from `backend/database` and `frontend/static_files`, exposing them through REST endpoints.

### Frontend (Vite + React)
1. `cd frontend`
2. Install dependencies: `npm install`
3. Run the dev server: `npm run dev`

The Vite dev server proxies requests to the FastAPI instance running on port 8000. Update `.env` settings if you customize ports.

## Project Structure
```
.
├── ai/                      # Training notebooks, datasets, and saved PyTorch checkpoints
├── backend/                 # FastAPI application serving data and AI-powered endpoints
│   └── database/            # CSV snapshots persisted on disk
├── frontend/                # Vite/React client with Tailwind styling and Supabase hooks
│   ├── src/                 # UI components, pages, hooks, and backend API helpers
│   └── static_files/        # JSON fixtures used for analytics and recommendations
└── README.md                # Project overview and local development guide
```

## Project Background
- **Inspiration:** Household food waste averages 30–40% of weekly purchases, costing families money and straining global food systems. S3A exists to make sustainable, wallet-friendly shopping automatic.
- **What it does:** Learns consumption habits, predicts future demand, warns of likely waste, and quantifies savings to keep users motivated—much like a personal finance tracker for the kitchen.
- **How it works:** A small agentic layer orchestrates demand forecasting, image classification, and reasoning models, all surfaced through a custom-built web experience.
- **Challenges:** Balancing multi-model orchestration with simplicity, building a full web stack from scratch, and adapting to shifting compute limits for AI workloads.
- **Accomplishments:** Delivered a working prototype that unifies real-time recognition, predictive analytics, and conversational guidance in one cohesive interface.
- **Lessons learned:** Everyday habits scale; consistent reductions per household compound into major economic and ecological gains. Adaptive AI and user-centered simplicity power real adoption.
- **What’s next:** Expand region-aware recommendations, partner with local retailers, and extend forecasting to new contexts (fast food, offices) using open datasets and richer AI pipelines.


