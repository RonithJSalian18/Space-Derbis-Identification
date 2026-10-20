# SpaceGuard Vision: Web Frontend

Next.js 16 (App Router, Tailwind CSS v4) interface for the Space Debris Identification models.

## Run

The frontend talks to the FastAPI inference server in [`../api/server.py`](../api/server.py). Start the API from the repository root first:

```bash
python -m uvicorn api.server:app --port 8000
```

Then start the frontend:

```bash
cd frontend
cp .env.example .env.local   # optional: change NEXT_PUBLIC_API_URL
npm install
npm run dev                  # http://localhost:3000
```

## Pages

| Route       | Purpose                                                                                          |
| ----------- | ------------------------------------------------------------------------------------------------ |
| `/`         | Overview of the pipeline, the four architectures and the 3-state decision policy                 |
| `/classify` | Pick a model, upload or select a sample image, view verdict, probabilities, decision zone, Grad-CAM, and a cross-model consensus |
| `/models`   | Benchmark table and charts (error rates, latency vs. size, checkpoint footprint)                  |

## Structure

```
src/
├── app/                 # routes: page.tsx, classify/, models/
├── components/
│   ├── classify/        # model selector, dropzone, samples, result panel, Grad-CAM viewer, consensus
│   ├── models/          # benchmark table and Recharts charts
│   └── ...              # navbar, footer, API status, UI primitives
└── utils/               # typed API client, model metadata, types
```
