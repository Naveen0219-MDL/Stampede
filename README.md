# Stampede

Stampede is a prototype crowd monitoring and evacuation-assistance system. It processes a camera feed into a 3 × 4 density heatmap, highlights a lower-density route, and can send LED states to an Arduino-compatible microcontroller over serial. A separate Flask dashboard displays the stream.

## Components

- `py/main_system.py` — camera processing, density and motion metrics, video stream API, and LED control used by the dashboard.
- `py/main_system_server.py` — an alternate server with direct video and LED controls.
- `crowd_dashboard/dashboard_app.py` — web dashboard and demo login.
- `stm/stm.ino` — firmware for the 12-LED indicator grid.
- Root-level Python files — experiments and alternate processing prototypes.

## Setup

Use Python 3.10 or newer. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set a unique `FLASK_SECRET_KEY`, dashboard username/password, camera index, and serial port. The `.env` file is ignored by Git. Connect the camera and flash `stm/stm.ino` to the board before enabling hardware output.

Run the processing server and dashboard in separate terminals:

```bash
python py/main_system.py
python crowd_dashboard/dashboard_app.py
```

Open `http://127.0.0.1:7000`. The serial device and camera must be connected and accessible. The processing server provides the `/video` and `/metrics` endpoints used by the dashboard. See `.env.example` for configuration names.

## Data and model files

Dataset images, trained model weights, screenshots, and local demo artifacts are excluded from Git to keep the repository small and avoid publishing files that are not needed to run the prototype. Add data or model download instructions here if they are required for a reproducible experiment.

## Limitations

This is an academic prototype, not a certified crowd-safety or evacuation system. Image texture-based density and the sample route logic should not be used for real emergency decisions without substantial validation.
