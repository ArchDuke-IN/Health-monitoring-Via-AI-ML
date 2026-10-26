# HealthScan

HealthScan is a local, real-time health-screening prototype that uses a webcam
or phone camera stream to estimate heart rate and provide visual indicators for
fatigue, stress, jaundice, and anemia. It is designed for experimentation and
education, not for clinical use.

## What it does

- Guides the user into a consistent face position before measurement.
- Calibrates skin-tone and facial baselines for the current session.
- Estimates heart rate from facial color changes using a remote photoplethysmography
  (rPPG) signal.
- Tracks eye openness and blink rate as fatigue and stress indicators.
- Samples the sclera and lips for color-based jaundice and anemia indicators.
- Displays the measurements in a live OpenCV window.

All processing is performed locally. The project does not include an account
system, telemetry, or a hosted service.

## Important safety notice

HealthScan is an experimental screening aid and must not be used to diagnose,
treat, or rule out any medical condition. Results can be affected by lighting,
camera quality, movement, skin tone, facial position, and many other factors.
Do not make health decisions based on this software. Consult a qualified
healthcare professional for medical advice or urgent symptoms.

## Requirements

- Python 3.10 or newer
- A webcam, or an MJPEG-compatible phone camera stream
- A desktop environment capable of opening an OpenCV window

The dependencies are listed in [`requirements.txt`](requirements.txt).

## Installation

```powershell
git clone https://github.com/ArchDuke-IN/Health-monitoring-Via-AI-ML.git
cd Health-monitoring-Via-AI-ML

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Running

```bash
python main.py
```

The application opens a window named `HealthScan`. Press `Q` to quit or `R` to
reset the current calibration and measurements.

By default, Linux uses camera device `0`; other platforms use the phone stream
URL currently configured in [`main.py`](main.py). Update `CAMERA_SOURCE` there
to use a different local webcam or stream before starting the application.

## Project layout

```text
main.py                    Application loop and camera integration
modules/calibration.py     Session baseline calibration
modules/face_detect.py     Standalone face-mesh demo
modules/face_guide.py      Face positioning guide
modules/heart_rate.py      rPPG heart-rate estimation
modules/jaundice_anemia.py Color-based screening indicators
modules/stress_fatigue.py  Eye and blink-based indicators
```

## Development

Changes should remain focused, readable, and testable. Before opening a pull
request:

1. Create a topic branch from the default branch.
2. Make the smallest change that addresses the issue.
3. Run the application or relevant checks locally.
4. Describe the behavior change, validation performed, and any limitations.
5. Include screenshots or short recordings for user-interface changes.

Because camera hardware and lighting vary, contributors should document their
camera, operating system, and lighting conditions when reporting measurement
behavior.

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md), [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md),
and [`SECURITY.md`](SECURITY.md) before contributing.

## License

HealthScan is released under the [MIT License](LICENSE).
