# Eyetrack

**English** · [Français](README.fr.md) · [Website](https://dimaland1.github.io/Eyetrack/) · [Changelog](CHANGELOG.md)

Free head and eye tracking with an ordinary webcam, for driving and flight simulators.

Eyetrack follows your head (6 degrees of freedom) and where you look on the screen. It sends
both to [OpenTrack](https://github.com/opentrack/opentrack) ("UDP over network", port 4242), so
the in-game camera moves with you. You turn your head a little, the view turns a lot. You glance
at a mirror, the view nudges toward it. It works with glasses.

- **Download:** [Eyetrack-Setup.exe](https://github.com/dimaland1/Eyetrack/releases/latest)
  (Windows 64-bit, no administrator rights needed)
- **License:** [GPL-3.0](LICENSE), free software

## Features

- **6DoF head tracking** from 478 face landmarks (MediaPipe), with adjustable amplification:
  20° of head rotation can become 50° in game.
- **Gaze on top of head tracking.** Your gaze adds a calibrated nudge to the camera. A smoothing
  filter and a dead zone keep the view calm while you read the dashboard.
- **Works with glasses.** Eyetrack warns you when a reflection hides your eyes, and tells you
  what to change.
- **9-point calibration** with an honest score: each target is predicted without its own data.
- **Per-game profiles.** Eyetrack switches profile automatically when you start Euro Truck
  Simulator 2, American Truck Simulator, BeamNG.drive, Assetto Corsa, iRacing, MSFS, X-Plane
  or DCS, and switches back when you quit.
- **Smooth output:** the webcam gives 30 images per second, and Eyetrack sends 120 updates per
  second to OpenTrack, with light prediction in between.
- **Stays out of your way:** global hotkeys that work in game, a tray icon, and a control panel
  in English or French.
- **Private:** all processing happens on your PC. No account, no telemetry, no image ever leaves
  your computer.

## Install

1. **Download and run** `Eyetrack-Setup.exe` from the
   [latest release](https://github.com/dimaland1/Eyetrack/releases/latest). The installer is not
   signed yet, so Windows SmartScreen may show "Windows protected your PC". Click **More info**,
   then **Run anyway**.
2. **The installer offers two optional extras:**
   - **OpenTrack**, if it is not installed yet. It is installed with winget, and Windows asks for
     permission.
   - **The precision gaze model.** Its box is unticked by default. This model belongs to a
     third party and its license only allows **non-commercial use**, so it is not part of
     Eyetrack. If you tick the box, you accept that license, and the installer downloads the
     model from its official source and checks its SHA-256 fingerprint. You can also get it
     later from the control panel (**Precision model…**). Without it, Eyetrack runs in simple
     mode, which uses the iris position only and is less precise.
3. **Set up OpenTrack.** Set the input to **UDP over network**, port **4242**. Set the output to
   the one your game supports, usually **freetrack 2.0 Enhanced**. The **Accela** filter is a
   good start; lower its smoothing if the camera feels late.
4. **Start Eyetrack.** On the first launch, a welcome screen explains the calibration and shows
   the camera image. A green frame means your face is detected. Press `Enter` to calibrate, or
   `Esc` to do it later.

Uninstalling (Windows "Installed apps") keeps your calibration and settings, unless you ask to
delete them.

## Calibration

Click **Calibrate gaze** in the control panel. Look at each of the 9 targets while gently moving
your head. The targets sit 10% from the screen edges, so that your gaze is learned all the way to
the corners. If a reflection covers your eyes, a red message appears: move the lamp or tilt the
screen a little.

At the end, a results screen shows where your gaze was estimated for each target, and the error
in degrees. Each point is predicted **without its own data** (cross-validation), so the score is
not flattering. Corners are naturally a bit less precise than the center.

| Score | Mean error |
|---|---|
| **Good** | up to 5.5° |
| **Fair** | up to 7° |
| **Redo** | above 7° |

| Key | Action |
|---|---|
| `Enter` | Accept the calibration |
| `R` | Redo only the red points, or those taken with glare (otherwise the worst one) |
| `Esc` | Cancel and keep the previous calibration |

## Hotkeys

These work even while the game has focus:

| Keys | Action |
|---|---|
| `Ctrl`+`Shift`+`F9` | Recenter: your current head pose and gaze become the center |
| `Ctrl`+`Shift`+`F10` | Turn the gaze influence on or off |
| `Ctrl`+`Shift`+`F11` | Pause: the camera goes back to the center and stays there (same keys to resume) |

You can change them in the profile (`hotkey_recenter`, `hotkey_gaze`, `hotkey_pause`), for
example `"pause"`, `"scrolllock"` or `"ctrl+shift+f12"`; `""` disables one. Many games still
see the keys. Avoid letters (in BeamNG, `R` resets the vehicle), and if `Ctrl` or `Shift`
change gear in your game, pick a single key. Eyetrack tells you at startup if another program
already uses a combination.

## Control panel

- **Status:** face detected or lost, blinks, glare on glasses, frames per second, updates sent to
  OpenTrack, gaze on or off, calibration score.
- **Commands:** recenter, pause, gaze on/off, calibrate, gaze overlay, show camera, open
  OpenTrack, quit.
- **Show camera:** a preview with the detected irises and a green, orange or red frame (tracked,
  blink, lost). Useful to check framing, light and glare. It costs nothing when closed.
- **Profile:** choose a profile and save your settings into it.
- **Live settings:** head amplification, gaze share and gaze smoothing. More in
  **Advanced settings**: up/down amplification, dead zone, language, auto profile switching, and
  opening OpenTrack at startup. Changes apply at once, and are kept only when you click
  **Save settings**.

Closing the panel only hides it: tracking continues. Click the tray icon to reopen it.
Right-click the icon to recenter, pause, toggle gaze or quit.

## Profiles

Profiles are JSON files: `default`, `ets2`, `beamng`, `racing` and `flight_sim`. The main
settings are:

| Key | Effect |
|---|---|
| `head_yaw_gain`, `head_pitch_gain` | Head rotation amplification (2.5 = 20° of head rotation gives 50° in game) |
| `gaze_yaw_gain`, `gaze_pitch_gain` | Share of gaze in the camera rotation |
| `gaze_smoothing_s` | Gaze smoothing (higher = calmer but slower camera) |
| `gaze_deadzone` | Central zone where gaze barely acts |
| `send_rate_hz` | Updates per second sent to OpenTrack (120; 0 = once per webcam image) |
| `output_smoothing_s`, `prediction_s` | Short smoothing between images, and the prediction that makes up for its delay |
| `recalib_on_click` | Gaze recalibration on mouse clicks (off in game profiles) |
| `game_processes` | Games that load this profile automatically, e.g. `["MyGame.exe"]` (`[]` = never) |

Tip: tune head tracking alone first (turn gaze off with `Ctrl`+`Shift`+`F10`), then add gaze
with a moderate share.

## Accuracy and limits

- **Gaze accuracy:** about 3° on targets left out of the calibration, with a Logitech C920 at
  720p and glasses. That is enough to look toward a mirror or a screen area. It is not enough to
  aim at a small button.
- **Light matters:** light your face evenly from the front, and avoid strong reflections on
  glasses.
- **A dedicated infrared tracker is more precise** and works in the dark. Eyetrack is the free
  option for the webcam you already have.
- **Windows only** for now.

## Files

- Program: `%LOCALAPPDATA%\Programs\Eyetrack`, replaced at each update.
- Calibration, profiles and log: `%LOCALAPPDATA%\Eyetrack`. Updates never delete them. The log of
  the last session is `eyetrack.log` (written in French).
- Errors such as a missing or busy camera are shown in a window.

## Run from source

Eyetrack is developed with Python 3.12 on Windows.

```bash
pip install -r requirements.txt
```

```bash
python main.py --gaze-mode net --profile default
```

The MediaPipe face model is downloaded on the first run. The precision gaze model is never
downloaded without your consent: use **Precision model…** in the control panel. On the first
run, finding the best camera mode takes about 20 seconds; the result is saved in
`data/camera_mode.json`. Delete that file to search again.

`Eyetrack.bat` opens a menu (in French) with the tracker, the diagnostics, the tests and the
builds. `python main.py --moniteur` also opens the diagnostic window, with an enlarged video and
a detailed overlay.

### Build the executable and the installer

```bash
python tools/build_exe.py --installer
```

This builds `Eyetrack.exe` with PyInstaller and runs its self-test. It then compiles
`dist\installer\Eyetrack-Setup.exe` with [Inno Setup 6](https://jrsoftware.org/isinfo.php),
using `installer/eyetrack.iss`. Inno Setup is free:
`winget install --id JRSoftware.InnoSetup -e --scope user`. Without `--installer`, the script
installs the build for the current user instead.

Each build writes `LICENSE.txt`, `THIRD_PARTY_NOTICES.txt` (the exact list of bundled libraries,
with versions) and a `licenses` folder next to `Eyetrack.exe`.

### Tests

```bash
python -m unittest discover -s tests -v
```

The tests replay simulated head movements and check the direction of every axis sent to
OpenTrack, as validated in game. They also check gaze direction, pause, recenter, prediction,
the 120 Hz output, hotkeys, calibration, translations and the bundled licenses. Run them after
any change to `face.py` or `output.py`: a failing test means an axis would turn the wrong way in
game.

### Project structure

```
main.py             entry point: capture -> face -> gaze -> filters -> OpenTrack
eyetrack_app.py     entry point of Eyetrack.exe (last profile, OpenTrack, single instance, errors)
capture.py          webcam: picks the mode that delivers the most frames
face.py             MediaPipe Face Landmarker (478 points, irises, 6DoF pose)
normalize.py        face normalization (Zhang 2018) from the landmarks
gaze_model.py       gaze: iris ratios (simple mode) or the MobileOne-S0 network (ONNX, supplied by
                    the user; MobileNetV2 accepted as a fallback), with a verified download
calibration.py      Ridge calibration (head leak compensated, robust centroids) + RLS on clicks
filters.py          One Euro filter (+ median), hold and fade when the face is lost
output.py           OpenTrack UDP output (120 Hz, prediction, pause), gaze curve
panel.py            control panel; tray.py: tray icon; hotkeys.py: global hotkeys
overlay.py          click-through gaze overlay
profile.py          JSON profiles per game, automatic switching
i18n.py             interface language: French text is the key, English is the translation
profiles/           game profiles
tools/              diagnostics, icon, build_exe.py (executable, installer, licenses)
installer/          Inno Setup script
research/           experimental pipeline, outside the tracker (see docs/these/, in French)
tests/              automated tests
docs/               project website (GitHub Pages) and research notes
```

The code comments and the development notes are in French. The detailed project status and
roadmap are in [README.fr.md](README.fr.md).

## License

Eyetrack is free software: you can redistribute it and modify it under the terms of the
[GNU General Public License v3](LICENSE).

The libraries included in the executable keep their own licenses: see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The precision gaze model
([yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation), trained on Gaze360) is
**not** distributed with Eyetrack. Users download it themselves, under its own non-commercial
license.

Game names belong to their respective owners. Eyetrack is not affiliated with them, nor with
OpenTrack.

## Credits

[MediaPipe](https://github.com/google-ai-edge/mediapipe) (face landmarks),
[OpenCV](https://opencv.org/), [ONNX Runtime](https://onnxruntime.ai/),
[OpenTrack](https://github.com/opentrack/opentrack), and
[yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) for the gaze models.
