# Changelog

All notable changes to Eyetrack are listed here. Versions follow
[semantic versioning](https://semver.org/).

## [0.1.0] - 2026-11-15

First public release.

### Features

- 6DoF head tracking with an ordinary webcam (MediaPipe, 478 face landmarks), sent to OpenTrack
  over "UDP over network" (port 4242) at 120 updates per second, with light prediction.
- Gaze adds a calibrated nudge to the in-game camera, with smoothing and a dead zone.
- Works with glasses: reflections on the lenses are detected, and Eyetrack tells you what to
  change.
- 9-point calibration with an honest score (each target predicted without its own data), and a
  quick redo of the missed points.
- Automatic profile switching for Euro Truck Simulator 2, American Truck Simulator,
  BeamNG.drive, Assetto Corsa, iRacing, MSFS, X-Plane and DCS.
- Global hotkeys: recenter (`Ctrl`+`Shift`+`F9`), gaze on/off (`Ctrl`+`Shift`+`F10`) and pause
  (`Ctrl`+`Shift`+`F11`).
- Control panel, tray icon and camera preview, in English or French.
- Windows installer without administrator rights. It offers to install OpenTrack, and to
  download the optional precision gaze model.

### Gaze model

- The default precision model is MobileOne-S0 from
  [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation). MobileNetV2 is accepted as
  a fallback. In a side-by-side test on the same images, both were equally accurate, and
  MobileOne-S0 is half the size and faster.
- The model is not included with Eyetrack: its license allows non-commercial use only. Users
  download it themselves from its official source, and Eyetrack checks its SHA-256 fingerprint.
  Without it, Eyetrack runs in simple mode.

### Known limitations

- Gaze accuracy is about 3° with a Logitech C920 at 720p. That is enough to glance at a mirror
  or a screen area, not to aim at a small button.
- Lighting matters: a dark room lowers the webcam frame rate and the precision.
- Windows only.
- The installer is not signed yet: Windows SmartScreen shows a warning on first launch.
