# Third-party notices

Eyetrack is free software under the [GNU General Public License v3](LICENSE).

The Windows build (`Eyetrack.exe` and `Eyetrack-Setup.exe`) includes the third-party components
listed below. Each one keeps its own license. All of them can be used in a GPL-3.0 program.

**The exact list, with versions, ships with each build.** When the executable is built,
`tools/build_exe.py` reads which packages PyInstaller actually bundled. It then writes
`THIRD_PARTY_NOTICES.txt` next to `Eyetrack.exe`, and copies every license text into the
`licenses` folder beside it.

## Main components

| Component | What it does in Eyetrack | License |
|---|---|---|
| [Python](https://www.python.org/) | Runtime, with its bundled libraries (OpenSSL, libffi, zlib, bzip2, expat, mpdecimal, Tcl/Tk) and the Microsoft C++ runtime | PSF-2.0 and the licenses listed in Python's own `LICENSE.txt` |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Face landmarks, iris and head pose. Its face landmark model (`face_landmarker.task`) is included too | Apache-2.0 |
| [OpenCV](https://opencv.org/) (`opencv-python` / `opencv-contrib-python`) | Webcam capture and image processing. The FFmpeg library is removed from the build | Apache-2.0, plus the libraries listed in `LICENSE-3RD-PARTY.txt` |
| [ONNX Runtime](https://onnxruntime.ai/) | Runs the optional precision gaze model | MIT, plus `ThirdPartyNotices.txt` |
| [NumPy](https://numpy.org/) | Numerical computing, including the bundled OpenBLAS library | BSD-3-Clause, plus the bundled libraries listed in its license |
| [Matplotlib](https://matplotlib.org/), with contourpy, cycler, kiwisolver, pyparsing, python-dateutil and the DejaVu and STIX fonts | Imported by MediaPipe for its drawing tools | Matplotlib license (PSF-based), BSD, MIT, Apache-2.0, Bitstream Vera and OFL |
| [Pillow](https://python-pillow.org/) | Image library used by Matplotlib | MIT-CMU |
| [pywin32](https://github.com/mhammond/pywin32) | Windows API access | PSF-2.0 |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) and [PortAudio](http://www.portaudio.com/), with cffi | Imported by MediaPipe. Eyetrack uses no audio, and the ASIO build of PortAudio is removed | MIT |
| [certifi](https://github.com/certifi/python-certifi) | Root certificates for HTTPS downloads | MPL-2.0 |
| Small helper packages (Jinja2, MarkupSafe, packaging, six, defusedxml, threadpoolctl, typing_extensions, pyreadline3) | Dependencies of the packages above | BSD, MIT, Apache-2.0 or PSF |

## Not included

- **The precision gaze model.** These are the MobileNetV2 and MobileOne-S0 weights from
  [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation), trained on Gaze360. Their
  license allows non-commercial use only, so Eyetrack never ships them. Users download them
  themselves from the official source, after accepting that license, from the installer or the
  control panel. Without them, Eyetrack works in simple mode.
- **FFmpeg.** Eyetrack does not include it: the webcam goes through DirectShow and Media Foundation.
- **OpenTrack.** It is a separate program, which users install themselves.

## Running from source

`pip install -r requirements.txt` installs the same libraries, each under its own license.
