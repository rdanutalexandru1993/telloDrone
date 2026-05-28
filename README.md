**DESCRIPTION** 
This repository contains experimental code for programming a tello drone. 

## Follow-me wave demo

`follow_me_with_wave.py` connects to the Tello, starts the video stream, takes off,
rotates in 30 degree steps until it sees a face, then watches for a side-to-side
wave near that face.

Each detected wave alternates the drone movement:

- first wave: move closer by 40 cm
- second wave: move back by 40 cm
- third wave: move closer again

Run it while connected to the Tello Wi-Fi:

```powershell
py follow_me_with_wave.py
```

Requirements:

- Python packages used by the existing Tello wrapper: `opencv-python`, `numpy`,
  `pillow`, and `pygame`
- Battery above 35%
- Clear indoor space, good lighting, and one hand ready to press `Ctrl+C`
