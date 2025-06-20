# BerryBox: Low-Cost Imaging Station for Seedling Phenotyping

**BerryBox** is a custom-built, low-cost imaging system designed for high-throughput, infrared-based phenotyping of vertically grown *Arabidopsis thaliana* seedlings. It is optimized for use with the DLhook pipeline to automatically quantify apical hook angles.

---

## 📦 Assembly Instructions

### 🧱 Build the Imaging Box
- Construct a closed wooden box (70 cm height × 60 cm length × 60 cm width).
- Keep the front panel removable or open for plate access.

### 🔩 Camera Rack and Holes
- Laser-cut 3 mm thick plexiglass using the [design files](https://github.com/SRobertGroup/DLhook/tree/main/BerryBox).
- Assemble racks using plastic-compatible glue.
- Attach racks to the inner side wall of the box.
- Drill 12 evenly spaced holes (3 columns × 4 rows) aligned with rack positions for camera mounting.

### 🎥 Camera and Hardware Setup
- Mount 12 Raspberry Pi NoIR V2 cameras through the drilled holes.
- Connect cameras to **3 IVPort multiplexers**, and then link them to **3 Raspberry Pi 3 Model B** units.
- Follow [IVPort setup instructions](https://github.com/ivmech/ivport) for configuration.

### 💡 Lighting and Diffusion
- Attach IR LED strips (940 nm) to the back of the box.
- Cut diffusion paper into 3 vertical panels.
- Mount paper between light and racks using 18 metal hooks and secure with paper clips or rubber bands for uniform lighting.

---

## 🔌 Powering the System

- **Power Unit 1 & 2**: Turn on with the button.
- **Power Unit 3**: Plug in manually.
- Begin with **Raspberry Pi #3 (Rpi n3)**:
  - Connect a screen, mouse, and keyboard.
  - Turn on the power supply.
  - Look for red (power) and green (boot) LEDs on the RPi.

---

## 🧫 Loading Petri Plates

- Insert Petri dishes vertically with the **back side facing the camera**.
- Align plates using the internal **alignment line**.
- Plate positions correspond to rows:
  - Top to bottom → Plate 1 → Plate 4

---

## 📸 Picture Acquisition Setup

1. **Navigate to the Script**:
   - Open File Manager on Rpi n3.
   - Go to `ivport-v2` folder and open `kinematic.py`.

2. **Edit Parameters**:
   | Parameter            | Line No. | Notes |
   |----------------------|----------|-------|
   | Number of pictures   | 11       | Total timepoints in time-lapse |
   | Output file names    | 13, 16, 19, 22 | Use underscores `_`, no special characters (e.g., `wt_mut1_mut2_IAA`) |
   | Image frequency      | 23       | Time between captures (sec/min) |

3. **Start Acquisition**:
   ```bash
   cd ivport-v2
   python init_ivport.py
   python kinematic.py
