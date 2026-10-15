# AURORA — Electrical

> Part of the [AURORA rover monorepo](../README.md). Previously its own repo (`NASA-Psyche-Rover/Electrical`); merged here with full commit history.

Schematics and bring-up firmware for the **Cubli reaction-wheel rover**: a ~17 cm cube that moves by spinning internal reaction wheels to tip itself from face to face, built for low-gravity terrain like asteroid 16 Psyche.

## Layout

```text
kicad/                      KiCad project for the Cubli electronics
  CUBLI_schematic.kicad_pro   Open this in KiCad
  CUBLI_schematic.kicad_sch   Current schematic
  CUBLI_schematic_v1.kicad_sch  Earlier revision, kept for reference
  CUBLI_Library.kicad_sym     Custom symbol library
  IMU.kicad_sym               IMU symbol
firmware/                   Arduino / ESP32 test sketches (one folder per sketch)
  imu_test/                   BNO055 orientation + angular velocity readout
  motorfunc_test/             Brushless motor continuous-run test
  MovingMech_1stdraft/        Early draft of the movement mechanism (incomplete)
```

## System overview

- **ESP32** microcontroller as the central controller
- **BNO055 IMU** for orientation feedback
- **Three brushless reaction-wheel motors** with encoders
- **11.1 V LiPo** with fuse, local capacitors, and 5 V regulation

## Notes

- `kicad/sym-lib-table` also points to an `ESP32_RF_Module` symbol at a local path on one teammate's machine. If KiCad can't find it, add that symbol library from your own copy.
- KiCad backups, lock files, and `.kicad_prl` user settings are gitignored, so commit only the real project files.
- Firmware sketches open directly in the Arduino IDE (each sketch is in a folder with the same name, as Arduino expects).
