# Newton's Rings Interactive Simulation 🔍

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/f/f8/Newton_Rings.jpg" alt="Newton's Rings" width="500"/>
  <p><em>Example of Newton's rings observed in a laboratory setting</em></p>
</div>

## 🌈 Overview

This interactive simulation brings the classic Newton's rings experiment to life in your terminal. Experience the fascinating interference patterns that occur when light reflects between a plano-convex lens and a flat glass surface, creating concentric colored rings.

The simulation includes a virtual traveling microscope that allows precise measurements of ring diameters, enabling you to experimentally determine the radius of curvature of the lens using the principles of optical interference.

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/7f/Newton%27s_rings_setup.svg" alt="Newton's Rings Setup" width="400"/>
  <p><em>Experimental setup diagram for Newton's rings</em></p>
</div>

## ✨ Features

- **Real-time Physics Calculations**: Accurately simulates optical interference based on the equation 2μt = mλ
- **Interactive Control**: Move the microscope in real-time with smooth physics-based animation
- **High-Precision Measurements**: Take ultra-precise measurements of ring diameters
- **Customizable Parameters**: Adjust the wavelength of light and radius of curvature
- **Visual Feedback**: ASCII visualization in the terminal plus matplotlib visualization
- **Educational**: Includes explanations of the physics principles and experimental methodology
- **Precision Mode**: Toggle ultra-fine adjustments for microscope positioning

## 🔬 The Physics Behind Newton's Rings

Newton's rings are an interference pattern caused by the reflection of light between a spherical surface and an adjacent flat surface. When light falls on the air gap between these surfaces, some is reflected from the top surface and some from the bottom. 

The optical path difference between these two reflections causes constructive and destructive interference, creating alternating bright and dark rings. The relationship is given by:

```
2μt = mλ
```

Where:
- μ is the refractive index of the medium in the gap (usually air)
- t is the thickness of the air gap
- m is the order of interference (ring number)
- λ is the wavelength of light

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/45/NewtonsRingsDiagram_650.gif" alt="Newton's Rings Formation" width="450"/>
  <p><em>Formation of Newton's rings due to interference</em></p>
</div>

## 🚀 Getting Started

### Prerequisites

- Python 3.6+
- Required libraries: matplotlib, numpy
- Optional: keyboard library for enhanced real-time input

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/newtons-rings.git
   cd newtons-rings
   ```

2. Install the required dependencies:
   ```bash
   pip install numpy matplotlib
   pip install keyboard  # Optional, for enhanced real-time input
   ```

3. Run the simulation:
   ```bash
   python newton_rings_simulation.py
   ```

## 🎮 How to Use

1. **Launch the simulation** to see the Newton's rings pattern
2. **Control the traveling microscope** using keyboard commands:
   - ← and → arrows to move left and right
   - ↑ and ↓ arrows to adjust step size
   - Space to take a measurement
   - P to toggle precision mode
   - Tab to switch between viewing modes
   - M to display measurement data
   - V to visualize with matplotlib

3. **Measure ring diameters** by positioning the microscope at the edges of each ring
4. **Calculate the radius of curvature** based on your measurements

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Newton-rings-radius-measure.jpg" alt="Microscope Measurement" width="400"/>
  <p><em>Taking measurements of Newton's rings with a traveling microscope</em></p>
</div>

## 📊 Experimental Method

1. Position the microscope at one edge of a ring
2. Take a measurement
3. Move to the opposite edge of the same ring
4. Take another measurement
5. The difference between measurements gives the ring diameter
6. Repeat for multiple rings for better accuracy
7. The radius of curvature (R) can be calculated using: R = (r²ₙ - r²ₘ)/[λ(n-m)]

## 🧪 Educational Value

This simulation is perfect for:
- Physics students studying wave optics and interference
- Laboratory preparation for real Newton's rings experiments
- Exploring the relationship between ring diameters and optical properties
- Understanding experimental methods in optical physics


## 🙏 Acknowledgements

- Images courtesy of Wikimedia Commons
- Based on classical optical physics principles established by Sir Isaac Newton 