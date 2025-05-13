"""
Newton's Rings Interactive Simulation
-------------------------------------
This program simulates the Newton's rings experiment with real-time calculations
and interactive controls in the terminal. It visualizes the interference pattern
created when light reflects between a plano-convex lens and a flat glass surface.
The simulation includes a traveling microscope that can be adjusted in real-time
to take precise measurements of the ring positions.

Physics principle: 2μt = mλ (where μ is refractive index, t is thickness, m is order, λ is wavelength)
"""

import math
import random
import time
import os
import matplotlib.pyplot as plt
import numpy as np

# Try to import keyboard module for real-time input, but provide fallback
try:
    import keyboard  # For real-time keyboard input
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

# Constants
SODIUM_WAVELENGTH = 589.3  # nm (sodium D line)
AIR_REFRACTIVE_INDEX = 1.0
DEFAULT_RADIUS_OF_CURVATURE = 100  # cm
MAX_RINGS = 15

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

class TravelingMicroscope:
    """Class representing a traveling microscope for measuring Newton's rings with ultra-high precision"""
    def __init__(self, max_position=30):
        self.position = 0.0  # Current position (0 is center) - using float for higher precision
        self.max_position = max_position  # Maximum distance from center
        self.step_size = 0.1  # Default step size in mm
        self.current_reading = None  # Current measurement reading
        self.is_measuring = False  # Whether currently taking a measurement
        self.measurements = {}  # Dictionary to store measurements {position: reading}
        self.movement_history = []  # Track recent movements for smoother animation
        self.precision_mode = False  # Toggle for ultra-precise movements
        self.sub_pixel_positions = []  # Track sub-pixel positions for smoother animation
        self.last_update_time = time.time()  # For calculating velocity
        self.velocity = 0.0  # Current velocity for momentum physics
        self.acceleration = 0.0  # Current acceleration
        self.movement_smoothness = 0.8  # Higher value = more smoothing (0-1)
        self.parent = None  # Reference to parent simulation for syncing measurements

    def move_left(self, factor=1.0):
        """Move the microscope left (negative direction) with ultra-smooth physics-based movement"""
        if self.position > -self.max_position:
            # Calculate time delta for physics-based movement
            current_time = time.time()
            dt = min(0.1, current_time - self.last_update_time)  # Cap dt to prevent large jumps
            self.last_update_time = current_time
            
            # Apply precision mode if enabled
            actual_step = self.step_size * factor
            if self.precision_mode:
                actual_step /= 10.0  # Much smaller steps in precision mode
            
            # Apply smoother acceleration based on previous velocity
            target_velocity = -actual_step * 10.0  # Target velocity based on step size
            # Smooth velocity transition using exponential moving average
            self.velocity = self.velocity * self.movement_smoothness + target_velocity * (1.0 - self.movement_smoothness)
            
            # Calculate new position using physics integration
            new_position = self.position + self.velocity * dt
            
            # Ensure we don't exceed the maximum position
            if new_position < -self.max_position:
                new_position = -self.max_position
                self.velocity = 0.0  # Stop at boundary
            
            # Calculate actual change
            actual_change = new_position - self.position
            self.position = new_position

            # Add sub-pixel position for ultra-smooth animation
            self.sub_pixel_positions.append((self.position, current_time))
            if len(self.sub_pixel_positions) > 20:  # Keep only recent history
                self.sub_pixel_positions.pop(0)

            # Record movement for animation smoothing with precise timing
            self.movement_history.append(('left', abs(actual_change), current_time))
            if len(self.movement_history) > 15:  # Keep only recent history
                self.movement_history.pop(0)

            return True
        else:
            # Decelerate when at boundary
            self.velocity *= 0.5
            return False

    def move_right(self, factor=1.0):
        """Move the microscope right (positive direction) with ultra-smooth physics-based movement"""
        if self.position < self.max_position:
            # Calculate time delta for physics-based movement
            current_time = time.time()
            dt = min(0.1, current_time - self.last_update_time)  # Cap dt to prevent large jumps
            self.last_update_time = current_time
            
            # Apply precision mode if enabled
            actual_step = self.step_size * factor
            if self.precision_mode:
                actual_step /= 10.0  # Much smaller steps in precision mode
            
            # Apply smoother acceleration based on previous velocity
            target_velocity = actual_step * 10.0  # Target velocity based on step size
            # Smooth velocity transition using exponential moving average
            self.velocity = self.velocity * self.movement_smoothness + target_velocity * (1.0 - self.movement_smoothness)
            
            # Calculate new position using physics integration
            new_position = self.position + self.velocity * dt
            
            # Ensure we don't exceed the maximum position
            if new_position > self.max_position:
                new_position = self.max_position
                self.velocity = 0.0  # Stop at boundary
            
            # Calculate actual change
            actual_change = new_position - self.position
            self.position = new_position

            # Add sub-pixel position for ultra-smooth animation
            self.sub_pixel_positions.append((self.position, current_time))
            if len(self.sub_pixel_positions) > 20:  # Keep only recent history
                self.sub_pixel_positions.pop(0)

            # Record movement for animation smoothing with precise timing
            self.movement_history.append(('right', abs(actual_change), current_time))
            if len(self.movement_history) > 15:  # Keep only recent history
                self.movement_history.pop(0)

            return True
        else:
            # Decelerate when at boundary
            self.velocity *= 0.5
            return False

    def increase_step_size(self):
        """Increase the step size for faster movement with finer granularity"""
        if self.step_size < 2.0:  # Allow larger steps for faster movement across the field
            # Use exponential scaling for more intuitive control
            if self.step_size < 0.1:
                self.step_size *= 1.25  # Smaller increments at low values
            elif self.step_size < 0.5:
                self.step_size *= 1.2  # Medium increments
            else:
                self.step_size *= 1.15  # Larger increments at higher values

            # Round to 3 decimal places for precise display
            self.step_size = round(self.step_size, 3)
        return self.step_size

    def decrease_step_size(self):
        """Decrease the step size for more precise movement with ultra-fine granularity"""
        if self.step_size > 0.001:  # Allow much smaller steps for extreme precision
            # Use exponential scaling for more intuitive control
            if self.step_size <= 0.1:
                self.step_size *= 0.8  # Smaller decrements at low values
                # Ensure we don't go below minimum
                self.step_size = max(0.001, self.step_size)
            elif self.step_size <= 0.5:
                self.step_size *= 0.85  # Medium decrements
            else:
                self.step_size *= 0.9  # Larger decrements at higher values

            # Round to 3 decimal places for precise display
            self.step_size = round(self.step_size, 3)
        return self.step_size

    def toggle_precision_mode(self):
        """Toggle precision mode for ultra-fine adjustments with enhanced feedback"""
        self.precision_mode = not self.precision_mode
        if self.precision_mode:
            # Store the current step size and set a very small one
            self._normal_step_size = self.step_size
            self.step_size = max(0.001, self.step_size / 10.0)  # Even smaller step size
            # Reset velocity for more controlled movement
            self.velocity = 0.0
            # Increase movement smoothness for more precise control
            self.movement_smoothness = 0.9
        else:
            # Restore normal step size if we saved one
            if hasattr(self, '_normal_step_size'):
                self.step_size = self._normal_step_size
            # Reset velocity
            self.velocity = 0.0
            # Reset movement smoothness
            self.movement_smoothness = 0.8
            
        # Round to 3 decimal places for precise display
        self.step_size = round(self.step_size, 3)
        return self.precision_mode

    def take_measurement(self, ring_radii):
        """Take a measurement at the current position with ultra-high precision and detailed analysis"""
        # Find the closest ring to the current position with highest possible precision
        closest_ring = None
        min_distance = float('inf')
        second_closest = None
        second_min_distance = float('inf')

        for ring_num, radius in enumerate(ring_radii):
            distance = abs(abs(self.position) - radius)
            if distance < min_distance:
                second_closest = closest_ring
                second_min_distance = min_distance
                min_distance = distance
                closest_ring = ring_num
            elif distance < second_min_distance:
                second_min_distance = distance
                second_closest = ring_num

        # Add precision-based noise that decreases with better alignment
        # Calculate alignment quality - extremely low noise when very precisely aligned
        alignment_quality = min(1.0, min_distance * 100)  # 0 = perfect alignment
        
        # Base noise is scaled by alignment and precision mode
        base_noise = 0.01  # Base noise level in mm
        if self.precision_mode:
            base_noise *= 0.1  # Much less noise in precision mode
            
        # Scale noise based on alignment quality - extreme precision when perfectly aligned
        if min_distance < 0.001:  # Ultra-precise alignment (< 1 micron)
            noise_scale = 0.01  # Almost no noise
        elif min_distance < 0.01:  # Very precise alignment (< 10 microns)
            noise_scale = 0.1  # Very little noise
        elif min_distance < 0.1:  # Good alignment (< 100 microns)
            noise_scale = 0.2  # Low noise
        else:
            noise_scale = 1.0  # Normal noise

        # Apply the noise with realistic distribution
        # Use triangular distribution for more realistic measurement noise
        noise = (random.triangular(-1.0, 1.0, 0.0) * base_noise * noise_scale)

        # Calculate timestamp for the measurement
        timestamp = time.strftime("%H:%M:%S", time.localtime())

        # Calculate instantaneous velocity at time of measurement
        velocity = 0.0
        if len(self.sub_pixel_positions) >= 2:
            recent_positions = self.sub_pixel_positions[-2:]
            pos_diff = recent_positions[1][0] - recent_positions[0][0]
            time_diff = recent_positions[1][1] - recent_positions[0][1]
            if time_diff > 0:
                velocity = pos_diff / time_diff

        # Determine measurement quality based on multiple factors
        if min_distance < 0.001 and abs(velocity) < 0.001:
            quality = "Ultra-High"  # Perfect alignment and stationary
        elif min_distance < 0.01 and abs(velocity) < 0.01:
            quality = "Very High"   # Nearly perfect alignment and very slow movement
        elif min_distance < 0.05 and abs(velocity) < 0.05:
            quality = "High"        # Good alignment and slow movement
        elif min_distance < 0.1 and abs(velocity) < 0.1:
            quality = "Good"        # Decent alignment and moderate movement
        else:
            quality = "Standard"    # Rough alignment or significant movement

        # Store the measurement with enhanced data
        self.current_reading = {
            'position': self.position,
            'closest_ring': closest_ring,
            'second_closest_ring': second_closest,
            'distance_from_ring': min_distance,
            'reading': self.position + noise,
            'precision': quality,
            'velocity': velocity,
            'noise_estimate': base_noise * noise_scale,
            'timestamp': timestamp,
            'alignment_quality': 100 - alignment_quality,  # 100 = perfect, 0 = poor
        }

        # Use position rounded to 9 decimal places as key to avoid floating point issues
        # This provides nanometer-level precision in the position key
        position_key = round(self.position, 9)
        self.measurements[position_key] = self.current_reading

        # Sync with parent simulation's measurements if parent is set
        if self.parent is not None and closest_ring > 0:  # Only sync non-central measurements
            if self.position < 0:
                self.parent.left_measurements[closest_ring] = self.position
            else:
                self.parent.right_measurements[closest_ring] = self.position

        return self.current_reading

    def get_position_display(self, width=60):
        """Get a string representation of the microscope position for display with enhanced visualization"""
        # Create a scale with the microscope position marked
        scale = ['-'] * width
        center = width // 2

        # Calculate exact position with sub-pixel accuracy
        pos_exact = center + (self.position / self.max_position) * (width // 2)
        pos_index = int(pos_exact)
        pos_fraction = pos_exact - pos_index

        # Ensure pos_index is within bounds
        pos_index = max(0, min(width-1, pos_index))

        # Mark the center position
        scale[center] = '|'

        # Mark the microscope position with different symbol based on precision mode
        if self.precision_mode:
            scale[pos_index] = f"{Colors.CYAN}◆{Colors.RESET}"  # Diamond for precision mode
        else:
            scale[pos_index] = f"{Colors.CYAN}▼{Colors.RESET}"  # Triangle for normal mode

        # Add movement indicator (direction of travel) with velocity indication
        if self.movement_history:
            # Calculate time since last movement
            time_since_last = time.time() - self.movement_history[-1][2]
            
            # Only show movement indicators for recent movement
            if time_since_last < 0.5:
                direction = self.movement_history[-1][0]
                velocity_magnitude = abs(self.velocity)
                
                # Use different indicators based on velocity
                if velocity_magnitude < 0.01:
                    indicator = "·"  # Very slow
                elif velocity_magnitude < 0.1:
                    indicator = direction[0]  # First letter (l or r)
                elif velocity_magnitude < 1.0:
                    indicator = "<" if direction == "left" else ">"  # Medium speed
                else:
                    indicator = "«" if direction == "left" else "»"  # High speed
                
                # Show indicator in appropriate position
                if direction == 'left' and pos_index > 0:
                    scale[pos_index-1] = f"{Colors.CYAN}{indicator}{Colors.RESET}"
                elif direction == 'right' and pos_index < width-1:
                    scale[pos_index+1] = f"{Colors.CYAN}{indicator}{Colors.RESET}"
                
                # Add secondary indicator for higher speeds
                if velocity_magnitude > 0.5 and ((direction == 'left' and pos_index > 1) or 
                                               (direction == 'right' and pos_index < width-2)):
                    if direction == 'left':
                        scale[pos_index-2] = f"{Colors.CYAN}·{Colors.RESET}"
                    else:
                        scale[pos_index+2] = f"{Colors.CYAN}·{Colors.RESET}"

        return ''.join(scale)


class NewtonRingsSimulation:
    def __init__(self):
        # Physics parameters
        self.wavelength = SODIUM_WAVELENGTH  # nm
        self.refractive_index = AIR_REFRACTIVE_INDEX
        self.radius_of_curvature = DEFAULT_RADIUS_OF_CURVATURE * 10**7  # Convert cm to nm

        # Measurement data
        self.ring_measurements = {}
        self.current_ring = 0
        self.left_measurements = {}
        self.right_measurements = {}

        # Initialize the traveling microscope
        self.microscope = TravelingMicroscope()
        self.microscope.parent = self  # Set parent reference for measurement syncing

        # Calculate ring radii
        self.calculate_ring_radii()

        # Start the simulation
        self.run_simulation()

    def calculate_ring_radii(self):
        """Calculate the radii of Newton's rings based on current parameters"""
        self.ring_radii = []

        for m in range(MAX_RINGS + 1):
            # Calculate ring radius using the formula: r² = mλR/μ
            # Where r is ring radius, m is ring order, λ is wavelength,
            # R is radius of curvature, μ is refractive index
            if m == 0:
                # Central dark spot
                radius = 0  # Just for data structure consistency
                self.ring_radii.append(radius)
            else:
                # Calculate physical radius in nm
                physical_radius = math.sqrt(m * self.wavelength * self.radius_of_curvature / self.refractive_index)
                # Convert to mm for easier visualization
                radius_mm = physical_radius / 1_000_000
                self.ring_radii.append(radius_mm)

    def display_welcome(self):
        """Display welcome message and instructions"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.BOLD}{Colors.YELLOW}Newton's Rings Interactive Simulation{Colors.RESET}")
        print(f"{Colors.YELLOW}-------------------------------------{Colors.RESET}")
        print("This program simulates the Newton's rings experiment with real-time calculations")
        print("and interactive controls in the terminal.")
        print()
        print(f"{Colors.BOLD}Physics principle:{Colors.RESET} 2μt = mλ")
        print("  where μ is refractive index, t is thickness, m is order, λ is wavelength")
        print()
        print(f"{Colors.BOLD}Current Parameters:{Colors.RESET}")
        print(f"  - Wavelength (λ): {self.wavelength:.1f} nm (Sodium D line)")
        print(f"  - Refractive Index (μ): {self.refractive_index}")
        print(f"  - Radius of Curvature (R): {self.radius_of_curvature/10**7:.1f} cm")
        print()
        print(f"{Colors.BOLD}Commands:{Colors.RESET}")
        print("  1. Adjust traveling microscope in realtime")
        print("  2. Change wavelength")
        print("  3. Change radius of curvature")
        print("  4. Take measurements")
        print("  5. Calculate radius of curvature from measurements")
        print("  6. Reset measurements")
        print("  7. Show experiment diagram")
        print("  8. Generate matplotlib visualization")
        print("  9. Exit")
        print()

    def display_rings_ascii(self):
        """Display Newton's rings using ASCII art in the terminal"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.BOLD}{Colors.YELLOW}Newton's Rings Visualization{Colors.RESET}")
        print(f"{Colors.YELLOW}---------------------------{Colors.RESET}")

        # Terminal dimensions
        width = 60
        height = 30
        center_x = width // 2
        center_y = height // 2

        # Create a 2D grid for the ASCII art
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        # Draw rings
        for m in range(MAX_RINGS, 0, -1):
            # Scale radius for terminal display
            radius = int(self.ring_radii[m] * 100)
            if radius >= min(center_x, center_y):
                continue

            # Draw a circle using ASCII characters
            for y in range(height):
                for x in range(width):
                    dx = x - center_x
                    dy = y - center_y
                    distance = math.sqrt(dx*dx + dy*dy)

                    if abs(distance - radius) < 0.5:
                        if m % 2 == 1:  # Bright rings
                            grid[y][x] = f"{Colors.YELLOW}●{Colors.RESET}"
                        else:  # Dark rings
                            grid[y][x] = f"{Colors.BLACK}●{Colors.RESET}"

        # Draw the central dark spot
        grid[center_y][center_x] = f"{Colors.BLACK}●{Colors.RESET}"

        # Print the grid
        for row in grid:
            print(''.join(row))

        print()
        print("Press Enter to return to the main menu...")
        input()

    def display_experiment_diagram(self):
        """Display ASCII art diagram of the Newton's rings experiment setup"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.BOLD}{Colors.YELLOW}Newton's Rings Experiment Setup{Colors.RESET}")
        print(f"{Colors.YELLOW}------------------------------{Colors.RESET}")
        print()
        print(f"{Colors.CYAN}                  Traveling Microscope{Colors.RESET}")
        print(f"{Colors.CYAN}                         ↓{Colors.RESET}")
        print(f"{Colors.CYAN}                   _____|_____                  ←→ {Colors.RESET}")
        print(f"{Colors.CYAN}                  |     |     |                 Horizontal{Colors.RESET}")
        print(f"{Colors.CYAN}                  |     |     |                 Movement{Colors.RESET}")
        print(f"{Colors.CYAN}                  |_____|_____|{Colors.RESET}")
        print(f"{Colors.CYAN}                        |{Colors.RESET}")
        print(f"{Colors.CYAN}                        |{Colors.RESET}")
        print(f"{Colors.CYAN}                        ↓{Colors.RESET}")
        print(f"{Colors.YELLOW}                     \\  |  /{Colors.RESET}")
        print(f"{Colors.YELLOW}                      \\ | /{Colors.RESET}")
        print(f"{Colors.YELLOW}                       \\|/{Colors.RESET}")
        print(f"{Colors.YELLOW}                        V{Colors.RESET}")
        print(f"{Colors.BLUE}          ____________________________________________{Colors.RESET}")
        print(f"{Colors.BLUE}         /                                           \\{Colors.RESET}")
        print(f"{Colors.BLUE}        /                                             \\{Colors.RESET}")
        print(f"{Colors.BLUE}       /                                               \\{Colors.RESET}")
        print(f"{Colors.BLUE}      /                                                 \\{Colors.RESET}")
        print(f"{Colors.BLUE} _____/                                                   \\_____")
        print(f"{Colors.BLUE}|____________________________________________________________{Colors.RESET}")
        print()
        print(f"{Colors.BOLD}Explanation:{Colors.RESET}")
        print("1. A plano-convex lens is placed on a flat glass plate")
        print("2. Monochromatic light (e.g., sodium light) illuminates from above")
        print("3. Interference occurs in the air gap between the lens and plate")
        print("4. This creates concentric rings (Newton's rings) visible through a microscope")
        print("5. The traveling microscope can be moved horizontally to measure ring positions")
        print("6. The central spot is dark due to a phase change on reflection")
        print()
        print("Press Enter to return to the main menu...")
        input()

    def change_wavelength(self):
        """Allow user to change the wavelength"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.BOLD}{Colors.YELLOW}Change Wavelength{Colors.RESET}")
        print(f"{Colors.YELLOW}----------------{Colors.RESET}")
        print()
        print(f"Current wavelength: {self.wavelength:.1f} nm")
        print()
        print("Common wavelengths:")
        print("1. Sodium D line (589.3 nm) - Yellow")
        print("2. Mercury green line (546.1 nm)")
        print("3. Hydrogen-beta (486.1 nm) - Blue-green")
        print("4. Hydrogen-alpha (656.3 nm) - Red")
        print("5. Custom wavelength")
        print()

        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            self.wavelength = 589.3
        elif choice == '2':
            self.wavelength = 546.1
        elif choice == '3':
            self.wavelength = 486.1
        elif choice == '4':
            self.wavelength = 656.3
        elif choice == '5':
            try:
                custom = float(input("Enter custom wavelength (400-700 nm): "))
                if 400 <= custom <= 700:
                    self.wavelength = custom
                else:
                    print("Wavelength must be between 400 and 700 nm. Using previous value.")
                    time.sleep(1.5)
            except ValueError:
                print("Invalid input. Using previous value.")
                time.sleep(1.5)

        # Recalculate ring radii with new wavelength
        self.calculate_ring_radii()

        print(f"\nWavelength set to {self.wavelength:.1f} nm")
        time.sleep(1.5)

    def change_radius(self):
        """Allow user to change the radius of curvature"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.BOLD}{Colors.YELLOW}Change Radius of Curvature{Colors.RESET}")
        print(f"{Colors.YELLOW}-------------------------{Colors.RESET}")
        print()
        print(f"Current radius of curvature: {self.radius_of_curvature/10**7:.1f} cm")
        print()

        try:
            new_radius = float(input("Enter new radius of curvature (50-200 cm): "))
            if 50 <= new_radius <= 200:
                self.radius_of_curvature = new_radius * 10**7  # Convert cm to nm
                # Recalculate ring radii with new radius
                self.calculate_ring_radii()
                print(f"\nRadius of curvature set to {new_radius:.1f} cm")
            else:
                print("Radius must be between 50 and 200 cm. Using previous value.")
        except ValueError:
            print("Invalid input. Using previous value.")

        time.sleep(1.5)

    def take_measurements(self):
        """Take measurements of ring diameters"""
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"{Colors.BOLD}{Colors.YELLOW}Take Measurements{Colors.RESET}")
            print(f"{Colors.YELLOW}----------------{Colors.RESET}")
            print()
            self.display_measurements()
            print()
            print("1. Measure left side of a ring")
            print("2. Measure right side of a ring")
            print("3. Return to main menu")
            print()

            choice = input("Enter your choice (1-3): ")

            if choice == '1':
                self.measure_left_side()
            elif choice == '2':
                self.measure_right_side()
            elif choice == '3':
                break

    def measure_left_side(self):
        """Measure the left side of a ring"""
        try:
            ring_number = int(input(f"Enter ring number to measure (1-{MAX_RINGS}): "))
            if 1 <= ring_number <= MAX_RINGS:
                # Add some random noise for realism
                noise = (random.random() - 0.5) * 0.02
                self.left_measurements[ring_number] = -self.ring_radii[ring_number] + noise
                print(f"Measured left side of ring {ring_number}")
                time.sleep(1)
            else:
                print(f"Ring number must be between 1 and {MAX_RINGS}")
                time.sleep(1.5)
        except ValueError:
            print("Invalid input")
            time.sleep(1.5)

    def measure_right_side(self):
        """Measure the right side of a ring"""
        try:
            ring_number = int(input(f"Enter ring number to measure (1-{MAX_RINGS}): "))
            if 1 <= ring_number <= MAX_RINGS:
                # Add some random noise for realism
                noise = (random.random() - 0.5) * 0.02
                self.right_measurements[ring_number] = self.ring_radii[ring_number] + noise
                print(f"Measured right side of ring {ring_number}")
                time.sleep(1)
            else:
                print(f"Ring number must be between 1 and {MAX_RINGS}")
                time.sleep(1.5)
        except ValueError:
            print("Invalid input")
            time.sleep(1.5)

    def display_measurements(self):
        """Display current measurements"""
        print(f"{Colors.BOLD}Current Measurements:{Colors.RESET}")
        if self.left_measurements or self.right_measurements:
            print("Ring #  |  Left Position  |  Right Position  |  Diameter²")
            print("------------------------------------------------------")

            for ring in sorted(set(list(self.left_measurements.keys()) + list(self.right_measurements.keys()))):
                left_pos = self.left_measurements.get(ring, "N/A")
                right_pos = self.right_measurements.get(ring, "N/A")

                if left_pos != "N/A" and right_pos != "N/A":
                    diameter = abs(right_pos - left_pos)
                    diameter_squared = diameter ** 2
                    print(f"{ring:6d}  |  {left_pos:13.4f}  |  {right_pos:14.4f}  |  {diameter_squared:10.6f}")
                else:
                    print(f"{ring:6d}  |  {left_pos:13}  |  {right_pos:14}  |  N/A")
        else:
            print("No measurements taken yet.")

    def calculate_radius_from_measurements(self):
        """Calculate the radius of curvature from measurements"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.BOLD}{Colors.YELLOW}Calculate Radius of Curvature{Colors.RESET}")
        print(f"{Colors.YELLOW}---------------------------{Colors.RESET}")
        print()

        valid_rings = []
        diameters_squared = []

        for ring in sorted(self.left_measurements.keys()):
            if ring in self.right_measurements:
                valid_rings.append(ring)
                diameter = abs(self.right_measurements[ring] - self.left_measurements[ring])
                diameters_squared.append(diameter ** 2)

        if len(valid_rings) < 2:
            print("Need at least two complete ring measurements to calculate radius.")
            print("Please measure both sides of at least two rings.")
            print()
            input("Press Enter to return to the main menu...")
            return

        # Calculate radius of curvature using the formula: Dn² = n·λ·R
        # Where Dn is the diameter of the nth ring, λ is wavelength, R is radius of curvature

        # Display the measurements used for calculation
        print("Using the following measurements:")
        print("Ring #  |  Diameter  |  Diameter²")
        print("--------------------------------")
        for i, ring in enumerate(valid_rings):
            diameter = abs(self.right_measurements[ring] - self.left_measurements[ring])
            print(f"{ring:6d}  |  {diameter:9.4f}  |  {diameters_squared[i]:10.6f}")
        print()

        # Linear regression to find the slope
        n = len(valid_rings)
        sum_x = sum(valid_rings)
        sum_y = sum(diameters_squared)
        sum_xy = sum(x * y for x, y in zip(valid_rings, diameters_squared))
        sum_x_squared = sum(x ** 2 for x in valid_rings)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)

        # Calculate radius of curvature
        calculated_radius = slope / (self.wavelength * 10**-7)  # Convert nm to cm

        print(f"{Colors.BOLD}Results:{Colors.RESET}")
        print(f"Calculated Radius of Curvature: {calculated_radius:.2f} cm")
        print(f"Actual Radius of Curvature: {self.radius_of_curvature/10**7:.2f} cm")
        print(f"Error: {abs(calculated_radius - self.radius_of_curvature/10**7)/(self.radius_of_curvature/10**7)*100:.2f}%")
        print()

        print("The formula used: Dn² = n·λ·R")
        print("Where:")
        print("  Dn = diameter of the nth ring")
        print("  n = ring number")
        print("  λ = wavelength")
        print("  R = radius of curvature")
        print()

        input("Press Enter to return to the main menu...")

    def reset_all_measurements(self):
        """Reset all measurements"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.BOLD}{Colors.YELLOW}Reset Measurements{Colors.RESET}")
        print(f"{Colors.YELLOW}-----------------{Colors.RESET}")
        print()

        confirm = input("Are you sure you want to reset all measurements? (y/n): ")
        if confirm.lower() == 'y':
            self.left_measurements = {}
            self.right_measurements = {}
            self.microscope.measurements = {}
            self.microscope.current_reading = None
            print("All measurements have been reset.")
        else:
            print("Reset cancelled.")

        time.sleep(1.5)


    def adjust_microscope_realtime(self):
        """Interactive mode for adjusting the traveling microscope in real-time with ultra-smooth animations"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.BOLD}{Colors.YELLOW}Traveling Microscope Adjustment{Colors.RESET}")
        print(f"{Colors.YELLOW}-----------------------------{Colors.RESET}")
        print()
        print(f"{Colors.BOLD}Controls:{Colors.RESET}")
        print("  ← / A: Move microscope left")
        print("  → / D: Move microscope right")
        print("  + / =: Increase step size")
        print("  - / _: Decrease step size")
        print("  P: Toggle precision mode for ultra-fine adjustments")
        print("  Space: Take measurement at current position")
        print("  M: View all measurements")
        print("  C: Clear all microscope measurements")
        print("  Q: Return to main menu")
        print()

        # Main interactive loop
        running = True
        last_update = time.time()
        update_interval = 0.01  # Ultra-fast update interval (10ms) for extremely smooth real-time display

        # Display initial view
        self.display_microscope_view()

        # Track continuous movement for smoother animation
        continuous_movement = False
        last_movement_time = time.time()
        movement_acceleration = 1.0  # Start with normal speed
        
        # Performance tracking to ensure responsive updates
        frame_times = []
        
        # Keep track of last position to detect small changes
        last_position = self.microscope.position
        position_change_threshold = 0.000001  # Detect even tiny changes for perfect precision

        while running:
            frame_start_time = time.time()
            current_time = frame_start_time
            movement_detected = False
            position_changed = abs(self.microscope.position - last_position) > position_change_threshold
            last_position = self.microscope.position

            # Process keyboard input based on available modules
            if KEYBOARD_AVAILABLE:
                # Real-time keyboard input with keyboard module
                try:
                    # Handle movement with acceleration for smoother motion
                    if keyboard.is_pressed('left') or keyboard.is_pressed('a'):
                        # Calculate time-based acceleration for smooth movement
                        if continuous_movement and current_time - last_movement_time < 1.0:
                            # Accelerate up to 4x speed if key is held down with smoother ramping
                            movement_acceleration = min(4.0, movement_acceleration * 1.05)
                        else:
                            movement_acceleration = 1.0
                            continuous_movement = True

                        # Apply multiple small movements for smoother animation
                        steps = max(1, int(movement_acceleration * 1.5))  # More steps for smoother motion
                        step_size = movement_acceleration / steps  # Divide the movement into smaller increments
                        
                        for _ in range(steps):
                            self.microscope.move_left(step_size)  # Smaller movements for smoother animation

                        last_movement_time = current_time
                        movement_detected = True

                    elif keyboard.is_pressed('right') or keyboard.is_pressed('d'):
                        # Calculate time-based acceleration for smooth movement
                        if continuous_movement and current_time - last_movement_time < 1.0:
                            # Accelerate up to 4x speed if key is held down with smoother ramping
                            movement_acceleration = min(4.0, movement_acceleration * 1.05)
                        else:
                            movement_acceleration = 1.0
                            continuous_movement = True

                        # Apply multiple small movements for smoother animation
                        steps = max(1, int(movement_acceleration * 1.5))  # More steps for smoother motion
                        step_size = movement_acceleration / steps  # Divide the movement into smaller increments
                        
                        for _ in range(steps):
                            self.microscope.move_right(step_size)  # Smaller movements for smoother animation

                        last_movement_time = current_time
                        movement_detected = True
                    else:
                        # Gradual deceleration when keys are released for more natural motion
                        if continuous_movement:
                            movement_acceleration = max(1.0, movement_acceleration * 0.95)
                            if movement_acceleration < 1.05:  # Close enough to 1.0
                                continuous_movement = False
                                movement_acceleration = 1.0

                    # Handle other controls with responsive debouncing
                    if keyboard.is_pressed('+') or keyboard.is_pressed('='):
                        new_step = self.microscope.increase_step_size()
                        print(f"\nStep size increased to {new_step:.3f} mm")
                        time.sleep(0.15)  # Short delay for responsiveness while preventing too rapid changes

                    if keyboard.is_pressed('-') or keyboard.is_pressed('_'):
                        new_step = self.microscope.decrease_step_size()
                        print(f"\nStep size decreased to {new_step:.3f} mm")
                        time.sleep(0.15)  # Short delay for responsiveness while preventing too rapid changes

                    if keyboard.is_pressed('space'):
                        reading = self.microscope.take_measurement(self.ring_radii)
                        print(f"\nMeasurement taken at position {reading['position']:.6f} mm")
                        print(f"Precision: {reading['precision']}, Time: {reading['timestamp']}")
                        time.sleep(0.15)  # Short delay for responsiveness while preventing multiple measurements

                    if keyboard.is_pressed('p'):
                        # Toggle precision mode
                        precision_mode = self.microscope.toggle_precision_mode()
                        if precision_mode:
                            print(f"\nPrecision mode enabled. Step size: {self.microscope.step_size:.3f} mm")
                        else:
                            print(f"\nPrecision mode disabled. Step size: {self.microscope.step_size:.3f} mm")
                        time.sleep(0.15)  # Short delay for responsiveness while preventing too rapid changes

                    if keyboard.is_pressed('m'):
                        self.display_microscope_measurements()
                        # Reset the view after returning from measurements
                        self.display_microscope_view()
                        last_update = time.time()

                    if keyboard.is_pressed('c'):
                        self.microscope.measurements = {}
                        print("\nAll measurements cleared")
                        time.sleep(0.15)  # Short delay for responsiveness

                    if keyboard.is_pressed('q'):
                        running = False

                except Exception as e:
                    # If keyboard module fails, fall back to manual input
                    print(f"\nKeyboard input error: {e}")
                    key = input("\nEnter command (A/D/+/-/Space/M/C/Q): ").lower()
                    self.process_microscope_key(key)
                    if key == 'q':
                        running = False
            else:
                # Manual input mode for systems without keyboard module
                print("\nKeyboard module not available. Using manual input mode.")
                key = input("Enter command (A/D/+/-/Space/M/C/Q): ").lower()
                self.process_microscope_key(key)
                if key == 'q':
                    running = False

            # Determine if we should update the display
            # Update in these cases:
            # 1. Movement detected
            # 2. Position has changed (even slightly)
            # 3. Regular time interval has passed
            # This ensures both smooth continuous movement and updates for small changes
            should_update = (
                movement_detected or 
                position_changed or 
                current_time - last_update > update_interval
            )

            if should_update:
                self.display_microscope_view()
                last_update = current_time
                
                # Track frame time for performance monitoring
                frame_time = time.time() - frame_start_time
                frame_times.append(frame_time)
                if len(frame_times) > 100:  # Keep only recent history
                    frame_times.pop(0)

            # Adaptive delay to balance responsiveness and CPU usage
            # Use extremely short delay during movement for ultra-responsive feedback
            if movement_detected:
                time.sleep(0.001)  # Extremely short delay during movement (1ms)
            else:
                # Adjust delay based on recent performance
                if frame_times:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    # If rendering is fast, we can afford slightly longer delay when idle
                    if avg_frame_time < 0.01:  # Very fast rendering
                        time.sleep(0.005)  # 5ms delay when idle and rendering is fast
                    else:
                        # Keep delay shorter when rendering is slower
                        time.sleep(0.003)  # 3ms delay
                else:
                    time.sleep(0.005)  # Default 5ms delay when idle

    def process_microscope_key(self, key):
        """Process a single key press for microscope control"""
        if key in ['a', 'left']:
            self.microscope.move_left()
        elif key in ['d', 'right']:
            self.microscope.move_right()
        elif key in ['+', '=']:
            new_step = self.microscope.increase_step_size()
            print(f"\nStep size increased to {new_step:.2f} mm")
        elif key in ['-', '_']:
            new_step = self.microscope.decrease_step_size()
            print(f"\nStep size decreased to {new_step:.2f} mm")
        elif key == ' ' or key == 'space':
            reading = self.microscope.take_measurement(self.ring_radii)
            print(f"\nMeasurement taken at position {reading['position']:.6f} mm")
        elif key == 'm':
            self.display_microscope_measurements()
        elif key == 'c':
            self.microscope.measurements = {}
            print("\nAll measurements cleared")
        elif key == 'p':
            # Toggle precision mode
            precision_mode = self.microscope.toggle_precision_mode()
            if precision_mode:
                print(f"\nPrecision mode enabled. Step size: {self.microscope.step_size:.2f} mm")
            else:
                print(f"\nPrecision mode disabled. Step size: {self.microscope.step_size:.2f} mm")

    def display_microscope_view(self):
        """Display the current view through the traveling microscope with enhanced precision and real-time visualization"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.BOLD}{Colors.YELLOW}Traveling Microscope View{Colors.RESET}")
        print(f"{Colors.YELLOW}------------------------{Colors.RESET}")

        # Display microscope position on a high-precision scale
        scale_width = 100  # Wider scale for better precision
        print("\nMicroscope Position:")

        # Create a more detailed position scale with markers
        scale = ['-'] * scale_width
        center = scale_width // 2

        # Add position markers every 1mm for ultra-precise representation
        for i in range(-30, 31, 1):
            marker_pos = center + int((i / self.microscope.max_position) * (scale_width // 2))
            if 0 <= marker_pos < scale_width:
                if i % 5 == 0:  # Make 5mm markers more prominent
                    scale[marker_pos] = f"{Colors.WHITE}|{Colors.RESET}"
                else:
                    scale[marker_pos] = "·"

        # Add numerical labels with finer gradations
        scale_labels = ' ' * 5
        for i in range(-30, 31, 5):
            label_pos = center + int((i / self.microscope.max_position) * (scale_width // 2))
            if 0 <= label_pos < scale_width - 2:
                # Add leading spaces for alignment
                label = f"{i:2d}"
                offset = len(label) // 2
                scale_labels = scale_labels[:label_pos-offset] + label + scale_labels[label_pos-offset+len(label):]

        # Mark the center position
        scale[center] = f"{Colors.GREEN}|{Colors.RESET}"

        # Mark the microscope position with ultra-high precision
        # Calculate the exact fractional position for sub-pixel positioning
        pos_exact = center + (self.microscope.position / self.microscope.max_position) * (scale_width // 2)
        pos_index = int(pos_exact)
        pos_fraction = pos_exact - pos_index
        
        # Ensure indices are within bounds
        pos_index = max(0, min(scale_width-1, pos_index))
        
        # Mark the exact position with a special character based on precision mode
        if self.microscope.precision_mode:
            position_marker = f"{Colors.CYAN}◆{Colors.RESET}"  # Diamond for precision mode
        else:
            position_marker = f"{Colors.CYAN}▼{Colors.RESET}"  # Triangle for normal mode
        
        scale[pos_index] = position_marker

        # Add fractional position indicator for sub-pixel precision
        if pos_index < scale_width-1 and pos_fraction > 0.5:
            scale[pos_index+1] = f"{Colors.CYAN}·{Colors.RESET}"
        elif pos_index > 0 and pos_fraction < 0.5:
            scale[pos_index-1] = f"{Colors.CYAN}·{Colors.RESET}"

        # Show the direction of movement with animated indicators
        if self.microscope.movement_history and time.time() - self.microscope.movement_history[-1][2] < 0.25:
            direction = self.microscope.movement_history[-1][0]
            # Create a multi-character movement indicator for smoother appearance
            if direction == 'left' and pos_index > 2:
                scale[pos_index-1] = f"{Colors.CYAN}<{Colors.RESET}" if scale[pos_index-1] == '-' else scale[pos_index-1]
                scale[pos_index-2] = f"{Colors.CYAN}·{Colors.RESET}" if scale[pos_index-2] == '-' else scale[pos_index-2]
            elif direction == 'right' and pos_index < scale_width-3:
                scale[pos_index+1] = f"{Colors.CYAN}>{Colors.RESET}" if scale[pos_index+1] == '-' else scale[pos_index+1]
                scale[pos_index+2] = f"{Colors.CYAN}·{Colors.RESET}" if scale[pos_index+2] == '-' else scale[pos_index+2]

        print(scale_labels)
        print(''.join(scale))
        # Display exact position with more decimal places for ultra-precision
        print(f"Position: {self.microscope.position:.6f} mm, Step Size: {self.microscope.step_size:.3f} mm")
        if self.microscope.precision_mode:
            print(f"{Colors.GREEN}Precision Mode Enabled{Colors.RESET}")

        # Display the rings view at current position with enhanced detail
        width = 100  # Wider view for better detail
        height = 30  # Taller view for better aspect ratio
        center_x = width // 2
        center_y = height // 2

        # Create a 2D grid for the ASCII art with enhanced characters
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        # Calculate the visible range based on microscope position with higher precision
        # Use a more precise scaling factor to show subtle movements
        scale_factor = 4.0  # Higher scale factor for more detailed visualization
        visible_center_x = center_x - int(self.microscope.position * scale_factor)

        # Draw background grid for better position reference (faint dots)
        for y in range(0, height, 2):
            for x in range(0, width, 2):
                grid[y][x] = f"{Colors.BLACK}·{Colors.RESET}"

        # Draw rings with ultra-enhanced precision using sub-pixel rendering technique
        for m in range(MAX_RINGS, 0, -1):
            # Scale radius for terminal display with better precision
            radius = self.ring_radii[m] * scale_factor
            if radius >= min(center_x, center_y) * 1.2:  # Allow slightly larger rings to be partially visible
                continue

            # Draw a circle using ASCII characters with sub-pixel precision
            for y in range(height):
                for x in range(width):
                    dx = x - visible_center_x
                    dy = y - center_y
                    distance = math.sqrt(dx*dx + dy*dy)

                    # Use smaller tolerance for more precise rings with gradual transitions
                    # Calculate how close we are to the exact radius
                    precision = abs(distance - radius)
                    
                    if precision < 1.0:  # Thicker rings for better visibility
                        intensity = 1.0 - precision  # Value between 0-1 based on distance from exact radius
                        
                        if m % 2 == 1:  # Bright rings
                            # Use different characters based on intensity for smooth gradation
                            if intensity > 0.8:
                                grid[y][x] = f"{Colors.YELLOW}●{Colors.RESET}"
                            elif intensity > 0.6:
                                grid[y][x] = f"{Colors.YELLOW}◉{Colors.RESET}"
                            elif intensity > 0.4:
                                grid[y][x] = f"{Colors.YELLOW}○{Colors.RESET}"
                            elif intensity > 0.2:
                                grid[y][x] = f"{Colors.YELLOW}·{Colors.RESET}"
                        else:  # Dark rings
                            if intensity > 0.8:
                                grid[y][x] = f"{Colors.BLACK}●{Colors.RESET}"
                            elif intensity > 0.6:
                                grid[y][x] = f"{Colors.BLACK}◉{Colors.RESET}"
                            elif intensity > 0.4:
                                grid[y][x] = f"{Colors.BLACK}○{Colors.RESET}"
                            elif intensity > 0.2:
                                grid[y][x] = f"{Colors.BLACK}·{Colors.RESET}"

        # Draw the central dark spot if visible with enhanced appearance
        if 0 <= visible_center_x < width:
            # Draw a more precise central spot with gradation
            for y in range(center_y-2, center_y+3):
                for x in range(visible_center_x-2, visible_center_x+3):
                    if 0 <= y < height and 0 <= x < width:
                        distance = math.sqrt((x-visible_center_x)**2 + (y-center_y)**2)
                        if distance <= 2:
                            intensity = 1.0 - distance/2  # Gradation based on distance from center
                            if intensity > 0.7:
                                grid[y][x] = f"{Colors.BLACK}●{Colors.RESET}"
                            elif intensity > 0.4:
                                grid[y][x] = f"{Colors.BLACK}◉{Colors.RESET}"
                            elif intensity > 0.1:
                                grid[y][x] = f"{Colors.BLACK}·{Colors.RESET}"

        # Draw the microscope crosshair with enhanced visibility using unicode box characters
        for y in range(height):
            if 0 <= center_x < width:
                grid[y][center_x] = f"{Colors.CYAN}│{Colors.RESET}"  # Vertical line

        for x in range(width):
            if 0 <= center_y < height:
                grid[center_y][x] = f"{Colors.CYAN}─{Colors.RESET}"  # Horizontal line

        # Mark the center of the crosshair
        if 0 <= center_x < width and 0 <= center_y < height:
            grid[center_y][center_x] = f"{Colors.CYAN}┼{Colors.RESET}"

        # Add ring number labels with enhanced visibility
        for m in range(1, MAX_RINGS + 1):
            radius = self.ring_radii[m] * scale_factor
            # Only label rings that are close to the current view
            if abs(radius - abs(self.microscope.position * scale_factor)) < width/3:
                # Calculate position to place the label
                label_x = visible_center_x + int(radius) if self.microscope.position <= 0 else visible_center_x - int(radius)
                if 0 <= label_x < width and 0 <= center_y+2 < height:
                    # Add ring number with color coding based on proximity to current position
                    proximity = abs(radius - abs(self.microscope.position * scale_factor))
                    ring_label = f"{m}"
                    
                    # Choose color based on proximity to current position
                    if proximity < 5:
                        label_color = Colors.GREEN  # Very close to current position
                    elif proximity < 15:
                        label_color = Colors.YELLOW  # Moderately close
                    else:
                        label_color = Colors.WHITE  # Further away
                    
                    for i, char in enumerate(ring_label):
                        if 0 <= label_x + i < width:
                            grid[center_y+2][label_x + i] = f"{label_color}{char}{Colors.RESET}"

        # Add exact measurement markers for positions where measurements were taken
        for pos_key in self.microscope.measurements:
            measurement_pos = self.microscope.measurements[pos_key]['position']
            screen_pos = visible_center_x + int(measurement_pos * scale_factor)
            if 0 <= screen_pos < width:
                # Mark measured points with distinctive symbols
                mark_y = center_y - 3  # Place above the crosshair
                if 0 <= mark_y < height:
                    grid[mark_y][screen_pos] = f"{Colors.MAGENTA}▼{Colors.RESET}"  # Measurement marker

        # Print the grid with a border for better framing
        print(f"\n{Colors.BOLD}Microscope View:{Colors.RESET}")
        print("┌" + "─" * width + "┐")
        for row in grid:
            print("│" + ''.join(row) + "│")
        print("└" + "─" * width + "┘")

        # Display current measurement with enhanced detail and real-time calculations
        if self.microscope.current_reading:
            reading = self.microscope.current_reading
            print(f"\n{Colors.BOLD}{Colors.GREEN}Current Reading:{Colors.RESET}")
            print(f"Position: {reading['position']:.6f} mm")
            if reading['closest_ring'] > 0:
                print(f"Closest Ring: {reading['closest_ring']}")
                print(f"Distance from Ring: {reading['distance_from_ring']:.6f} mm")

                # Calculate and show the wavelength based on this measurement with more details
                if reading['closest_ring'] > 0:
                    ring_number = reading['closest_ring']
                    ring_radius = self.ring_radii[ring_number]
                    calculated_wavelength = (ring_radius**2 * self.refractive_index) / (ring_number * self.radius_of_curvature/10**7)
                    calculated_wavelength *= 1_000_000  # Convert to nm
                    error_pct = abs(calculated_wavelength-self.wavelength)/self.wavelength*100
                    
                    # Color code the error value
                    if error_pct < 1.0:
                        error_color = Colors.GREEN
                    elif error_pct < 5.0:
                        error_color = Colors.YELLOW
                    else:
                        error_color = Colors.RED
                    
                    print(f"Calculated λ from this ring: {calculated_wavelength:.4f} nm")
                    print(f"Actual λ: {self.wavelength:.4f} nm (Error: {error_color}{error_pct:.4f}%{Colors.RESET})")
                    
                    # Add additional calculated physical values for more detailed display
                    thickness = ring_radius**2 / (2 * self.radius_of_curvature)
                    print(f"Air gap at this ring: {thickness:.4e} mm")
                    print(f"Order of interference: {ring_number}")
            else:
                print(f"{Colors.BOLD}{Colors.BLUE}At central spot{Colors.RESET}")
                print(f"Air gap approaches zero at center")

        # Real-time calculation of nearest ring regardless of measurement
        # This shows continuous updates as the microscope moves
        nearest_ring = None
        min_distance = float('inf')
        for ring_num, radius in enumerate(self.ring_radii):
            distance = abs(abs(self.microscope.position) - radius)
            if distance < min_distance:
                min_distance = distance
                nearest_ring = ring_num
        
        if nearest_ring is not None and nearest_ring > 0:
            print(f"\n{Colors.BOLD}Real-time Analysis:{Colors.RESET}")
            print(f"Nearest Ring: {nearest_ring}, Distance: {min_distance:.6f} mm")
            if min_distance < 0.01:
                print(f"{Colors.GREEN}✓ Precisely aligned with ring {nearest_ring}{Colors.RESET}")
            
            # Calculate potential measurement error in real-time
            potential_error = min_distance / self.ring_radii[nearest_ring] * 100
            print(f"Potential measurement error: {potential_error:.4f}%")

        # Display controls reminder with better formatting
        print(f"\n{Colors.BOLD}Controls:{Colors.RESET}")
        print(f"  {Colors.CYAN}←/A{Colors.RESET}: Move Left     {Colors.CYAN}→/D{Colors.RESET}: Move Right")
        print(f"  {Colors.CYAN}+/={Colors.RESET}: Increase Step {Colors.CYAN}-/_{Colors.RESET}: Decrease Step")
        print(f"  {Colors.CYAN}P{Colors.RESET}: Toggle Precision Mode {Colors.BOLD}{'[ON]' if self.microscope.precision_mode else '[OFF]'}{Colors.RESET}")
        print(f"  {Colors.CYAN}Space{Colors.RESET}: Take Measurement")
        print(f"  {Colors.CYAN}M{Colors.RESET}: View All Measurements  {Colors.CYAN}C{Colors.RESET}: Clear Measurements  {Colors.CYAN}Q{Colors.RESET}: Return to Menu")

    def display_microscope_measurements(self):
        """Display all measurements taken with the microscope with enhanced detailed visualization"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.BOLD}{Colors.YELLOW}Microscope Measurements - Ultra-Precision Analysis{Colors.RESET}")
        print(f"{Colors.YELLOW}------------------------------------------{Colors.RESET}")

        if not self.microscope.measurements:
            print("\nNo measurements have been taken yet.")
            print("\nPress Enter to return...")
            input()
            return

        # Sort measurements by position
        sorted_positions = sorted(self.microscope.measurements.keys())

        # Create a visual representation of all measurements on a scale with enhanced precision
        scale_width = 100  # Wider scale for more detailed representation
        center = scale_width // 2

        print(f"\n{Colors.BOLD}Measurement Positions (High Precision Scale):{Colors.RESET}")

        # Draw a more precise scale with finer gradations
        scale = ['-'] * scale_width
        
        # Add position markers with finer graduation
        for i in range(-30, 31, 1):
            marker_pos = center + int((i / self.microscope.max_position) * (scale_width // 2))
            if 0 <= marker_pos < scale_width:
                if i % 5 == 0:  # Major markers
                    scale[marker_pos] = f"{Colors.WHITE}|{Colors.RESET}"
                else:  # Minor markers
                    scale[marker_pos] = "·"

        # Add numerical labels with finer gradations
        scale_labels = ' ' * 5
        for i in range(-30, 31, 5):
            label_pos = center + int((i / self.microscope.max_position) * (scale_width // 2))
            if 0 <= label_pos < scale_width - 2:
                # Add leading spaces for alignment
                label = f"{i:2d}"
                offset = len(label) // 2
                scale_labels = scale_labels[:label_pos-offset] + label + scale_labels[label_pos-offset+len(label):]

        # Mark the center position
        scale[center] = f"{Colors.GREEN}|{Colors.RESET}"

        # Mark all measurement positions with enhanced symbols based on precision
        measurement_scale = scale.copy()
        for pos in sorted_positions:
            reading = self.microscope.measurements[pos]
            precision = reading.get('precision', 'Standard')
            
            # Calculate position on scale with high precision
            pos_exact = center + (pos / self.microscope.max_position) * (scale_width // 2)
            pos_index = int(pos_exact)
            
            if 0 <= pos_index < scale_width:
                # Use different symbols and colors based on position and measurement quality
                if pos < 0:  # Left measurements
                    if precision in ["Ultra-High", "Very High"]:
                        measurement_scale[pos_index] = f"{Colors.BLUE}◆{Colors.RESET}"  # Diamond for highest precision
                    elif precision == "High":
                        measurement_scale[pos_index] = f"{Colors.BLUE}▼{Colors.RESET}"  # Triangle for high precision
                    else:
                        measurement_scale[pos_index] = f"{Colors.BLUE}●{Colors.RESET}"  # Circle for normal precision
                else:  # Right measurements
                    if precision in ["Ultra-High", "Very High"]:
                        measurement_scale[pos_index] = f"{Colors.RED}◆{Colors.RESET}"  # Diamond for highest precision
                    elif precision == "High":
                        measurement_scale[pos_index] = f"{Colors.RED}▼{Colors.RESET}"  # Triangle for high precision
                    else:
                        measurement_scale[pos_index] = f"{Colors.RED}●{Colors.RESET}"  # Circle for normal precision

        print(scale_labels)
        print(''.join(measurement_scale))
        print(f"{Colors.BLUE}◆/▼/●{Colors.RESET} Left measurements   {Colors.RED}◆/▼/●{Colors.RESET} Right measurements")
        print(f"(◆: Ultra/Very-High Precision, ▼: High Precision, ●: Standard Precision)")

        # Display detailed table of measurements with enhanced precision and more fields
        print(f"\n{Colors.BOLD}Detailed Measurements:{Colors.RESET}")
        print("┌────────────────┬─────────────┬────────────────┬────────────────────┬───────────────┬────────────┬────────────┬────────────┐")
        print("│ Position (mm)  │ Closest Ring│ Reading (mm)   │ Distance from Ring │ Precision     │ Quality(%) │ Noise Est. │ Timestamp  │")
        print("├────────────────┼─────────────┼────────────────┼────────────────────┼───────────────┼────────────┼────────────┼────────────┤")

        for pos in sorted_positions:
            reading = self.microscope.measurements[pos]
            position_color = Colors.BLUE if reading['position'] < 0 else Colors.RED
            
            # Choose precision color based on measurement quality
            precision = reading.get('precision', 'Standard')
            if precision in ["Ultra-High", "Very High"]:
                precision_color = Colors.GREEN
            elif precision == "High":
                precision_color = Colors.YELLOW
            else:
                precision_color = Colors.RESET
            
            # Format quality percentage with color
            quality = reading.get('alignment_quality', 0)
            if quality > 99:
                quality_color = Colors.GREEN
            elif quality > 90:
                quality_color = Colors.YELLOW
            else:
                quality_color = Colors.RED
            
            # Format noise estimate with appropriate scaling
            noise = reading.get('noise_estimate', 0.01)
            if noise < 0.001:
                noise_str = f"{noise*1000:.2f}μm"  # Show in μm for tiny values
            else:
                noise_str = f"{noise:.4f}mm"
            
            timestamp = reading.get('timestamp', 'N/A')

            print(f"│ {position_color}{reading['position']:12.6f}{Colors.RESET} │ " +
                  f"{reading['closest_ring']:11d} │ {reading['reading']:12.6f} │ " +
                  f"{reading['distance_from_ring']:16.6f} │ " +
                  f"{precision_color}{precision:13s}{Colors.RESET} │ " +
                  f"{quality_color}{quality:9.2f}%{Colors.RESET} │ " +
                  f"{noise_str:10s} │ {timestamp:10s} │")

        print("└────────────────┴─────────────┴────────────────┴────────────────────┴───────────────┴────────────┴────────────┴────────────┘")

        # Calculate diameter measurements for rings with enhanced precision and visualization
        left_positions = [p for p in sorted_positions if p < 0]
        right_positions = [p for p in sorted_positions if p > 0]

        if left_positions and right_positions:
            print(f"\n{Colors.BOLD}Calculated Ring Diameters (Ultra-Precision Analysis):{Colors.RESET}")

            # Group measurements by closest ring with enhanced organization
            ring_measurements = {}
            for pos in sorted_positions:
                reading = self.microscope.measurements[pos]
                ring = reading['closest_ring']
                precision = reading.get('precision', 'Standard')
                
                if ring not in ring_measurements:
                    ring_measurements[ring] = {'left': [], 'right': [], 'left_precision': {}, 'right_precision': {}}

                if reading['position'] < 0:
                    ring_measurements[ring]['left'].append(reading['position'])
                    # Also track precision of each measurement
                    if precision not in ring_measurements[ring]['left_precision']:
                        ring_measurements[ring]['left_precision'][precision] = 0
                    ring_measurements[ring]['left_precision'][precision] += 1
                else:
                    ring_measurements[ring]['right'].append(reading['position'])
                    # Also track precision of each measurement
                    if precision not in ring_measurements[ring]['right_precision']:
                        ring_measurements[ring]['right_precision'][precision] = 0
                    ring_measurements[ring]['right_precision'][precision] += 1

            if any(ring > 0 for ring in ring_measurements.keys()):
                # Create header with more detailed columns
                print("┌────────┬────────────────┬─────────────────┬────────────────┬────────────────┬───────────────┬────────────────┐")
                print("│ Ring # │ Left Pos (mm)  │ Right Pos (mm)  │ Diameter (mm)  │ Diameter²      │ Calculated λ  │ Precision      │")
                print("├────────┼────────────────┼─────────────────┼────────────────┼────────────────┼───────────────┼────────────────┤")

                for ring in sorted(ring_measurements.keys()):
                    if ring_measurements[ring]['left'] and ring_measurements[ring]['right'] and ring > 0:  # Skip central spot
                        # Find the best measurements for this ring based on precision
                        left_measurements = {}
                        for pos in ring_measurements[ring]['left']:
                            reading = next(r for r in [self.microscope.measurements[p] for p in sorted_positions] 
                                         if r['position'] == pos)
                            precision = reading.get('precision', 'Standard')
                            # Group by precision quality
                            if precision not in left_measurements:
                                left_measurements[precision] = []
                            left_measurements[precision].append(pos)
                        
                        right_measurements = {}
                        for pos in ring_measurements[ring]['right']:
                            reading = next(r for r in [self.microscope.measurements[p] for p in sorted_positions] 
                                         if r['position'] == pos)
                            precision = reading.get('precision', 'Standard')
                            # Group by precision quality
                            if precision not in right_measurements:
                                right_measurements[precision] = []
                            right_measurements[precision].append(pos)
                        
                        # Prefer highest precision measurements if available
                        if "Ultra-High" in left_measurements and "Ultra-High" in right_measurements:
                            left_pos = max(left_measurements["Ultra-High"])  # Rightmost left measurement
                            right_pos = min(right_measurements["Ultra-High"])  # Leftmost right measurement
                            precision_type = "Ultra-High"
                        elif "Very High" in left_measurements and "Very High" in right_measurements:
                            left_pos = max(left_measurements["Very High"])
                            right_pos = min(right_measurements["Very High"])
                            precision_type = "Very High"
                        elif "High" in left_measurements and "High" in right_measurements:
                            left_pos = max(left_measurements["High"])
                            right_pos = min(right_measurements["High"])
                            precision_type = "High"
                        else:
                            # Fall back to the best available positions
                            left_pos = max(ring_measurements[ring]['left'])  # Rightmost left measurement
                            right_pos = min(ring_measurements[ring]['right'])  # Leftmost right measurement
                            precision_type = "Standard"

                        # Calculate diameter and wavelength with ultra-high precision
                        diameter = right_pos - left_pos
                        diameter_squared = diameter ** 2

                        # Calculate wavelength from this measurement with more precision
                        if ring > 0:
                            wavelength_nm = (diameter_squared * self.refractive_index) / (4 * ring * self.radius_of_curvature/10**7)
                            wavelength_nm *= 1_000_000  # Convert to nm
                            
                            # Calculate error percentage
                            error_pct = abs(wavelength_nm - self.wavelength) / self.wavelength * 100
                            
                            # Color code based on error
                            if error_pct < 1.0:
                                wavelength_color = Colors.GREEN
                            elif error_pct < 5.0:
                                wavelength_color = Colors.YELLOW
                            else:
                                wavelength_color = Colors.RED
                                
                            calculated_wavelength = f"{wavelength_color}{wavelength_nm:.4f} nm{Colors.RESET}"
                        else:
                            calculated_wavelength = "N/A"
                        
                        # Color code for precision type
                        if precision_type == "Ultra-High":
                            precision_color = Colors.GREEN
                        elif precision_type == "Very High":
                            precision_color = Colors.GREEN
                        elif precision_type == "High":
                            precision_color = Colors.YELLOW
                        else:
                            precision_color = Colors.RESET
                            
                        # Add summary of all measurements for this ring
                        precision_summary = ""
                        for prec in ["Ultra-High", "Very High", "High", "Good", "Standard"]:
                            left_count = ring_measurements[ring]['left_precision'].get(prec, 0)
                            right_count = ring_measurements[ring]['right_precision'].get(prec, 0)
                            if left_count > 0 or right_count > 0:
                                precision_summary += f"{prec[0]}:{left_count}L/{right_count}R "

                        print(f"│ {ring:6d} │ {left_pos:12.6f} │ {right_pos:13.6f} │ " +
                              f"{diameter:12.6f} │ {diameter_squared:12.6f} │ " +
                              f"{calculated_wavelength:13s} │ " +
                              f"{precision_color}{precision_type:12s}{Colors.RESET} │")

                        # Add to the main measurements for radius calculation
                        self.left_measurements[ring] = left_pos
                        self.right_measurements[ring] = right_pos

                print("└────────┴────────────────┴─────────────────┴────────────────┴────────────────┴───────────────┴────────────────┘")

                # Calculate and display the average wavelength from all measurements with statistical analysis
                if any(ring > 0 for ring in ring_measurements.keys()):
                    wavelengths = []
                    ring_numbers = []
                    diameters = []
                    precisions = []
                    
                    for ring in sorted(ring_measurements.keys()):
                        if ring_measurements[ring]['left'] and ring_measurements[ring]['right'] and ring > 0:
                            # Use the best available measurements as determined above
                            if "Ultra-High" in left_measurements and "Ultra-High" in right_measurements:
                                left_pos = max(left_measurements["Ultra-High"])
                                right_pos = min(right_measurements["Ultra-High"])
                                precision_class = 3  # Numerical value for statistics
                            elif "Very High" in left_measurements and "Very High" in right_measurements:
                                left_pos = max(left_measurements["Very High"])
                                right_pos = min(right_measurements["Very High"])
                                precision_class = 2
                            elif "High" in left_measurements and "High" in right_measurements:
                                left_pos = max(left_measurements["High"])
                                right_pos = min(right_measurements["High"])
                                precision_class = 1
                            else:
                                left_pos = max(ring_measurements[ring]['left'])
                                right_pos = min(ring_measurements[ring]['right'])
                                precision_class = 0
                                
                            diameter = right_pos - left_pos
                            diameters.append(diameter)
                            ring_numbers.append(ring)
                            precisions.append(precision_class)
                            
                            wavelength_nm = (diameter**2 * self.refractive_index) / (4 * ring * self.radius_of_curvature/10**7)
                            wavelength_nm *= 1_000_000  # Convert to nm
                            wavelengths.append(wavelength_nm)

                    if wavelengths:
                        # Calculate weighted average giving more weight to higher precision measurements
                        if sum(precisions) > 0:
                            # Weights from 1 to 4 based on precision_class (0-3)
                            weights = [p + 1 for p in precisions]
                            weighted_avg = sum(w * wl for w, wl in zip(weights, wavelengths)) / sum(weights)
                            
                            # Also calculate standard statistics
                            avg_wavelength = sum(wavelengths) / len(wavelengths)
                            min_wavelength = min(wavelengths)
                            max_wavelength = max(wavelengths)
                            std_dev = (sum((wl - avg_wavelength) ** 2 for wl in wavelengths) / len(wavelengths)) ** 0.5
                            
                            print(f"\n{Colors.BOLD}Statistical Wavelength Analysis:{Colors.RESET}")
                            print(f"Weighted average wavelength: {weighted_avg:.4f} nm (weighted by measurement precision)")
                            print(f"Simple average wavelength: {avg_wavelength:.4f} nm")
                            print(f"Range: {min_wavelength:.4f} - {max_wavelength:.4f} nm (Δ = {max_wavelength-min_wavelength:.4f} nm)")
                            print(f"Standard deviation: {std_dev:.4f} nm (coefficient of variation: {100*std_dev/avg_wavelength:.2f}%)")
                            
                            error_pct = abs(weighted_avg - self.wavelength)/self.wavelength*100
                            if error_pct < 1.0:
                                error_color = Colors.GREEN
                            elif error_pct < 5.0:
                                error_color = Colors.YELLOW
                            else:
                                error_color = Colors.RED
                                
                            print(f"{Colors.BOLD}Actual wavelength:{Colors.RESET} {self.wavelength:.4f} nm")
                            print(f"{Colors.BOLD}Error:{Colors.RESET} {error_color}{error_pct:.4f}%{Colors.RESET}")
                        else:
                            # Fallback if no precision weighting is possible
                            avg_wavelength = sum(wavelengths) / len(wavelengths)
                            print(f"\n{Colors.BOLD}Average calculated wavelength:{Colors.RESET} {avg_wavelength:.4f} nm")
                            print(f"{Colors.BOLD}Actual wavelength:{Colors.RESET} {self.wavelength:.4f} nm")
                            print(f"{Colors.BOLD}Error:{Colors.RESET} {abs(avg_wavelength - self.wavelength)/self.wavelength*100:.4f}%")

                # Visual representation of ring diameters with enhanced color-coded bars
                print(f"\n{Colors.BOLD}Ring Diameter Visualization (Precision Color-Coded):{Colors.RESET}")
                
                # Calculate max diameter for scaling
                max_diameter = 0
                for ring in ring_measurements:
                    if ring_measurements[ring]['left'] and ring_measurements[ring]['right'] and ring > 0:
                        # Find best measurements as above
                        left_pos = max(ring_measurements[ring]['left'])
                        right_pos = min(ring_measurements[ring]['right'])
                        diameter = right_pos - left_pos
                        max_diameter = max(max_diameter, diameter)
                
                # Visualization width
                vis_width = 60
                
                # Define a precision lookup function for cleaner code
                def get_precision_for_ring(ring_data):
                    if "Ultra-High" in ring_data['left_precision'] and "Ultra-High" in ring_data['right_precision']:
                        return "Ultra-High"
                    elif "Very High" in ring_data['left_precision'] and "Very High" in ring_data['right_precision']:
                        return "Very High"
                    elif "High" in ring_data['left_precision'] and "High" in ring_data['right_precision']:
                        return "High"
                    elif "Good" in ring_data['left_precision'] and "Good" in ring_data['right_precision']:
                        return "Good"
                    else:
                        return "Standard"
                
                for ring in sorted(ring_measurements.keys()):
                    if ring_measurements[ring]['left'] and ring_measurements[ring]['right'] and ring > 0:
                        left_pos = max(ring_measurements[ring]['left'])
                        right_pos = min(ring_measurements[ring]['right'])
                        diameter = right_pos - left_pos
                        
                        # Choose color based on precision of measurements
                        precision = get_precision_for_ring(ring_measurements[ring])
                        if precision == "Ultra-High":
                            bar_color = Colors.GREEN
                        elif precision == "Very High":
                            bar_color = Colors.GREEN
                        elif precision == "High":
                            bar_color = Colors.YELLOW
                        elif precision == "Good":
                            bar_color = Colors.YELLOW
                        else:
                            bar_color = Colors.WHITE
                        
                        # Calculate error bars for visualization - show the variation in measurements
                        all_diameters = []
                        for left in ring_measurements[ring]['left']:
                            for right in ring_measurements[ring]['right']:
                                all_diameters.append(right - left)
                        
                        # Create a bar representing the diameter
                        bar_length = int((diameter / max_diameter) * vis_width)
                        bar = f"{bar_color}" + "█" * bar_length + f"{Colors.RESET}"
                        
                        # Add ring number, diameter value, and precision
                        print(f"Ring {ring:2d}: {bar} {diameter:.6f} mm ({precision})")
                        
                        # If multiple measurements exist, show uncertainty
                        if len(all_diameters) > 1:
                            min_diam = min(all_diameters)
                            max_diam = max(all_diameters)
                            range_str = f"Range: {min_diam:.6f} - {max_diam:.6f} mm (Δ = {max_diam-min_diam:.6f} mm)"
                            print(" " * 9 + range_str)

        print("\nPress Enter to return...")
        input()

    def run_simulation(self):
        """Run the main simulation loop"""
        while True:
            self.display_welcome()
            choice = input("Enter your choice (1-9): ")

            if choice == '1':
                self.adjust_microscope_realtime()
            elif choice == '2':
                self.change_wavelength()
            elif choice == '3':
                self.change_radius()
            elif choice == '4':
                self.take_measurements()
            elif choice == '5':
                self.calculate_radius_from_measurements()
            elif choice == '6':
                self.reset_all_measurements()
            elif choice == '7':
                self.display_experiment_diagram()
            elif choice == '8':
                try:
                    print("Generating matplotlib visualization...")
                    self.visualize_with_matplotlib()
                except Exception as e:
                    print(f"Error generating visualization: {e}")
                    print("Make sure matplotlib and numpy are installed.")
                    time.sleep(2)
            elif choice == '9':
                print("Thank you for using the Newton's Rings Simulation!")
                break
            else:
                print("Invalid choice. Please try again.")
                time.sleep(1)

    def visualize_with_matplotlib(self):
        """Create a more detailed visualization using matplotlib"""
        try:
            # Create a figure
            plt.figure(figsize=(10, 8))

            # Create a more direct visualization of Newton's rings
            # This approach creates a more realistic pattern

            # Set up the coordinate grid
            size = 1000
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            X, Y = np.meshgrid(x, y)

            # Calculate the radius from center for each point
            R = np.sqrt(X**2 + Y**2)

            # Create a circular mask
            mask = R <= 1.0

            # Calculate the air gap thickness
            # Scale factor to make rings clearly visible
            # Adjust the scale based on wavelength to get a realistic number of rings
            # For sodium light (589.3 nm), we want about 15-20 rings
            wavelength_normalized = self.wavelength / 589.3  # Normalize to sodium wavelength
            radius_normalized = (self.radius_of_curvature/10**7) / 100.0  # Normalize to 100 cm

            # Adjust scale to show appropriate number of rings
            # More rings for shorter wavelengths and larger radii
            scale = 100 * (1.0 / wavelength_normalized) * (radius_normalized**0.5)
            thickness = scale * R**2

            # Calculate the interference pattern
            # For monochromatic light, intensity varies with cos²(2πt/λ)
            # where t is the thickness and λ is the wavelength
            wavelength_factor = 1.0  # Normalized wavelength
            phase = 2 * np.pi * thickness / wavelength_factor

            # Create the intensity pattern (bright and dark rings)
            # The central spot is dark for Newton's rings
            intensity = np.cos(phase)**2
            intensity = np.where(mask, intensity, 0)

            # Create the RGB image based on the wavelength
            rgb_image = np.zeros((size, size, 3))

            # Set color based on wavelength (approximate)
            if 400 <= self.wavelength < 490:  # Blue
                rgb_image[:,:,2] = intensity  # Blue channel
            elif 490 <= self.wavelength < 570:  # Green
                rgb_image[:,:,1] = intensity  # Green channel
            elif 570 <= self.wavelength < 590:  # Yellow
                rgb_image[:,:,0] = intensity  # Red channel
                rgb_image[:,:,1] = intensity  # Green channel
            elif 590 <= self.wavelength < 620:  # Orange
                rgb_image[:,:,0] = intensity  # Red channel
                rgb_image[:,:,1] = 0.6 * intensity  # Green channel
            else:  # Red
                rgb_image[:,:,0] = intensity  # Red channel

            # Apply a gradient to make it look more realistic
            # Create a radial gradient for lighting effect
            gradient = 1.0 - 0.5 * R**2
            gradient = np.where(mask, gradient, 0)

            # Apply the gradient to the image
            for i in range(3):
                rgb_image[:,:,i] = rgb_image[:,:,i] * gradient

            # Make the central spot dark (phase change on reflection)
            central_spot = R < 0.02
            for i in range(3):
                rgb_image[:,:,i] = np.where(central_spot, 0, rgb_image[:,:,i])

            # Display the image
            plt.imshow(rgb_image)
            plt.title(f"Newton's Rings (λ = {self.wavelength} nm, R = {self.radius_of_curvature/10**7:.1f} cm)")
            plt.axis('off')

            # Save the image
            plt.savefig("newtons_rings.png")

            # Show the plot
            plt.show()

        except Exception as e:
            print(f"Error in visualization: {e}")
            print("Make sure matplotlib and numpy are installed correctly.")
            time.sleep(2)


if __name__ == "__main__":
    # Start the simulation
    app = NewtonRingsSimulation()
