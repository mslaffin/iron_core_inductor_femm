"""
Post-simulation Analysis Tools for Iron Core Inductor
Extracts and analyzes magnetic field results from pyfemm
"""

import femm
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class AnalysisResults:
    """Container for analysis results"""
    inductance: float = 0.0                    # Calculated inductance (H)
    flux_linkage: float = 0.0                  # Total flux linkage (Wb)
    stored_energy: float = 0.0                 # Stored magnetic energy (J)
    max_flux_density: float = 0.0              # Maximum flux density (T)
    avg_flux_density_core: float = 0.0         # Average flux density in core (T)
    core_losses: float = 0.0                   # Core losses (W)
    copper_losses: float = 0.0                 # Copper losses (W)
    field_data: Dict = None                    # Field distribution data
    force_data: Dict = None                    # Force calculation data


class MagneticAnalyzer:
    """Post-processing analysis for magnetic simulations"""
    
    def __init__(self, config, circuit_properties):
        self.config = config
        self.circuit = circuit_properties
        self.results = AnalysisResults()
        
    def calculate_inductance(self) -> float:
        """Calculate inductance from flux linkage and current"""
        try:
            # Get circuit properties
            circuit_info = femm.mo_getcircuitproperties(self.circuit.circuit_name)
            current = circuit_info[0]  # Circuit current
            flux_linkage = circuit_info[2]  # Flux linkage
            
            if abs(current) > 1e-12:  # Avoid division by zero
                inductance = abs(flux_linkage / current)
            else:
                inductance = 0.0
                
            self.results.inductance = inductance
            self.results.flux_linkage = flux_linkage
            
            print(f"Calculated inductance: {inductance*1000:.3f} mH")
            print(f"Flux linkage: {flux_linkage:.6f} Wb")
            
            return inductance
            
        except Exception as e:
            print(f"Error calculating inductance: {e}")
            return 0.0
    
    def calculate_stored_energy(self) -> float:
        """Calculate total stored magnetic energy"""
        try:
            # Integrate energy density over the entire domain
            energy = 0.0
            n_points = 50  # Grid resolution for integration
            
            x_range = np.linspace(-self.config.air_gap_radius*0.9, 
                                self.config.air_gap_radius*0.9, n_points)
            y_range = np.linspace(-self.config.air_gap_radius*0.9, 
                                self.config.air_gap_radius*0.9, n_points)
            
            dx = x_range[1] - x_range[0]
            dy = y_range[1] - y_range[0]
            
            for x in x_range:
                for y in y_range:
                    # Get field values at this point
                    field_data = femm.mo_getpointvalues(x, y)
                    if field_data is not None:
                        b_magnitude = math.sqrt(field_data[1]**2 + field_data[2]**2)
                        mu0 = 4 * math.pi * 1e-7  # Permeability of free space
                        energy_density = (b_magnitude**2) / (2 * mu0)
                        energy += energy_density * dx * dy
            
            self.results.stored_energy = energy
            print(f"Stored magnetic energy: {energy:.6f} J")
            
            return energy
            
        except Exception as e:
            print(f"Error calculating stored energy: {e}")
            return 0.0
    
    def analyze_flux_density_distribution(self) -> Dict:
        """Analyze flux density distribution in the core"""
        try:
            flux_densities = []
            core_flux_densities = []
            n_samples = 100
            
            # Sample points in a grid
            x_range = np.linspace(-self.config.air_gap_radius*0.8, 
                                self.config.air_gap_radius*0.8, n_samples)
            y_range = np.linspace(-self.config.air_gap_radius*0.8, 
                                self.config.air_gap_radius*0.8, n_samples)
            
            field_map = {'x': [], 'y': [], 'Bx': [], 'By': [], 'B_mag': []}
            
            for x in x_range:
                for y in y_range:
                    field_data = femm.mo_getpointvalues(x, y)
                    if field_data is not None:
                        bx, by = field_data[1], field_data[2]
                        b_magnitude = math.sqrt(bx**2 + by**2)
                        
                        field_map['x'].append(x)
                        field_map['y'].append(y)
                        field_map['Bx'].append(bx)
                        field_map['By'].append(by)
                        field_map['B_mag'].append(b_magnitude)
                        
                        flux_densities.append(b_magnitude)
                        
                        # Check if point is in iron core
                        radius = math.sqrt(x**2 + y**2)
                        if (self.config.core_inner_radius < radius < 
                            self.config.core_outer_radius):
                            core_flux_densities.append(b_magnitude)
            
            max_flux = max(flux_densities) if flux_densities else 0.0
            avg_core_flux = np.mean(core_flux_densities) if core_flux_densities else 0.0
            
            self.results.max_flux_density = max_flux
            self.results.avg_flux_density_core = avg_core_flux
            self.results.field_data = field_map
            
            print(f"Maximum flux density: {max_flux:.3f} T")
            print(f"Average core flux density: {avg_core_flux:.3f} T")
            
            return field_map
            
        except Exception as e:
            print(f"Error analyzing flux density: {e}")
            return {}
    
    def calculate_losses(self) -> Tuple[float, float]:
        """Calculate core and copper losses"""
        try:
            copper_losses = 0.0
            core_losses = 0.0
            
            # Get wire positions for copper loss calculation
            from dxf_generator import DXFGenerator
            generator = DXFGenerator(self.config)
            wire_positions = generator.calculate_wire_positions()
            
            # Calculate copper losses (I²R losses)
            wire_area = math.pi * (self.config.wire_diameter / 2)**2  # mm²
            wire_area_m2 = wire_area * 1e-6  # Convert to m²
            copper_resistivity = 1.7e-8  # Ohm⋅m at room temperature
            
            # Estimate wire length (simplified)
            avg_turn_radius = (self.config.winding_inner_radius + 
                             self.config.winding_outer_radius) / 2
            wire_length_per_turn = 2 * math.pi * avg_turn_radius * 1e-3  # Convert to meters
            total_wire_length = wire_length_per_turn * self.config.num_turns
            
            wire_resistance = copper_resistivity * total_wire_length / wire_area_m2
            copper_losses = self.circuit.current_amplitude**2 * wire_resistance
            
            # Core losses (simplified - would need B-H curve for accurate calculation)
            if self.results.avg_flux_density_core > 0:
                # Simplified Steinmetz equation parameters for silicon steel
                k = 0.01  # Material constant
                alpha = 1.5  # Frequency exponent
                beta = 2.0  # Flux density exponent
                
                core_volume = (math.pi * (self.config.core_outer_radius**2 - 
                              self.config.core_inner_radius**2) * 
                              self.config.core_height * 1e-9)  # m³
                
                if self.circuit.frequency > 0:
                    core_losses = (k * (self.circuit.frequency**alpha) * 
                                 (self.results.avg_flux_density_core**beta) * 
                                 core_volume)
                else:
                    core_losses = 0.0  # No hysteresis losses for DC
            
            self.results.copper_losses = copper_losses
            self.results.core_losses = core_losses
            
            print(f"Copper losses: {copper_losses:.6f} W")
            print(f"Core losses: {core_losses:.6f} W")
            
            return core_losses, copper_losses
            
        except Exception as e:
            print(f"Error calculating losses: {e}")
            return 0.0, 0.0
    
    def plot_field_distribution(self, save_plots: bool = True):
        """Create plots of magnetic field distribution"""
        if self.results.field_data is None:
            print("No field data available for plotting")
            return
            
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            x = np.array(self.results.field_data['x'])
            y = np.array(self.results.field_data['y'])
            bx = np.array(self.results.field_data['Bx'])
            by = np.array(self.results.field_data['By'])
            b_mag = np.array(self.results.field_data['B_mag'])
            
            # Reshape for contour plots
            n_points = int(math.sqrt(len(x)))
            X = x.reshape(n_points, n_points)
            Y = y.reshape(n_points, n_points)
            Bx = bx.reshape(n_points, n_points)
            By = by.reshape(n_points, n_points)
            B_mag = b_mag.reshape(n_points, n_points)
            
            # Plot 1: Flux density magnitude
            contour1 = ax1.contourf(X, Y, B_mag, levels=20, cmap='viridis')
            ax1.set_title('Flux Density Magnitude (T)')
            ax1.set_xlabel('X (mm)')
            ax1.set_ylabel('Y (mm)')
            plt.colorbar(contour1, ax=ax1)
            
            # Add core outline
            theta = np.linspace(0, 2*np.pi, 100)
            core_outer_x = self.config.core_outer_radius * np.cos(theta)
            core_outer_y = self.config.core_outer_radius * np.sin(theta)
            core_inner_x = self.config.core_inner_radius * np.cos(theta)
            core_inner_y = self.config.core_inner_radius * np.sin(theta)
            
            ax1.plot(core_outer_x, core_outer_y, 'w-', linewidth=2, label='Core Outer')
            ax1.plot(core_inner_x, core_inner_y, 'w-', linewidth=2, label='Core Inner')
            
            # Plot 2: Flux density vectors
            skip = 5  # Plot every 5th vector for clarity
            ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                      Bx[::skip, ::skip], By[::skip, ::skip],
                      B_mag[::skip, ::skip], cmap='plasma')
            ax2.set_title('Flux Density Vectors')
            ax2.set_xlabel('X (mm)')
            ax2.set_ylabel('Y (mm)')
            ax2.plot(core_outer_x, core_outer_y, 'k-', linewidth=1)
            ax2.plot(core_inner_x, core_inner_y, 'k-', linewidth=1)
            
            # Plot 3: Radial flux density distribution
            center_idx = n_points // 2
            radial_distances = X[center_idx, center_idx:]
            radial_flux = B_mag[center_idx, center_idx:]
            
            ax3.plot(radial_distances, radial_flux, 'b-', linewidth=2)
            ax3.axvline(self.config.core_inner_radius, color='r', linestyle='--', 
                       label='Core Inner')
            ax3.axvline(self.config.core_outer_radius, color='r', linestyle='--', 
                       label='Core Outer')
            ax3.set_title('Radial Flux Density Distribution')
            ax3.set_xlabel('Radius (mm)')
            ax3.set_ylabel('Flux Density (T)')
            ax3.legend()
            ax3.grid(True)
            
            # Plot 4: Energy density distribution
            mu0 = 4 * math.pi * 1e-7
            energy_density = B_mag**2 / (2 * mu0)
            contour4 = ax4.contourf(X, Y, energy_density, levels=20, cmap='hot')
            ax4.set_title('Energy Density (J/m³)')
            ax4.set_xlabel('X (mm)')
            ax4.set_ylabel('Y (mm)')
            plt.colorbar(contour4, ax=ax4)
            ax4.plot(core_outer_x, core_outer_y, 'w-', linewidth=1)
            ax4.plot(core_inner_x, core_inner_y, 'w-', linewidth=1)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('inductor_field_analysis.png', dpi=300, bbox_inches='tight')
                print("Field analysis plots saved as 'inductor_field_analysis.png'")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = f"""
=== Iron Core Inductor Analysis Report ===

Geometry Configuration:
- Core inner radius: {self.config.core_inner_radius} mm
- Core outer radius: {self.config.core_outer_radius} mm
- Wire diameter: {self.config.wire_diameter} mm
- Number of turns: {self.config.num_turns}
- Excitation current: {self.circuit.current_amplitude} A

Electrical Properties:
- Inductance: {self.results.inductance*1000:.3f} mH
- Flux linkage: {self.results.flux_linkage:.6f} Wb
- Stored energy: {self.results.stored_energy:.6f} J

Magnetic Field Analysis:
- Maximum flux density: {self.results.max_flux_density:.3f} T
- Average core flux density: {self.results.avg_flux_density_core:.3f} T

Loss Analysis:
- Copper losses: {self.results.copper_losses:.6f} W
- Core losses: {self.results.core_losses:.6f} W
- Total losses: {self.results.copper_losses + self.results.core_losses:.6f} W

Performance Metrics:
- Inductance per turn²: {self.results.inductance/(self.config.num_turns**2)*1e6:.2f} µH/turn²
- Energy factor: {self.results.stored_energy/(self.circuit.current_amplitude**2):.6f} J/A²
"""
        return report
    
    def run_complete_analysis(self) -> AnalysisResults:
        """Run all analysis routines"""
        print("Starting complete magnetic analysis...")
        
        self.calculate_inductance()
        self.calculate_stored_energy()
        self.analyze_flux_density_distribution()
        self.calculate_losses()
        
        # Generate plots
        self.plot_field_distribution()
        
        # Print report
        report = self.generate_analysis_report()
        print(report)
        
        # Save report to file
        with open('analysis_report.txt', 'w') as f:
            f.write(report)
        print("Analysis report saved to 'analysis_report.txt'")
        
        return self.results


if __name__ == "__main__":
    # Example usage (requires active FEMM solution)
    from femm_simulation import create_default_simulation
    from dxf_generator import InductorConfig
    
    # This would typically be called after running the simulation
    config = InductorConfig()
    sim = create_default_simulation()
    
    analyzer = MagneticAnalyzer(config, sim.circuit)
    
    print("Analysis tools ready. Run after FEMM simulation is complete.")
    # results = analyzer.run_complete_analysis()