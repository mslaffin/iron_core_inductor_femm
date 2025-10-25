"""
DXF Generator for Iron Core Inductor
Generates configurable DXF files for pyfemm simulation
"""

import ezdxf
import math
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class InductorConfig:
    """Configuration parameters for inductor geometry"""
    # Iron core dimensions (mm)
    core_outer_radius: float = 20.0
    core_inner_radius: float = 10.0
    core_height: float = 30.0
    
    # Wire specifications
    wire_diameter: float = 1.0
    num_turns: int = 50
    wire_layers: int = 5
    
    # Winding volume definition
    winding_inner_radius: float = 10.5  # Slightly larger than core inner radius
    winding_outer_radius: float = 19.5  # Slightly smaller than core outer radius
    winding_height: float = 25.0       # Axial height for windings
    
    # Simulation domain
    air_gap_radius: float = 50.0       # Outer boundary for air region


class DXFGenerator:
    """Generates DXF geometry for iron core inductor"""
    
    def __init__(self, config: InductorConfig):
        self.config = config
        self.doc = ezdxf.new('R2010')
        self.msp = self.doc.modelspace()
        
    def calculate_wire_positions(self) -> List[Tuple[float, float]]:
        """Calculate wire positions with hexagonal close packing"""
        positions = []

        # Hexagonal packing parameters
        # Axial (vertical) spacing between rows
        axial_pitch = self.config.wire_diameter
        # Radial (horizontal) spacing between layers
        radial_pitch = self.config.wire_diameter * 0.866  # sqrt(3)/2 for hex packing
        # Vertical offset for alternating layers (half wire diameter)
        hex_offset = self.config.wire_diameter / 2

        # Calculate available winding area
        radial_space = self.config.winding_outer_radius - self.config.winding_inner_radius
        axial_space = self.config.winding_height

        # Calculate how many wires fit
        wires_per_layer_radial = int(radial_space / radial_pitch)
        wires_per_layer_axial = int(axial_space / axial_pitch)

        total_wires_per_layer = wires_per_layer_radial * wires_per_layer_axial

        if total_wires_per_layer * self.config.wire_layers < self.config.num_turns:
            print(f"Warning: Only {total_wires_per_layer * self.config.wire_layers} wires fit, "
                  f"but {self.config.num_turns} turns requested")

        # Generate positions for each radial layer
        wires_placed = 0
        for r_idx in range(wires_per_layer_radial):
            if wires_placed >= self.config.num_turns:
                break

            # Calculate radial position
            radius = self.config.winding_inner_radius + self.config.wire_diameter / 2 + r_idx * radial_pitch

            # Alternate layers get vertical offset for hexagonal packing
            z_offset = hex_offset if r_idx % 2 == 1 else 0

            # Place wires vertically in this radial layer
            for a_idx in range(wires_per_layer_axial):
                if wires_placed >= self.config.num_turns:
                    break

                # Calculate vertical position with hex offset
                z_pos = -self.config.winding_height / 2 + self.config.wire_diameter / 2 + a_idx * axial_pitch + z_offset

                # Check if still within bounds
                if abs(z_pos) <= self.config.winding_height / 2:
                    positions.append((radius, z_pos))
                    wires_placed += 1

        return positions[:self.config.num_turns]
    
    def add_iron_core(self):
        """Add iron core geometry to DXF (axisymmetric cross-section)"""
        # r-axis is horizontal (x in DXF), z-axis is vertical (y in DXF)

        # Core rectangle from inner radius to outer radius, with specified height
        half_height = self.config.core_height / 2

        # Create closed polyline for iron core
        points = [
            (self.config.core_inner_radius, -half_height),  # bottom-left
            (self.config.core_outer_radius, -half_height),  # bottom-right
            (self.config.core_outer_radius, half_height),   # top-right
            (self.config.core_inner_radius, half_height),   # top-left
            (self.config.core_inner_radius, -half_height),  # close the loop
        ]

        self.msp.add_lwpolyline(
            points,
            dxfattribs={'layer': 'IRON_CORE'}
        )
        
    def add_wire_cross_sections(self):
        """Add individual wire cross-sections to axisymmetric DXF"""
        # Calculate wire positions in the r-z plane
        positions = self.calculate_wire_positions()

        # In axisymmetric view, wires appear as circles at their r,z positions
        for i, (r, z) in enumerate(positions):
            self.msp.add_circle(
                center=(r, z),
                radius=self.config.wire_diameter / 2,
                dxfattribs={'layer': f'WIRE_{i+1}'}
            )
    
    def add_air_boundary(self):
        """Add outer air boundary for simulation domain (axisymmetric)"""
        # For axisymmetric simulation, create a rectangular boundary
        # The left edge is at r=0 (axis of symmetry)
        half_height = self.config.air_gap_radius

        points = [
            (0, -half_height),
            (self.config.air_gap_radius, -half_height),
            (self.config.air_gap_radius, half_height),
            (0, half_height),
            (0, -half_height),
        ]

        self.msp.add_lwpolyline(
            points,
            dxfattribs={'layer': 'AIR_BOUNDARY'}
        )
    
    def add_construction_lines(self):
        """Add construction lines for reference (axisymmetric view)"""
        # Add axis of symmetry (r=0)
        self.msp.add_line(
            start=(0, -self.config.air_gap_radius),
            end=(0, self.config.air_gap_radius),
            dxfattribs={'layer': 'CONSTRUCTION', 'linetype': 'DASHED'}
        )

        # Add horizontal centerline (z=0)
        self.msp.add_line(
            start=(0, 0),
            end=(self.config.air_gap_radius, 0),
            dxfattribs={'layer': 'CONSTRUCTION', 'linetype': 'DASHED'}
        )
    
    def create_layers(self):
        """Create and configure DXF layers"""
        layers = [
            ('IRON_CORE', 1),          # Red
            ('AIR_BOUNDARY', 3),       # Green
            ('CONSTRUCTION', 8),       # Dark gray
        ]

        # Add wire layers
        for i in range(self.config.num_turns):
            layers.append((f'WIRE_{i+1}', 4))  # Cyan for all wires

        for layer_name, color in layers:
            layer = self.doc.layers.new(name=layer_name)
            layer.color = color
    
    def generate_dxf(self, filename: str = "inductor.dxf"):
        """Generate complete DXF file"""
        self.create_layers()
        self.add_air_boundary()
        self.add_iron_core()
        self.add_wire_cross_sections()
        self.add_construction_lines()

        self.doc.saveas(filename)
        print(f"DXF file saved as: {filename}")

        # Print summary
        print(f"Generated axisymmetric inductor geometry:")
        print(f"  - Core: r={self.config.core_inner_radius}-{self.config.core_outer_radius}mm, h={self.config.core_height}mm")
        print(f"  - Winding: r={self.config.winding_inner_radius}-{self.config.winding_outer_radius}mm, h={self.config.winding_height}mm")
        print(f"  - Turns: {self.config.num_turns}")
        print(f"  - Simulation domain: {self.config.air_gap_radius}mm")


def create_sample_inductor():
    """Create a sample inductor DXF file"""
    # For axisymmetric solenoid (iron rod with wire wrapped around it):
    # - Core is a solid cylinder at the center (r=0 to core_outer_radius)
    # - Windings wrap around the outside of the core
    config = InductorConfig(
        core_outer_radius=8.5,      # Iron rod radius
        core_inner_radius=0.0,       # Solid rod (no hole)
        core_height=30.0,            # Rod length
        wire_diameter=1.29,
        num_turns=160,
        wire_layers=9,
        winding_inner_radius=8.55,      # Just outside the core
        winding_outer_radius=16.0,    # Outer edge of windings
        winding_height=29.0,
        air_gap_radius=50.0
    )

    generator = DXFGenerator(config)
    generator.generate_dxf("iron_core_inductor.dxf")
    return config


if __name__ == "__main__":
    # Create sample inductor
    config = create_sample_inductor()
    
    # Also create a smaller test inductor
    test_config = InductorConfig(
        core_outer_radius=10.0,
        core_inner_radius=4.0,
        wire_diameter=0.5,
        num_turns=20,
        winding_inner_radius=4.5,
        winding_outer_radius=9.5,
        winding_height=15.0,
        air_gap_radius=25.0
    )
    
    test_generator = DXFGenerator(test_config)
    test_generator.generate_dxf("test_inductor.dxf")