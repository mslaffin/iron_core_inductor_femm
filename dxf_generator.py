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
        """Calculate optimal wire positions based on winding volume and turns"""
        positions = []
        
        # Calculate available winding area
        radial_space = self.config.winding_outer_radius - self.config.winding_inner_radius
        axial_space = self.config.winding_height
        
        # Calculate how many wires fit radially and axially
        wires_per_layer_radial = int(radial_space / self.config.wire_diameter)
        wires_per_layer_axial = int(axial_space / self.config.wire_diameter)
        
        total_wires_per_layer = wires_per_layer_radial * wires_per_layer_axial
        
        if total_wires_per_layer * self.config.wire_layers < self.config.num_turns:
            print(f"Warning: Only {total_wires_per_layer * self.config.wire_layers} wires fit, "
                  f"but {self.config.num_turns} turns requested")
        
        # Generate positions for each layer
        wires_placed = 0
        for layer in range(self.config.wire_layers):
            if wires_placed >= self.config.num_turns:
                break
                
            # Calculate radial positions for this layer
            radial_step = radial_space / max(1, wires_per_layer_radial - 1) if wires_per_layer_radial > 1 else 0
            axial_step = axial_space / max(1, wires_per_layer_axial - 1) if wires_per_layer_axial > 1 else 0
            
            for r_idx in range(wires_per_layer_radial):
                for a_idx in range(wires_per_layer_axial):
                    if wires_placed >= self.config.num_turns:
                        break
                        
                    # Calculate position
                    radius = self.config.winding_inner_radius + r_idx * radial_step
                    z_pos = -self.config.winding_height/2 + a_idx * axial_step
                    
                    # For 2D simulation, we use radius and z as x and y coordinates
                    positions.append((radius, z_pos))
                    wires_placed += 1
                    
                if wires_placed >= self.config.num_turns:
                    break
        
        return positions[:self.config.num_turns]
    
    def add_iron_core(self):
        """Add iron core geometry to DXF"""
        # Create outer circle of iron core
        self.msp.add_circle(
            center=(0, 0),
            radius=self.config.core_outer_radius,
            dxfattribs={'layer': 'IRON_CORE_OUTER'}
        )
        
        # Create inner circle of iron core (air gap)
        self.msp.add_circle(
            center=(0, 0),
            radius=self.config.core_inner_radius,
            dxfattribs={'layer': 'IRON_CORE_INNER'}
        )
        
    def add_wire_cross_sections(self):
        """Add wire cross-sections to DXF"""
        positions = self.calculate_wire_positions()
        
        for i, (x, y) in enumerate(positions):
            self.msp.add_circle(
                center=(x, y),
                radius=self.config.wire_diameter / 2,
                dxfattribs={'layer': f'WIRE_{i+1}'}
            )
    
    def add_air_boundary(self):
        """Add outer air boundary for simulation domain"""
        self.msp.add_circle(
            center=(0, 0),
            radius=self.config.air_gap_radius,
            dxfattribs={'layer': 'AIR_BOUNDARY'}
        )
    
    def add_construction_lines(self):
        """Add construction lines for reference"""
        # Add centerlines
        self.msp.add_line(
            start=(-self.config.air_gap_radius, 0),
            end=(self.config.air_gap_radius, 0),
            dxfattribs={'layer': 'CONSTRUCTION', 'linetype': 'DASHED'}
        )
        self.msp.add_line(
            start=(0, -self.config.air_gap_radius),
            end=(0, self.config.air_gap_radius),
            dxfattribs={'layer': 'CONSTRUCTION', 'linetype': 'DASHED'}
        )
        
        # Add winding region boundaries
        self.msp.add_circle(
            center=(0, 0),
            radius=self.config.winding_inner_radius,
            dxfattribs={'layer': 'CONSTRUCTION', 'linetype': 'DASHED'}
        )
        self.msp.add_circle(
            center=(0, 0),
            radius=self.config.winding_outer_radius,
            dxfattribs={'layer': 'CONSTRUCTION', 'linetype': 'DASHED'}
        )
    
    def create_layers(self):
        """Create and configure DXF layers"""
        layers = [
            ('IRON_CORE_OUTER', 1),    # Red
            ('IRON_CORE_INNER', 2),    # Yellow
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
        wire_positions = self.calculate_wire_positions()
        print(f"Generated inductor with:")
        print(f"  - Core: {self.config.core_inner_radius}-{self.config.core_outer_radius}mm radius")
        print(f"  - {len(wire_positions)} wire cross-sections")
        print(f"  - Wire diameter: {self.config.wire_diameter}mm")
        print(f"  - Simulation domain: {self.config.air_gap_radius}mm radius")


def create_sample_inductor():
    """Create a sample inductor DXF file"""
    config = InductorConfig(
        core_outer_radius=20.0,
        core_inner_radius=8.0,
        core_height=30.0,
        wire_diameter=0.8,
        num_turns=40,
        wire_layers=1,
        winding_inner_radius=8.5,
        winding_outer_radius=19.5,
        winding_height=25.0,
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