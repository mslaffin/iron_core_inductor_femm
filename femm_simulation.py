"""
PyFEMM Simulation Setup for Iron Core Inductor
Handles DXF import, material properties, circuit setup, and simulation
"""

import femm
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from dxf_generator import InductorConfig


@dataclass
class MaterialProperties:
    """Material property definitions for simulation"""
    # Iron core material (typical silicon steel)
    iron_mu_r: float = 4000.0          # Relative permeability
    iron_h_c: float = 0.0              # Coercivity (A/m)
    iron_j: float = 0.0                # Applied current density (MA/m²)
    iron_conductivity: float = 1.6e6   # Conductivity (S/m)
    
    # Copper wire material
    copper_mu_r: float = 1.0           # Non-magnetic
    copper_h_c: float = 0.0
    copper_j: float = 0.0
    copper_conductivity: float = 58e6  # High conductivity
    
    # Air material
    air_mu_r: float = 1.0
    air_h_c: float = 0.0
    air_j: float = 0.0
    air_conductivity: float = 0.0


@dataclass
class CircuitProperties:
    """Circuit configuration for the inductor"""
    current_amplitude: float = 1.0     # Excitation current (A)
    frequency: float = 0.0             # Frequency (Hz) - 0 for magnetostatic
    circuit_name: str = "Coil"
    turns: int = 1                     # Total turns in series


class FEMMSimulation:
    """Main simulation class for iron core inductor"""
    
    def __init__(self, config: InductorConfig, materials: MaterialProperties,
                 circuit: CircuitProperties, fem_filename: str = "inductor.fem"):
        self.config = config
        self.materials = materials
        self.circuit = circuit
        self.simulation_open = False
        self.fem_filename = fem_filename
        
    def initialize_femm(self):
        """Initialize FEMM environment"""
        femm.openfemm(1)  # Open FEMM with window minimized
        self.simulation_open = True
        print("FEMM initialized")
        
    def setup_problem(self):
        """Setup magnetostatic problem"""
        femm.newdocument(0)  # New magnetostatic document
        
        # Problem definition
        femm.mi_probdef(
            self.circuit.frequency,  # Frequency
            'millimeters',           # Length units
            'planar',               # Problem type (planar/axisymmetric)
            1e-8,                   # Precision
            0,                      # Depth (for planar problems)
            30,                      # Min angle for meshing
            'axi'                   # axisymmetric coordinates
        )
        
        print("Problem setup completed")
        
    def define_materials(self):
        """Define all materials used in simulation"""
        # Define iron core material
        femm.mi_addmaterial(
            'Iron_Core',
            self.materials.iron_mu_r,
            self.materials.iron_mu_r,  # mu_x and mu_y (isotropic)
            self.materials.iron_h_c,
            self.materials.iron_j,
            self.materials.iron_conductivity,
            0,  # Hysteresis lag angle
            0,  # Lamination thickness
            1,  # Lamination fill factor
            0,  # Wire type (0=solid)
            0   # Stranding type
        )
        
        # Define copper wire material
        femm.mi_addmaterial(
            'Copper',
            self.materials.copper_mu_r,
            self.materials.copper_mu_r,
            self.materials.copper_h_c,
            self.materials.copper_j,
            self.materials.copper_conductivity,
            0, 0, 1, 0, 0
        )
        
        # Define air material
        femm.mi_addmaterial(
            'Air',
            self.materials.air_mu_r,
            self.materials.air_mu_r,
            self.materials.air_h_c,
            self.materials.air_j,
            self.materials.air_conductivity,
            0, 0, 1, 0, 0
        )
        
        print("Materials defined")
        
    def define_circuit(self):
        """Define circuit properties"""
        femm.mi_addcircprop(
            self.circuit.circuit_name,
            self.circuit.current_amplitude,
            1  # Circuit type (1=current, 0=voltage)
        )
        
        print(f"Circuit '{self.circuit.circuit_name}' defined with {self.circuit.current_amplitude}A")
        
    def import_geometry_from_dxf(self, dxf_filename: str):
        """Import geometry from DXF file"""
        if not os.path.exists(dxf_filename):
            raise FileNotFoundError(f"DXF file not found: {dxf_filename}")
            
        femm.mi_readdxf(dxf_filename)
        print(f"Geometry imported from {dxf_filename}")
        
    def create_geometry_programmatically(self):
        """Create axisymmetric solenoid geometry directly in FEMM"""
        half_height_air = self.config.air_gap_radius
        half_height_core = self.config.core_height / 2

        # Air boundary rectangle (r=0 to air_gap_radius, z=-half_height to +half_height)
        femm.mi_addnode(0, -half_height_air)
        femm.mi_addnode(self.config.air_gap_radius, -half_height_air)
        femm.mi_addnode(self.config.air_gap_radius, half_height_air)
        femm.mi_addnode(0, half_height_air)

        femm.mi_addsegment(0, -half_height_air, self.config.air_gap_radius, -half_height_air)
        femm.mi_addsegment(self.config.air_gap_radius, -half_height_air, self.config.air_gap_radius, half_height_air)
        femm.mi_addsegment(self.config.air_gap_radius, half_height_air, 0, half_height_air)
        femm.mi_addsegment(0, half_height_air, 0, -half_height_air)

        # Iron core rectangle (solid rod from r=0 to core_outer_radius)
        if self.config.core_inner_radius > 0:
            # Hollow core
            femm.mi_addnode(self.config.core_inner_radius, -half_height_core)
            femm.mi_addnode(self.config.core_outer_radius, -half_height_core)
            femm.mi_addnode(self.config.core_outer_radius, half_height_core)
            femm.mi_addnode(self.config.core_inner_radius, half_height_core)

            femm.mi_addsegment(self.config.core_inner_radius, -half_height_core, self.config.core_outer_radius, -half_height_core)
            femm.mi_addsegment(self.config.core_outer_radius, -half_height_core, self.config.core_outer_radius, half_height_core)
            femm.mi_addsegment(self.config.core_outer_radius, half_height_core, self.config.core_inner_radius, half_height_core)
            femm.mi_addsegment(self.config.core_inner_radius, half_height_core, self.config.core_inner_radius, -half_height_core)
        else:
            # Solid core (rod from r=0)
            femm.mi_addnode(0, -half_height_core)
            femm.mi_addnode(self.config.core_outer_radius, -half_height_core)
            femm.mi_addnode(self.config.core_outer_radius, half_height_core)
            femm.mi_addnode(0, half_height_core)

            femm.mi_addsegment(0, -half_height_core, self.config.core_outer_radius, -half_height_core)
            femm.mi_addsegment(self.config.core_outer_radius, -half_height_core, self.config.core_outer_radius, half_height_core)
            femm.mi_addsegment(self.config.core_outer_radius, half_height_core, 0, half_height_core)
            femm.mi_addsegment(0, half_height_core, 0, -half_height_core)

        # Add individual wire circles
        from dxf_generator import DXFGenerator
        generator = DXFGenerator(self.config)
        wire_positions = generator.calculate_wire_positions()

        for r, z in wire_positions:
            femm.mi_drawcircle(r, z, self.config.wire_diameter / 2)

        print(f"Axisymmetric geometry created: {len(wire_positions)} wires")
        
    def assign_block_properties(self):
        """Assign material properties to regions"""
        print("Assigning block properties...")

        # Assign iron to core region FIRST (most important)
        if self.config.core_inner_radius > 0:
            # Hollow core - assign iron to annular region
            core_radius = (self.config.core_outer_radius + self.config.core_inner_radius) / 2
            core_z = 0  # Center of core
            print(f"  Adding iron core label at r={core_radius}, z={core_z} (hollow)")
            femm.mi_addblocklabel(core_radius, core_z)
            femm.mi_selectlabel(core_radius, core_z)
            femm.mi_setblockprop('Iron_Core', 1, 0, '', 0, 0, 0)
            femm.mi_clearselected()

            # Assign air to inner core region (this is a separate region)
            inner_air_radius = self.config.core_inner_radius * 0.5
            print(f"  Adding inner air label at r={inner_air_radius}, z=0")
            femm.mi_addblocklabel(inner_air_radius, 0)
            femm.mi_selectlabel(inner_air_radius, 0)
            femm.mi_setblockprop('Air', 1, 0, '', 0, 0, 0)
            femm.mi_clearselected()
        else:
            # Solid core - assign iron from axis to outer radius
            # Place label well inside the core region
            core_radius = self.config.core_outer_radius * 0.5
            core_z = 0  # Center of core
            print(f"  Adding iron core label at r={core_radius}, z={core_z} (solid)")
            femm.mi_addblocklabel(core_radius, core_z)
            femm.mi_selectlabel(core_radius, core_z)
            femm.mi_setblockprop('Iron_Core', 1, 0, '', 0, 0, 0)
            femm.mi_clearselected()

        # Assign ONE air label for all air regions outside the core
        # This includes: above/below core, between core and wires, and outside windings
        # Place it in the outer region where it's guaranteed to be in air
        air_region_radius = (self.config.winding_outer_radius + self.config.air_gap_radius) / 2
        air_region_z = 0
        print(f"  Adding single air label at r={air_region_radius}, z={air_region_z}")
        femm.mi_addblocklabel(air_region_radius, air_region_z)
        femm.mi_selectlabel(air_region_radius, air_region_z)
        femm.mi_setblockprop('Air', 1, 0, '', 0, 0, 0)
        femm.mi_clearselected()

        # Assign copper and circuit properties to each individual wire
        from dxf_generator import DXFGenerator
        generator = DXFGenerator(self.config)
        wire_positions = generator.calculate_wire_positions()

        turns_per_wire = self.circuit.turns / len(wire_positions)

        for i, (r, z) in enumerate(wire_positions):
            femm.mi_addblocklabel(r, z)
            femm.mi_selectlabel(r, z)
            # All wires carry current in same direction for solenoid
            femm.mi_setblockprop('Copper', 1, 0, self.circuit.circuit_name,
                               0, 0, turns_per_wire)
            femm.mi_clearselected()

        print(f"Block properties assigned: {len(wire_positions)} wires configured")
        
    def apply_boundary_conditions(self):
        """Apply boundary conditions"""
        # For axisymmetric problems, we need to apply boundary at r=0 and outer boundary
        # Add boundary property for zero potential
        femm.mi_addboundprop('A0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Apply to outer boundary segments
        # Select and apply boundary to the rightmost vertical segment
        femm.mi_selectsegment(self.config.air_gap_radius, 0)

        femm.mi_setsegmentprop('A0', 0, 1, 0, 0)
        femm.mi_clearselected()
        
        print("Boundary conditions applied")


    def save_model(self):
        try:
            femm.mi_saveas(self.fem_filename)
            print(f"Model saved as {self.fem_filename}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
        
    def create_mesh_and_solve(self):
        """Generate mesh and solve the problem"""
        
        if not self.save_model():
            raise RuntimeError("Failed to save model before analysis")
        
        femm.mi_createmesh()
        print("Mesh generated")
        
        femm.mi_analyze(1)  # Solve with FEMM visible
        print("Analysis completed")

        time.sleep(1)

        ans_file = self.fem_filename.replace('.fem', '.ans')
        if not os.path.exists(ans_file):
            raise RuntimeError(f"Analysis failed - no solution file created: {ans_file}")

        
        
        try:
            femm.mi_loadsolution()
            print("Solution loaded")
        except Exception as e:
            raise RuntimeError(f"failed to load solution")

    def run_full_simulation(self, dxf_filename: Optional[str] = None):
        """Run complete simulation workflow"""
        try:
            self.initialize_femm()
            self.setup_problem()
            self.define_materials()
            self.define_circuit()

            if dxf_filename and os.path.exists(dxf_filename):
                self.import_geometry_from_dxf(dxf_filename)
            else:
                print("Warning: No DXF file provided, creating geometry programmatically")
                self.create_geometry_programmatically()

            self.assign_block_properties()
            self.apply_boundary_conditions()
            self.create_mesh_and_solve()

            print("Simulation completed successfully")
            return True

        except Exception as e:
            print(f"Simulation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def cleanup(self):
        """Clean up FEMM resources"""
        if self.simulation_open:
            femm.closefemm()
            self.simulation_open = False
            print("FEMM closed")


def create_default_simulation() -> FEMMSimulation:
    """Create a simulation with default parameters"""
    config = InductorConfig(
        core_outer_radius=8.5,
        core_inner_radius=0.0,
        core_height=30.0,
        wire_diameter=1.29,
        num_turns=160,
        wire_layers=9,
        winding_inner_radius=8.55,
        winding_outer_radius=16.0,
        winding_height=29.0,
        air_gap_radius=50.0
    )

    materials = MaterialProperties()

    circuit = CircuitProperties(
        current_amplitude=1.0,
        circuit_name="MainCoil",
        turns=160
    )

    return FEMMSimulation(config, materials, circuit)


if __name__ == "__main__":
    # Create and run simulation
    sim = create_default_simulation()
    
    try:
        # Run simulation with DXF import
        success = sim.run_full_simulation("iron_core_inductor.dxf")
        
        if success:
            print("Ready for post-processing analysis")
        else:
            print("Simulation failed")
            
    finally:
        sim.cleanup()