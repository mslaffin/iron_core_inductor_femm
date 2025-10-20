"""
Complete Iron Core Inductor Simulation Workflow
Main automation script that integrates DXF generation, FEMM simulation, and analysis
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any
import json

from dxf_generator import InductorConfig, DXFGenerator
from femm_simulation import FEMMSimulation, MaterialProperties, CircuitProperties
from analysis_tools import MagneticAnalyzer, AnalysisResults


class InductorWorkflow:
    """Complete workflow manager for inductor simulation"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = None
        self.materials = None
        self.circuit = None
        self.simulation = None
        self.analyzer = None
        self.results = None
        
        if config_file and os.path.exists(config_file):
            self.load_configuration(config_file)
        else:
            self.create_default_configuration()
    
    def create_default_configuration(self):
        """Create default configuration for all components"""
        self.config = InductorConfig(
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
        
        self.materials = MaterialProperties(
            iron_mu_r=4000.0,
            iron_h_c=0.0,
            iron_j=0.0,
            iron_conductivity=1.6e6,
            copper_mu_r=1.0,
            copper_h_c=0.0,
            copper_j=0.0,
            copper_conductivity=58e6,
            air_mu_r=1.0,
            air_h_c=0.0,
            air_j=0.0,
            air_conductivity=0.0
        )
        
        self.circuit = CircuitProperties(
            current_amplitude=1.0,
            frequency=0.0,
            circuit_name="MainCoil",
            turns=40
        )
    
    def load_configuration(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            # Load inductor configuration
            if 'geometry' in data:
                self.config = InductorConfig(**data['geometry'])
            else:
                self.config = InductorConfig()
                
            # Load material properties
            if 'materials' in data:
                self.materials = MaterialProperties(**data['materials'])
            else:
                self.materials = MaterialProperties()
                
            # Load circuit properties
            if 'circuit' in data:
                self.circuit = CircuitProperties(**data['circuit'])
            else:
                self.circuit = CircuitProperties()
                
            print(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration instead")
            self.create_default_configuration()
    
    def save_configuration(self, config_file: str):
        """Save current configuration to JSON file"""
        try:
            data = {
                'geometry': self.config.__dict__,
                'materials': self.materials.__dict__,
                'circuit': self.circuit.__dict__
            }
            
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Configuration saved to {config_file}")
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def generate_geometry(self, output_file: str = "inductor.dxf") -> bool:
        """Generate DXF geometry file"""
        try:
            print("=== Generating DXF Geometry ===")
            generator = DXFGenerator(self.config)
            generator.generate_dxf(output_file)
            return True
            
        except Exception as e:
            print(f"Error generating geometry: {e}")
            return False
    
    def run_simulation(self, dxf_file: Optional[str] = None) -> bool:
        """Run FEMM simulation"""
        try:
            print("=== Running FEMM Simulation ===")
            self.simulation = FEMMSimulation(self.config, self.materials, self.circuit)
            success = self.simulation.run_full_simulation(dxf_file)
            return success
            
        except Exception as e:
            print(f"Error running simulation: {e}")
            return False
    
    def analyze_results(self) -> bool:
        """Perform post-simulation analysis"""
        try:
            print("=== Analyzing Results ===")
            self.analyzer = MagneticAnalyzer(self.config, self.circuit)
            self.results = self.analyzer.run_complete_analysis()
            return True
            
        except Exception as e:
            print(f"Error analyzing results: {e}")
            return False
    
    def cleanup(self):
        """Clean up simulation resources"""
        if self.simulation:
            self.simulation.cleanup()
    
    def run_complete_workflow(self, generate_dxf: bool = True, 
                             dxf_filename: str = "inductor.dxf") -> bool:
        """Run the complete simulation workflow"""
        success = True
        
        try:
            # Step 1: Generate DXF geometry
            if generate_dxf:
                if not self.generate_geometry(dxf_filename):
                    print("Failed to generate DXF geometry")
                    return False
            
            # Step 2: Run FEMM simulation
            dxf_file = dxf_filename if generate_dxf else None
            if not self.run_simulation(dxf_file):
                print("Failed to run simulation")
                return False
            
            # Step 3: Analyze results
            if not self.analyze_results():
                print("Failed to analyze results")
                success = False
            
            print("=== Workflow Complete ===")
            if self.results:
                print(f"Final inductance: {self.results.inductance*1000:.3f} mH")
                print(f"Total losses: {self.results.copper_losses + self.results.core_losses:.6f} W")
            
            return success
            
        except Exception as e:
            print(f"Workflow error: {e}")
            return False
        
        finally:
            self.cleanup()
    
    def print_configuration_summary(self):
        """Print current configuration summary"""
        print("=== Configuration Summary ===")
        print(f"Core: {self.config.core_inner_radius}-{self.config.core_outer_radius}mm radius")
        print(f"Wire: {self.config.wire_diameter}mm diameter, {self.config.num_turns} turns")
        print(f"Current: {self.circuit.current_amplitude}A")
        print(f"Frequency: {self.circuit.frequency}Hz")
        print(f"Iron permeability: {self.materials.iron_mu_r}")


def create_sample_configurations():
    """Create sample configuration files for different inductor types"""
    
    # Small inductor configuration
    small_config = {
        'geometry': {
            'core_outer_radius': 10.0,
            'core_inner_radius': 4.0,
            'core_height': 15.0,
            'wire_diameter': 0.5,
            'num_turns': 20,
            'wire_layers': 1,
            'winding_inner_radius': 4.5,
            'winding_outer_radius': 9.5,
            'winding_height': 12.0,
            'air_gap_radius': 25.0
        },
        'materials': {
            'iron_mu_r': 3000.0,
            'iron_h_c': 0.0,
            'iron_j': 0.0,
            'iron_conductivity': 1.6e6,
            'copper_mu_r': 1.0,
            'copper_h_c': 0.0,
            'copper_j': 0.0,
            'copper_conductivity': 58e6,
            'air_mu_r': 1.0,
            'air_h_c': 0.0,
            'air_j': 0.0,
            'air_conductivity': 0.0
        },
        'circuit': {
            'current_amplitude': 0.5,
            'frequency': 0.0,
            'circuit_name': 'SmallCoil',
            'turns': 20
        }
    }
    
    # Large inductor configuration
    large_config = {
        'geometry': {
            'core_outer_radius': 40.0,
            'core_inner_radius': 15.0,
            'core_height': 60.0,
            'wire_diameter': 1.2,
            'num_turns': 100,
            'wire_layers': 2,
            'winding_inner_radius': 16.0,
            'winding_outer_radius': 38.0,
            'winding_height': 50.0,
            'air_gap_radius': 100.0
        },
        'materials': {
            'iron_mu_r': 5000.0,
            'iron_h_c': 0.0,
            'iron_j': 0.0,
            'iron_conductivity': 2.0e6,
            'copper_mu_r': 1.0,
            'copper_h_c': 0.0,
            'copper_j': 0.0,
            'copper_conductivity': 58e6,
            'air_mu_r': 1.0,
            'air_h_c': 0.0,
            'air_j': 0.0,
            'air_conductivity': 0.0
        },
        'circuit': {
            'current_amplitude': 2.0,
            'frequency': 0.0,
            'circuit_name': 'LargeCoil',
            'turns': 100
        }
    }
    
    # Save sample configurations
    with open('small_inductor_config.json', 'w') as f:
        json.dump(small_config, f, indent=2)
    
    with open('large_inductor_config.json', 'w') as f:
        json.dump(large_config, f, indent=2)
    
    print("Sample configuration files created:")
    print("- small_inductor_config.json")
    print("- large_inductor_config.json")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Iron Core Inductor Simulation Workflow')
    parser.add_argument('--config', '-c', type=str, 
                       help='Configuration file (JSON format)')
    parser.add_argument('--output-dxf', '-o', type=str, default='inductor.dxf',
                       help='Output DXF filename')
    parser.add_argument('--no-dxf', action='store_true',
                       help='Skip DXF generation (use existing geometry)')
    parser.add_argument('--save-config', type=str,
                       help='Save current configuration to file')
    parser.add_argument('--create-samples', action='store_true',
                       help='Create sample configuration files')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Run analysis only (requires active FEMM solution)')
    
    args = parser.parse_args()
    
    if args.create_samples:
        create_sample_configurations()
        return
    
    # Create workflow instance
    workflow = InductorWorkflow(args.config)
    workflow.print_configuration_summary()
    
    # Save configuration if requested
    if args.save_config:
        workflow.save_configuration(args.save_config)
    
    if args.analysis_only:
        # Run analysis only
        workflow.analyze_results()
    else:
        # Run complete workflow
        generate_dxf = not args.no_dxf
        success = workflow.run_complete_workflow(generate_dxf, args.output_dxf)
        
        if success:
            print("Simulation workflow completed successfully!")
            sys.exit(0)
        else:
            print("Simulation workflow failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()