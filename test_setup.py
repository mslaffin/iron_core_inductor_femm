"""
Test script to verify installation and basic functionality
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import numpy
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        import ezdxf
        print("✓ ezdxf imported successfully")
    except ImportError as e:
        print(f"✗ ezdxf import failed: {e}")
        return False
    
    try:
        import femm
        print("✓ pyfemm imported successfully")
    except ImportError as e:
        print(f"✗ pyfemm import failed: {e}")
        print("  Make sure pyfemm 0.1.3 is installed and FEMM software is available")
        return False
    
    return True

def test_local_modules():
    """Test if local modules can be imported"""
    print("\nTesting local modules...")
    
    try:
        from dxf_generator import InductorConfig, DXFGenerator
        print("✓ dxf_generator module imported successfully")
    except ImportError as e:
        print(f"✗ dxf_generator import failed: {e}")
        return False
    
    try:
        from femm_simulation import FEMMSimulation, MaterialProperties, CircuitProperties
        print("✓ femm_simulation module imported successfully")
    except ImportError as e:
        print(f"✗ femm_simulation import failed: {e}")
        return False
    
    try:
        from analysis_tools import MagneticAnalyzer, AnalysisResults
        print("✓ analysis_tools module imported successfully")
    except ImportError as e:
        print(f"✗ analysis_tools import failed: {e}")
        return False
    
    try:
        from inductor_workflow import InductorWorkflow
        print("✓ inductor_workflow module imported successfully")
    except ImportError as e:
        print(f"✗ inductor_workflow import failed: {e}")
        return False
    
    return True

def test_dxf_generation():
    """Test DXF generation functionality"""
    print("\nTesting DXF generation...")
    
    try:
        from dxf_generator import InductorConfig, DXFGenerator
        
        # Create test configuration
        config = InductorConfig(
            core_outer_radius=10.0,
            core_inner_radius=4.0,
            wire_diameter=0.5,
            num_turns=10,
            winding_inner_radius=4.5,
            winding_outer_radius=9.5,
            air_gap_radius=20.0
        )
        
        # Generate test DXF
        generator = DXFGenerator(config)
        test_filename = "test_inductor.dxf"
        generator.generate_dxf(test_filename)
        
        if os.path.exists(test_filename):
            print(f"✓ DXF file generated successfully: {test_filename}")
            file_size = os.path.getsize(test_filename)
            print(f"  File size: {file_size} bytes")
            return True
        else:
            print("✗ DXF file was not created")
            return False
            
    except Exception as e:
        print(f"✗ DXF generation failed: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print("\nTesting configuration system...")
    
    try:
        from inductor_workflow import InductorWorkflow
        import json
        
        # Create workflow with default config
        workflow = InductorWorkflow()
        
        # Save configuration
        test_config_file = "test_config.json"
        workflow.save_configuration(test_config_file)
        
        if os.path.exists(test_config_file):
            print(f"✓ Configuration saved successfully: {test_config_file}")
            
            # Load configuration back
            workflow2 = InductorWorkflow(test_config_file)
            print("✓ Configuration loaded successfully")
            
            # Clean up
            os.remove(test_config_file)
            print("✓ Test configuration file cleaned up")
            return True
        else:
            print("✗ Configuration file was not created")
            return False
            
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Iron Core Inductor Simulation - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Local Module Tests", test_local_modules),
        ("DXF Generation Test", test_dxf_generation),
        ("Configuration Test", test_configuration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Setup is ready for simulation.")
        print("\nNext steps:")
        print("1. Run: python inductor_workflow.py --create-samples")
        print("2. Run: python inductor_workflow.py")
    else:
        print("❌ Some tests failed. Please check the requirements and installation.")
        print("\nTo install missing dependencies:")
        print("pip install -r requirements.txt")
    print("=" * 50)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)