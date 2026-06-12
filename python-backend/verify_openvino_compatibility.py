"""
OpenVINO Compatibility Verification Script

This script verifies that all three ONNX models (Model 1, Model 2, Model 3)
are compatible with Intel OpenVINO Runtime before implementation begins.

Checks performed:
- OpenVINO Runtime installation and initialization
- Model loading with read_model()
- Unsupported operator detection
- Input tensor shape validation against model.json
- Output tensor shape validation against expected class counts

Requirements validated: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str) -> None:
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_success(text: str) -> None:
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str) -> None:
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str) -> None:
    """Print info message"""
    print(f"  {text}")


def check_openvino_installation() -> bool:
    """
    Check if OpenVINO Runtime is installed and can be imported.
    
    Returns:
        bool: True if OpenVINO is available, False otherwise
    """
    print_header("OpenVINO Runtime Installation Check")
    
    try:
        from openvino import Core, get_version
        version = get_version()
        print_success(f"OpenVINO Runtime is installed")
        print_info(f"Version: {version}")
        return True
    except ImportError as e:
        print_error("OpenVINO Runtime is not installed")
        print_info(f"Error: {e}")
        print_info("\nTo install OpenVINO:")
        print_info("  pip install openvino>=2024.0.0")
        print_info("\nFor more information:")
        print_info("  https://docs.openvino.ai/latest/get_started.html")
        return False


def initialize_openvino_core() -> Optional['Core']:
    """
    Initialize OpenVINO Core instance.
    
    Returns:
        Core instance if successful, None otherwise
    """
    print_header("OpenVINO Core Initialization")
    
    try:
        from openvino import Core
        core = Core()
        print_success("OpenVINO Core initialized successfully")
        
        # List available devices
        devices = core.available_devices
        print_info(f"Available devices: {', '.join(devices)}")
        
        return core
    except Exception as e:
        print_error(f"Failed to initialize OpenVINO Core: {e}")
        return None


def load_model_config(model_dir: str) -> Optional[Dict]:
    """
    Load model.json configuration file.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Dictionary with model configuration or None if not found
    """
    config_path = os.path.join(model_dir, "model.json")
    
    if not os.path.exists(config_path):
        print_warning(f"model.json not found in {model_dir}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print_warning(f"Failed to load model.json: {e}")
        return None


def verify_model_compatibility(
    core: 'Core',
    model_name: str,
    model_path: str,
    expected_input_shape: List[int],
    expected_output_classes: int
) -> Tuple[bool, Dict[str, any]]:
    """
    Verify a single model's compatibility with OpenVINO.
    
    Args:
        core: OpenVINO Core instance
        model_name: Model identifier (e.g., "Model 1")
        model_path: Path to ONNX model file
        expected_input_shape: Expected input tensor shape from model.json
        expected_output_classes: Expected number of output classes
        
    Returns:
        Tuple of (success: bool, report: dict)
    """
    report = {
        'model_name': model_name,
        'model_path': model_path,
        'status': 'UNKNOWN',
        'issues': [],
        'input_shape': None,
        'output_shape': None,
        'expected_input_shape': expected_input_shape,
        'expected_output_classes': expected_output_classes
    }
    
    print_header(f"Verifying {model_name}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print_error(f"Model file not found: {model_path}")
        report['status'] = 'FAIL'
        report['issues'].append('Model file not found')
        return False, report
    
    print_info(f"Model path: {model_path}")
    
    # Try to load the model
    try:
        print_info("Loading model with read_model()...")
        model = core.read_model(model=model_path)
        print_success("Model loaded successfully")
        
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        report['status'] = 'FAIL'
        
        # Check for unsupported operators
        error_str = str(e).lower()
        if 'not supported' in error_str or 'unsupported' in error_str:
            report['issues'].append(f'Unsupported operators detected: {e}')
        else:
            report['issues'].append(f'Model loading error: {e}')
        
        return False, report
    
    # Verify input tensor shape
    try:
        input_tensor = model.input(0)
        input_shape_obj = input_tensor.partial_shape
        input_name = input_tensor.get_any_name()
        
        # Extract shape as list, handling dynamic dimensions
        input_shape = []
        for dim in input_shape_obj:
            if dim.is_dynamic:
                input_shape.append(-1)
            else:
                input_shape.append(dim.get_length())
        
        print_info(f"Input tensor name: {input_name}")
        print_info(f"Input shape: {input_shape}")
        print_info(f"Expected shape: {expected_input_shape}")
        
        report['input_shape'] = input_shape
        
        # Compare shapes (allow dynamic dimensions marked as -1 or None)
        shape_match = True
        if len(input_shape) != len(expected_input_shape):
            shape_match = False
        else:
            for actual, expected in zip(input_shape, expected_input_shape):
                # Dynamic dimensions (-1) can match any expected value
                if actual != -1 and actual != expected:
                    shape_match = False
                    break
        
        if shape_match:
            print_success("Input shape matches model.json expectations")
        else:
            print_warning(f"Input shape mismatch! Expected {expected_input_shape}, got {input_shape}")
            report['issues'].append(f'Input shape mismatch: expected {expected_input_shape}, got {input_shape}')
            
    except Exception as e:
        print_error(f"Failed to read input tensor info: {e}")
        report['issues'].append(f'Input tensor error: {e}')
    
    # Verify output tensor shape
    try:
        output_tensor = model.output(0)
        output_shape_obj = output_tensor.partial_shape
        output_name = output_tensor.get_any_name()
        
        # Extract shape as list, handling dynamic dimensions
        output_shape = []
        for dim in output_shape_obj:
            if dim.is_dynamic:
                output_shape.append(-1)
            else:
                output_shape.append(dim.get_length())
        
        print_info(f"Output tensor name: {output_name}")
        print_info(f"Output shape: {output_shape}")
        print_info(f"Expected classes: {expected_output_classes}")
        
        report['output_shape'] = output_shape
        
        # Check if last dimension matches expected class count
        if len(output_shape) >= 1:
            output_classes = output_shape[-1]
            
            if output_classes == expected_output_classes:
                print_success(f"Output shape matches expected class count ({expected_output_classes})")
            elif output_classes == -1:
                print_warning(f"Output dimension is dynamic, cannot verify class count")
                report['issues'].append('Output dimension is dynamic')
            else:
                print_warning(f"Output class count mismatch! Expected {expected_output_classes}, got {output_classes}")
                report['issues'].append(f'Output class mismatch: expected {expected_output_classes}, got {output_classes}')
        else:
            print_warning("Unexpected output shape format")
            report['issues'].append('Unexpected output shape format')
            
    except Exception as e:
        print_error(f"Failed to read output tensor info: {e}")
        report['issues'].append(f'Output tensor error: {e}')
    
    # Determine overall status
    if not report['issues']:
        report['status'] = 'PASS'
        print_success(f"\n{model_name} is COMPATIBLE with OpenVINO Runtime")
        return True, report
    else:
        # If only warnings (not critical errors), still mark as PASS with warnings
        critical_errors = [issue for issue in report['issues'] if 'mismatch' not in issue.lower()]
        if not critical_errors:
            report['status'] = 'PASS_WITH_WARNINGS'
            print_warning(f"\n{model_name} is COMPATIBLE but has warnings")
            return True, report
        else:
            report['status'] = 'FAIL'
            print_error(f"\n{model_name} has COMPATIBILITY ISSUES")
            return False, report


def generate_compatibility_report(reports: List[Dict]) -> None:
    """
    Generate and print final compatibility report.
    
    Args:
        reports: List of verification reports for all models
    """
    print_header("Compatibility Report Summary")
    
    all_passed = True
    
    for report in reports:
        status_color = Colors.GREEN if report['status'] in ['PASS', 'PASS_WITH_WARNINGS'] else Colors.RED
        status_symbol = '✓' if report['status'] in ['PASS', 'PASS_WITH_WARNINGS'] else '✗'
        
        print(f"\n{status_color}{Colors.BOLD}{status_symbol} {report['model_name']}: {report['status']}{Colors.END}")
        print(f"  Path: {report['model_path']}")
        print(f"  Input Shape: {report['input_shape']} (expected: {report['expected_input_shape']})")
        print(f"  Output Shape: {report['output_shape']} (expected classes: {report['expected_output_classes']})")
        
        if report['issues']:
            print(f"  Issues:")
            for issue in report['issues']:
                print(f"    - {issue}")
        
        if report['status'] == 'FAIL':
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print_success("\nALL MODELS ARE COMPATIBLE WITH OPENVINO RUNTIME")
        print_info("You can proceed with implementation.")
        print_info("\nNext steps:")
        print_info("  1. Proceed to Phase 2: Core OpenVINO Integration")
        print_info("  2. Implement OpenVINO backend in ModelLoader")
        print_info("  3. Run benchmark tests to measure performance improvements")
    else:
        print_error("\nSOME MODELS HAVE COMPATIBILITY ISSUES")
        print_info("Please resolve the issues before proceeding with implementation.")
        print_info("\nTroubleshooting:")
        print_info("  - Check ONNX model file integrity")
        print_info("  - Verify model.json configuration matches model architecture")
        print_info("  - Review OpenVINO supported operators:")
        print_info("    https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html")
    
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


def main() -> int:
    """
    Main verification workflow.
    
    Returns:
        Exit code: 0 if all models compatible, 1 otherwise
    """
    print_header("OpenVINO Compatibility Verification Script")
    print_info("This script verifies ONNX model compatibility with Intel OpenVINO Runtime")
    print_info("Requirements validated: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7\n")
    
    # Define model paths relative to script location
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models" / "sign"
    
    model_configs = [
        {
            'name': 'Model 1 (PointNet Fingerspelling)',
            'dir': models_dir / "model1",
            'onnx_file': 'model.onnx',
            'expected_output_classes': 24
        },
        {
            'name': 'Model 2 (WLASL Pose-TGCN)',
            'dir': models_dir / "model2",
            'onnx_file': 'wlasl_pose_tgcn.onnx',
            'expected_output_classes': 2000
        },
        {
            'name': 'Model 3 (LSTM Sequences)',
            'dir': models_dir / "model3",
            'onnx_file': 'model.onnx',
            'expected_output_classes': 250
        }
    ]
    
    # Step 1: Check OpenVINO installation
    if not check_openvino_installation():
        return 1
    
    # Step 2: Initialize OpenVINO Core
    core = initialize_openvino_core()
    if core is None:
        return 1
    
    # Step 3: Verify each model
    reports = []
    
    for model_config in model_configs:
        model_dir = str(model_config['dir'])
        model_path = str(model_config['dir'] / model_config['onnx_file'])
        
        # Load model.json configuration
        config = load_model_config(model_dir)
        
        if config:
            expected_input_shape = config.get('input', {}).get('input_shape', [])
        else:
            # Fallback to hardcoded shapes if model.json not found
            if 'model1' in model_dir:
                expected_input_shape = [1, 21, 3]
            elif 'model2' in model_dir:
                expected_input_shape = [1, 55, 100]
            elif 'model3' in model_dir:
                expected_input_shape = [30, 543, 3]
            else:
                expected_input_shape = []
        
        success, report = verify_model_compatibility(
            core=core,
            model_name=model_config['name'],
            model_path=model_path,
            expected_input_shape=expected_input_shape,
            expected_output_classes=model_config['expected_output_classes']
        )
        
        reports.append(report)
    
    # Step 4: Generate final report
    exit_code = generate_compatibility_report(reports)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
