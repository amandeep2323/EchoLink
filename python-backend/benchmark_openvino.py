"""
OpenVINO Performance Benchmark Script
======================================
Comprehensive performance comparison between ONNX Runtime and OpenVINO Runtime.

Benchmarks:
  - Model load time (cold start, warm start)
  - Inference latency (per-frame)
  - Throughput (FPS over duration)
  - Memory usage (footprint)

Usage:
    python benchmark_openvino.py --model model1
    python benchmark_openvino.py --model model2
    python benchmark_openvino.py --model model3
    python benchmark_openvino.py --all

Requirements:
    - openvino>=2024.0.0
    - onnxruntime
    - numpy
    - psutil
"""

import os
import sys
import time
import argparse
import json
import shutil
import numpy as np
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.model_loader import ModelLoader
from models.model_config import ModelConfig


class PerformanceBenchmark:
    """
    Comprehensive performance comparison between ONNX Runtime and OpenVINO.
    
    Measures:
        - Model load time (cold start vs warm start)
        - Inference latency (per-frame mean and std)
        - Throughput (FPS over duration)
        - Memory usage (footprint in MB)
    """
    
    def __init__(self, model_config_path: str):
        """
        Initialize benchmark for a specific model.
        
        Args:
            model_config_path: Path to model.json configuration file
        """
        self.config = ModelConfig.load(model_config_path)
        self.model_path = self.config.model_path
        self.input_shape = tuple(self.config.input.input_shape)
        
        # Cache directory for OpenVINO
        self.cache_dir = "cache/openvino/"
        
        print(f"[Benchmark] Initialized for {self.config.name}")
        print(f"[Benchmark]   Model: {self.model_path}")
        print(f"[Benchmark]   Input shape: {self.input_shape}")
        print(f"[Benchmark]   Model type: {self.config.type}")
    
    def _clear_cache(self) -> None:
        """Clear OpenVINO cache to simulate cold start."""
        if os.path.exists(self.cache_dir):
            try:
                shutil.rmtree(self.cache_dir)
                print(f"[Benchmark] Cache cleared: {self.cache_dir}")
            except Exception as e:
                print(f"[Benchmark] ⚠ Failed to clear cache: {e}")
    
    def _generate_test_input(self) -> np.ndarray:
        """
        Generate random test input with correct shape and dtype.
        
        Returns:
            Random input tensor matching model's input shape
        """
        # Generate random input with values in [0, 1] range (typical for normalized inputs)
        test_input = np.random.rand(*self.input_shape).astype(np.float32)
        return test_input
    
    def benchmark_model_load_time(self) -> Dict:
        """
        Measure model load time for both runtimes.
        
        Measures:
            - ONNX Runtime baseline load time
            - OpenVINO cold start (no cache)
            - OpenVINO warm start (with cache)
        
        Returns:
            {
                'onnx_load_time': float (seconds),
                'openvino_cold_load_time': float (seconds),
                'openvino_warm_load_time': float (seconds),
                'speedup_cold': float (ratio),
                'speedup_warm': float (ratio)
            }
        """
        print("\n" + "="*60)
        print("Benchmark: Model Load Time")
        print("="*60)
        
        results = {}
        
        # Baseline: ONNX Runtime load time
        print("\n[1/3] Measuring ONNX Runtime load time...")
        loader_onnx = ModelLoader()
        # Temporarily disable OpenVINO compatibility check
        loader_onnx._is_model_openvino_compatible = lambda path: False
        
        start = time.time()
        loader_onnx.load_from_config(self.config, use_gpu=False)
        onnx_load_time = time.time() - start
        results['onnx_load_time'] = onnx_load_time
        print(f"   ONNX Runtime load time: {onnx_load_time:.3f}s")
        
        # Cleanup
        loader_onnx.unload()
        del loader_onnx
        
        # OpenVINO: Cold start (no cache)
        print("\n[2/3] Measuring OpenVINO cold start (no cache)...")
        self._clear_cache()
        time.sleep(0.5)  # Allow filesystem to settle
        
        loader_ov_cold = ModelLoader()
        start = time.time()
        loader_ov_cold.load_from_config(self.config, use_gpu=False)
        openvino_cold_time = time.time() - start
        results['openvino_cold_load_time'] = openvino_cold_time
        print(f"   OpenVINO cold start: {openvino_cold_time:.3f}s")
        
        # Cleanup
        loader_ov_cold.unload()
        del loader_ov_cold
        
        # OpenVINO: Warm start (with cache)
        print("\n[3/3] Measuring OpenVINO warm start (with cache)...")
        time.sleep(0.5)
        
        loader_ov_warm = ModelLoader()
        start = time.time()
        loader_ov_warm.load_from_config(self.config, use_gpu=False)
        openvino_warm_time = time.time() - start
        results['openvino_warm_load_time'] = openvino_warm_time
        print(f"   OpenVINO warm start: {openvino_warm_time:.3f}s")
        
        # Cleanup
        loader_ov_warm.unload()
        del loader_ov_warm
        
        # Calculate speedups
        results['speedup_cold'] = onnx_load_time / openvino_cold_time if openvino_cold_time > 0 else 0
        results['speedup_warm'] = onnx_load_time / openvino_warm_time if openvino_warm_time > 0 else 0
        results['cache_improvement_percent'] = ((openvino_cold_time - openvino_warm_time) / openvino_cold_time * 100) if openvino_cold_time > 0 else 0
        
        print(f"\n   Speedup (cold): {results['speedup_cold']:.2f}x")
        print(f"   Speedup (warm): {results['speedup_warm']:.2f}x")
        print(f"   Cache improvement: {results['cache_improvement_percent']:.1f}%")
        
        return results
    
    def benchmark_inference_latency(self, num_iterations: int = 100) -> Dict:
        """
        Measure per-frame inference latency.
        
        Args:
            num_iterations: Number of inference iterations to average
        
        Returns:
            {
                'onnx_mean_latency': float (ms),
                'onnx_std_latency': float (ms),
                'openvino_mean_latency': float (ms),
                'openvino_std_latency': float (ms),
                'speedup': float (ratio),
                'improvement_percent': float
            }
        """
        print("\n" + "="*60)
        print(f"Benchmark: Inference Latency ({num_iterations} iterations)")
        print("="*60)
        
        results = {}
        
        # Generate test inputs
        print(f"\nGenerating {num_iterations} test inputs...")
        test_inputs = [self._generate_test_input() for _ in range(num_iterations)]
        
        # Baseline: ONNX Runtime
        print("\n[1/2] Measuring ONNX Runtime latency...")
        loader_onnx = ModelLoader()
        loader_onnx._is_model_openvino_compatible = lambda path: False
        loader_onnx.load_from_config(self.config, use_gpu=False)
        
        # Warmup
        for _ in range(5):
            loader_onnx.predict_raw(test_inputs[0])
        
        # Measure
        onnx_latencies = []
        for test_input in test_inputs:
            start = time.perf_counter()
            _ = loader_onnx.predict_raw(test_input)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            onnx_latencies.append(elapsed)
        
        results['onnx_mean_latency'] = np.mean(onnx_latencies)
        results['onnx_std_latency'] = np.std(onnx_latencies)
        results['onnx_min_latency'] = np.min(onnx_latencies)
        results['onnx_max_latency'] = np.max(onnx_latencies)
        
        print(f"   ONNX Runtime: {results['onnx_mean_latency']:.2f} ± {results['onnx_std_latency']:.2f} ms")
        print(f"   ONNX Runtime range: [{results['onnx_min_latency']:.2f}, {results['onnx_max_latency']:.2f}] ms")
        
        # Cleanup
        loader_onnx.unload()
        del loader_onnx
        
        # OpenVINO Runtime
        print("\n[2/2] Measuring OpenVINO Runtime latency...")
        loader_ov = ModelLoader()
        loader_ov.load_from_config(self.config, use_gpu=False)
        
        # Warmup
        for _ in range(5):
            loader_ov.predict_raw(test_inputs[0])
        
        # Measure
        openvino_latencies = []
        for test_input in test_inputs:
            start = time.perf_counter()
            _ = loader_ov.predict_raw(test_input)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            openvino_latencies.append(elapsed)
        
        results['openvino_mean_latency'] = np.mean(openvino_latencies)
        results['openvino_std_latency'] = np.std(openvino_latencies)
        results['openvino_min_latency'] = np.min(openvino_latencies)
        results['openvino_max_latency'] = np.max(openvino_latencies)
        
        print(f"   OpenVINO: {results['openvino_mean_latency']:.2f} ± {results['openvino_std_latency']:.2f} ms")
        print(f"   OpenVINO range: [{results['openvino_min_latency']:.2f}, {results['openvino_max_latency']:.2f}] ms")
        
        # Cleanup
        loader_ov.unload()
        del loader_ov
        
        # Calculate speedup
        results['speedup'] = results['onnx_mean_latency'] / results['openvino_mean_latency'] if results['openvino_mean_latency'] > 0 else 0
        results['improvement_percent'] = ((results['onnx_mean_latency'] - results['openvino_mean_latency']) / results['onnx_mean_latency'] * 100) if results['onnx_mean_latency'] > 0 else 0
        
        print(f"\n   Speedup: {results['speedup']:.2f}x")
        print(f"   Improvement: {results['improvement_percent']:.1f}%")
        
        return results
    
    def benchmark_throughput(self, duration_seconds: int = 10) -> Dict:
        """
        Measure frames per second (FPS) over duration.
        
        Args:
            duration_seconds: Duration to run throughput test
        
        Returns:
            {
                'onnx_fps': float,
                'openvino_fps': float,
                'improvement_percent': float
            }
        """
        print("\n" + "="*60)
        print(f"Benchmark: Throughput (FPS over {duration_seconds}s)")
        print("="*60)
        
        results = {}
        test_input = self._generate_test_input()
        
        # Baseline: ONNX Runtime
        print("\n[1/2] Measuring ONNX Runtime throughput...")
        loader_onnx = ModelLoader()
        loader_onnx._is_model_openvino_compatible = lambda path: False
        loader_onnx.load_from_config(self.config, use_gpu=False)
        
        # Warmup
        for _ in range(10):
            loader_onnx.predict_raw(test_input)
        
        # Measure
        start = time.time()
        count = 0
        while time.time() - start < duration_seconds:
            _ = loader_onnx.predict_raw(test_input)
            count += 1
        elapsed = time.time() - start
        
        onnx_fps = count / elapsed
        results['onnx_fps'] = onnx_fps
        results['onnx_frame_count'] = count
        
        print(f"   ONNX Runtime: {onnx_fps:.2f} FPS ({count} frames in {elapsed:.2f}s)")
        
        # Cleanup
        loader_onnx.unload()
        del loader_onnx
        
        # OpenVINO Runtime
        print("\n[2/2] Measuring OpenVINO Runtime throughput...")
        loader_ov = ModelLoader()
        loader_ov.load_from_config(self.config, use_gpu=False)
        
        # Warmup
        for _ in range(10):
            loader_ov.predict_raw(test_input)
        
        # Measure
        start = time.time()
        count = 0
        while time.time() - start < duration_seconds:
            _ = loader_ov.predict_raw(test_input)
            count += 1
        elapsed = time.time() - start
        
        openvino_fps = count / elapsed
        results['openvino_fps'] = openvino_fps
        results['openvino_frame_count'] = count
        
        print(f"   OpenVINO: {openvino_fps:.2f} FPS ({count} frames in {elapsed:.2f}s)")
        
        # Cleanup
        loader_ov.unload()
        del loader_ov
        
        # Calculate improvement
        results['improvement_percent'] = ((openvino_fps - onnx_fps) / onnx_fps * 100) if onnx_fps > 0 else 0
        results['speedup'] = openvino_fps / onnx_fps if onnx_fps > 0 else 0
        
        print(f"\n   Improvement: {results['improvement_percent']:.1f}%")
        print(f"   Speedup: {results['speedup']:.2f}x")
        
        return results
    
    def benchmark_memory_usage(self) -> Dict:
        """
        Measure memory footprint for both runtimes.
        
        Returns:
            {
                'onnx_memory_mb': float,
                'openvino_memory_mb': float,
                'difference_mb': float,
                'difference_percent': float
            }
        """
        print("\n" + "="*60)
        print("Benchmark: Memory Usage")
        print("="*60)
        
        results = {}
        process = psutil.Process()
        test_input = self._generate_test_input()
        
        # Baseline: ONNX Runtime
        print("\n[1/2] Measuring ONNX Runtime memory...")
        
        # Get baseline memory before loading
        import gc
        gc.collect()
        time.sleep(1)
        baseline_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        loader_onnx = ModelLoader()
        loader_onnx._is_model_openvino_compatible = lambda path: False
        loader_onnx.load_from_config(self.config, use_gpu=False)
        
        # Run inference to stabilize memory
        for _ in range(10):
            loader_onnx.predict_raw(test_input)
        
        time.sleep(1)
        onnx_mem = process.memory_info().rss / (1024 * 1024)  # MB
        onnx_mem_usage = onnx_mem - baseline_mem
        results['onnx_memory_mb'] = onnx_mem_usage
        
        print(f"   ONNX Runtime memory: {onnx_mem_usage:.2f} MB")
        
        # Cleanup
        loader_onnx.unload()
        del loader_onnx
        gc.collect()
        time.sleep(1)
        
        # OpenVINO Runtime
        print("\n[2/2] Measuring OpenVINO Runtime memory...")
        
        baseline_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        loader_ov = ModelLoader()
        loader_ov.load_from_config(self.config, use_gpu=False)
        
        # Run inference to stabilize memory
        for _ in range(10):
            loader_ov.predict_raw(test_input)
        
        time.sleep(1)
        openvino_mem = process.memory_info().rss / (1024 * 1024)  # MB
        openvino_mem_usage = openvino_mem - baseline_mem
        results['openvino_memory_mb'] = openvino_mem_usage
        
        print(f"   OpenVINO memory: {openvino_mem_usage:.2f} MB")
        
        # Cleanup
        loader_ov.unload()
        del loader_ov
        
        # Calculate difference
        results['difference_mb'] = openvino_mem_usage - onnx_mem_usage
        results['difference_percent'] = (results['difference_mb'] / onnx_mem_usage * 100) if onnx_mem_usage > 0 else 0
        
        print(f"\n   Difference: {results['difference_mb']:+.2f} MB ({results['difference_percent']:+.1f}%)")
        
        return results
    
    def run_all_benchmarks(self) -> Dict:
        """
        Run all benchmark tests and return consolidated results.
        
        Returns:
            Dictionary containing all benchmark results
        """
        print("\n" + "="*60)
        print(f"Running Full Benchmark Suite: {self.config.name}")
        print("="*60)
        
        results = {
            'model_name': self.config.name,
            'model_path': self.model_path,
            'input_shape': self.input_shape,
            'model_type': self.config.type,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # Run benchmarks
        results['load_time'] = self.benchmark_model_load_time()
        results['latency'] = self.benchmark_inference_latency(num_iterations=100)
        results['throughput'] = self.benchmark_throughput(duration_seconds=10)
        results['memory'] = self.benchmark_memory_usage()
        
        return results
    
    def generate_report(self, results: Dict, output_file: Optional[str] = None) -> str:
        """
        Generate a formatted markdown report from benchmark results.
        
        Args:
            results: Benchmark results dictionary
            output_file: Optional path to save report
        
        Returns:
            Markdown formatted report string
        """
        report = []
        report.append(f"# Performance Benchmark Report: {results['model_name']}")
        report.append(f"\n**Generated:** {results['timestamp']}")
        report.append(f"\n**Model:** `{results['model_path']}`")
        report.append(f"\n**Input Shape:** `{results['input_shape']}`")
        report.append(f"\n**Model Type:** `{results['model_type']}`")
        
        # Load time
        report.append("\n## 1. Model Load Time")
        report.append("\n| Runtime | Load Time | Speedup vs ONNX |")
        report.append("|---------|-----------|-----------------|")
        lt = results['load_time']
        report.append(f"| ONNX Runtime | {lt['onnx_load_time']:.3f}s | 1.00x |")
        report.append(f"| OpenVINO (cold) | {lt['openvino_cold_load_time']:.3f}s | {lt['speedup_cold']:.2f}x |")
        report.append(f"| OpenVINO (warm) | {lt['openvino_warm_load_time']:.3f}s | {lt['speedup_warm']:.2f}x |")
        report.append(f"\n**Cache Improvement:** {lt['cache_improvement_percent']:.1f}% faster with cache")
        
        # Latency
        report.append("\n## 2. Inference Latency")
        report.append("\n| Runtime | Mean ± Std | Min | Max | Speedup |")
        report.append("|---------|------------|-----|-----|---------|")
        lat = results['latency']
        report.append(f"| ONNX Runtime | {lat['onnx_mean_latency']:.2f} ± {lat['onnx_std_latency']:.2f} ms | {lat['onnx_min_latency']:.2f} ms | {lat['onnx_max_latency']:.2f} ms | 1.00x |")
        report.append(f"| OpenVINO | {lat['openvino_mean_latency']:.2f} ± {lat['openvino_std_latency']:.2f} ms | {lat['openvino_min_latency']:.2f} ms | {lat['openvino_max_latency']:.2f} ms | {lat['speedup']:.2f}x |")
        report.append(f"\n**Performance Improvement:** {lat['improvement_percent']:.1f}% faster")
        
        # Throughput
        report.append("\n## 3. Throughput (FPS)")
        report.append("\n| Runtime | FPS | Frames Processed | Improvement |")
        report.append("|---------|-----|------------------|-------------|")
        tp = results['throughput']
        report.append(f"| ONNX Runtime | {tp['onnx_fps']:.2f} | {tp['onnx_frame_count']} | baseline |")
        report.append(f"| OpenVINO | {tp['openvino_fps']:.2f} | {tp['openvino_frame_count']} | +{tp['improvement_percent']:.1f}% |")
        
        # Memory
        report.append("\n## 4. Memory Usage")
        report.append("\n| Runtime | Memory Footprint | Difference |")
        report.append("|---------|------------------|------------|")
        mem = results['memory']
        report.append(f"| ONNX Runtime | {mem['onnx_memory_mb']:.2f} MB | baseline |")
        report.append(f"| OpenVINO | {mem['openvino_memory_mb']:.2f} MB | {mem['difference_mb']:+.2f} MB ({mem['difference_percent']:+.1f}%) |")
        
        # Summary
        report.append("\n## Summary")
        report.append(f"\n- **Load Time (warm):** {lt['speedup_warm']:.2f}x faster")
        report.append(f"- **Inference Latency:** {lat['speedup']:.2f}x faster ({lat['improvement_percent']:.1f}% improvement)")
        report.append(f"- **Throughput:** {tp['improvement_percent']:.1f}% improvement")
        report.append(f"- **Memory:** {mem['difference_mb']:+.2f} MB ({mem['difference_percent']:+.1f}%)")
        
        report_text = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n[Benchmark] Report saved to: {output_file}")
        
        return report_text


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="OpenVINO Performance Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_openvino.py --model model1
    python benchmark_openvino.py --model model2 --output results_model2.md
    python benchmark_openvino.py --all
        """
    )
    
    parser.add_argument(
        '--model',
        choices=['model1', 'model2', 'model3'],
        help='Model to benchmark (model1, model2, or model3)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Benchmark all three models'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for benchmark report (markdown format)'
    )
    
    parser.add_argument(
        '--json',
        type=str,
        help='Output file for JSON results'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model and not args.all:
        parser.error("Either --model or --all must be specified")
    
    # Determine which models to benchmark
    if args.all:
        models_to_test = ['model1', 'model2', 'model3']
    else:
        models_to_test = [args.model]
    
    # Find model directories
    models_dir = Path(__file__).parent / "models" / "sign"
    
    all_results = []
    
    for model_name in models_to_test:
        model_config_path = models_dir / model_name / "model.json"
        
        if not model_config_path.exists():
            print(f"\n[Error] Configuration not found: {model_config_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Starting benchmark for {model_name}")
        print(f"{'='*60}")
        
        try:
            benchmark = PerformanceBenchmark(str(model_config_path))
            results = benchmark.run_all_benchmarks()
            all_results.append(results)
            
            # Generate report for this model
            if args.output:
                if len(models_to_test) == 1:
                    output_file = args.output
                else:
                    base, ext = os.path.splitext(args.output)
                    output_file = f"{base}_{model_name}{ext}"
            else:
                output_file = f"benchmark_report_{model_name}.md"
            
            report = benchmark.generate_report(results, output_file)
            
            print("\n" + "="*60)
            print("Benchmark Complete!")
            print("="*60)
            print(report)
            
        except Exception as e:
            print(f"\n[Error] Benchmark failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save JSON results if requested
    if args.json and all_results:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[Benchmark] JSON results saved to: {args.json}")
    
    print("\n[Benchmark] All benchmarks completed!")


if __name__ == "__main__":
    main()
