#!/usr/bin/env python3
"""
Quick Start Script for Advanced OCR Enhancement System
Demonstrates the key features and capabilities
"""

import sys
from pathlib import Path
import time

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from advanced_ocr_enhancement import AdvancedOCRSystem
except ImportError as e:
    print(f"Error: Could not import AdvancedOCRSystem: {e}")
    print("Make sure advanced_ocr_enhancement.py is in the same directory")
    sys.exit(1)


def main():
    """Quick start demonstration"""
    print("🚀 ADVANCED OCR ENHANCEMENT SYSTEM - QUICK START")
    print("=" * 60)
    
    # Check for test image
    test_image = Path("Receipts/walmart.png")
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        print("Please ensure Receipts/walmart.png exists")
        return
    
    print(f"✅ Found test image: {test_image}")
    
    # Initialize system
    print("\n🔧 Initializing Advanced OCR System...")
    try:
        system = AdvancedOCRSystem()
        print(f"✅ System initialized with {len(system.extractors)} extractors:")
        for extractor in system.extractors:
            print(f"   - {extractor.get_name()}")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Test single extraction
    print("\n📄 Testing text extraction...")
    try:
        start_time = time.time()
        result = system.extract_text(test_image, use_ml_enhancement=False)
        processing_time = time.time() - start_time
        
        print(f"✅ Extraction completed in {processing_time:.2f}s")
        print(f"📊 Results:")
        print(f"   Method: {result.method}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Quality Score: {result.quality_score:.3f}")
        print(f"   Text Length: {len(result.text)} characters")
        
        # Show first 300 characters of extracted text
        preview = result.text[:300] + "..." if len(result.text) > 300 else result.text
        print(f"\n📝 Extracted Text Preview:")
        print("-" * 40)
        print(preview)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return
    
    # Test benchmarking (if multiple extractors available)
    if len(system.extractors) > 1:
        print("\n📊 Running quick benchmark...")
        try:
            benchmark_results = system.run_benchmark([test_image])
            
            if benchmark_results and 'rankings' in benchmark_results:
                print("🏆 Benchmark Rankings:")
                for metric, ranking in benchmark_results['rankings'].items():
                    if ranking:
                        print(f"   {metric}: {ranking[0]}")
            
        except Exception as e:
            print(f"⚠️  Benchmark failed: {e}")
    
    print("\n🎉 Quick start completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Run the full test suite: python Test/test_advanced_ocr_enhancement.py")
    print("   2. Try different images in the Receipts directory")
    print("   3. Configure custom extractors and weights")
    print("   4. Train the ML quality predictor with labeled data")


if __name__ == "__main__":
    main()
