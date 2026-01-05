

# **NPU vs CPU Benchmark Comparison**

### **Snapdragon X Elite \-  YOLOv8s INT8**

## ---

**Test Configuration**

* **Model:** yolov8s\_qdq.onnx (INT8 Quantized)  
* **Dataset:** COCO128 (100 images)  
* **Image Size:** 640x640  
* **Warmup:** 5 iterations

## ---

**Performance Results**

| Metric | NPU (QNN) | CPU Only | Speedup |
| :---- | :---- | :---- | :---- |
| **Total Time** | 2.92 s | 42.78 s | **14.6x** |
| **Average Latency** | 29.22 ms | 427.82 ms | **14.6x** |
| **Throughput (FPS)** | 34.23 FPS | 2.34 FPS | **14.6x** |

## ---

**Detailed Analysis**

### **NPU (QNN HTP Backend)**

* **Total Time:** 2.92 seconds for 100 images.  
* **Average Latency:** 29.22 ms per frame.  
* **Throughput:** 34.23 FPS (**Real-time capable**).  
* **Verdict:** Excellent for real-time applications and low-power inference.

### **CPU Only**

* **Total Time:** 42.78 seconds for 100 images.  
* **Average Latency:** 427.82 ms per frame.  
* **Throughput:** 2.34 FPS (**Not real-time**).  
* **Verdict:** Too slow for live video; suitable only for static batch processing.

## ---

**Key Findings**

1. **NPU Acceleration:** The NPU is **14.6x faster** than the CPU, reducing frame processing time from nearly half a second to under 30ms.  
2. **Real-time Capability:** At **34+ FPS**, the NPU provides smooth video processing. The CPU’s 2.34 FPS results in a "slideshow" effect.  
3. **Efficiency:** The NPU completes the entire workload in 3 seconds, whereas the CPU requires 43 seconds. This indicates massive power savings and thermal efficiency.  
4. **Ideal Use Cases:**  
   * **NPU:** Live object detection, augmented reality, and background blur.  
   * **CPU:** Fallback for models not yet optimized for Hexagon DSP/NPU.

## ---

**Conclusion**

The **Snapdragon X Elite NPU** provides exceptional acceleration for AI workloads. The transition from 2.34 FPS to 34.23 FPS marks the difference between a non-functional application and a seamless real-time experience.

