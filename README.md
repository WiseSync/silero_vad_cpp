# Voice Activity Detection for C++

## Overview

This package aims to provide an accurate, user-friendly voice activity detector (VAD) that runs in the C++. It runs [Silero VAD](https://github.com/snakers4/silero-vad) [[1]](#1) using [ONNX Runtime C++](https://onnxruntime.ai/docs/get-started/with-cpp.html). 

## Requirements

Code are tested in the environments bellow, feel free to try others.

- C++ 17 or above
- onnxruntime-1.12.1 or above
- cmake-3.5 or above
- SDL2 or above for examples

## Quick Start

To use the VAD via a script tag in the browser, include the following script tags:

```cpp
#include <silero_vad/vad.hpp>

silero_vad::SileroVAD vad("model/silero_vad.onnx");
std::vector<float> input_wav(vad.GetFrameSamples(), 0.0f);
bool running = true;

while(running){
  //Fill input_wav with audio samples
  bool b = vad.Detect(input_wav);
  if(b){
    //Detect voice activity in this frame
  }else{
    //No voice activity in this frame
  }
}
```

## Build Examples
```bash
brew install onnxruntime
brew install sdl2

cd {silero_vad_cpp}
# Build the example
mkdir build
cd build
cmake ..
make
cd ..
# Run the example
./build/silero_vad_cpp_example

```

## References

<a id="1">[1]</a>
Silero Team. (2021).
Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier.
GitHub, GitHub repository, https://github.com/snakers4/silero-vad, hello@silero.ai.
