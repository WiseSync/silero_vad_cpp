#include <silero_vad/vad.hpp>

#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <limits>
#include <chrono>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <cstdio>
#include <cstdarg>
#include <cassert>
// #ifdef __APPLE__
// #include <coreml_provider_factory.h>
// #endif

using namespace Ort;
using namespace silero_vad;
// default + parameterized constructor
timestamp_t::timestamp_t(int start, int end)
    : start(start), end(end){};

// assignment operator modifies object, therefore non-const
timestamp_t::timestamp_t(const timestamp_t &a)
{
    start = a.start;
    end = a.end;
};

timestamp_t& timestamp_t::operator=(const timestamp_t& a){
    start = a.start;
    end = a.end;
    return *this;

}

// equality comparison. doesn't modify object. therefore const.
bool timestamp_t::operator==(const timestamp_t &a) const
{
    return (start == a.start && end == a.end);
};
std::string timestamp_t::c_str()
{
    // return std::format("timestamp {:08d}, {:08d}", start, end);
    return format("{start:%08d,end:%08d}", start, end);
};

std::string timestamp_t::format(const char *fmt, ...)
{
    char buf[256];

    va_list args;
    va_start(args, fmt);
    const auto r = std::vsnprintf(buf, sizeof buf, fmt, args);
    va_end(args);

    if (r < 0)
        // conversion failed
        return {};

    const size_t len = r;
    if (len < sizeof buf)
        // we fit in the buffer
        return {buf, len};

#if __cplusplus >= 201703L
    // C++17: Create a string and write to its underlying array
    std::string s(len, '\0');
    va_start(args, fmt);
    std::vsnprintf(s.data(), len + 1, fmt, args);
    va_end(args);

    return s;
#else
    // C++11 or C++14: We need to allocate scratch memory
    auto vbuf = std::unique_ptr<char[]>(new char[len + 1]);
    va_start(args, fmt);
    std::vsnprintf(vbuf.get(), len + 1, fmt, args);
    va_end(args);

    return {vbuf.get(), len};
#endif
};

void SileroVAD::init_engine_threads(int inter_threads, int intra_threads)
{
    // The method should be called in each thread/proc in multi-thread/proc work
    session_options.SetIntraOpNumThreads(intra_threads);
    session_options.SetInterOpNumThreads(inter_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
};

void SileroVAD::init_onnx_model(const std::string &model_path)
{
    // Init threads = 1 for
    init_engine_threads(1, 1);
    /*
    #ifdef __APPLE__
    uint32_t coreml_flags = 0;
    coreml_flags |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;

    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags));
    #endif*/
    // Load model
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);


};

void SileroVAD::Reset()
{
    // Call reset before each audio start
    std::memset(_h.data(), 0.0f, _h.size() * sizeof(float));
    std::memset(_c.data(), 0.0f, _c.size() * sizeof(float));
    triggered = false;
    temp_end = 0;
    current_sample = 0;

    prev_end = next_start = 0;

    speeches.clear();
    current_speech = timestamp_t();
};

void SileroVAD::predict(const std::vector<float> &data)
{
    // Infer
    // Create ort tensors
    input.assign(data.begin(), data.end());
    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
        memory_info, input.data(), input.size(), input_node_dims, 2);
    Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
        memory_info, sr.data(), sr.size(), sr_node_dims, 1);
    Ort::Value h_ort = Ort::Value::CreateTensor<float>(
        memory_info, _h.data(), _h.size(), hc_node_dims, 3);
    Ort::Value c_ort = Ort::Value::CreateTensor<float>(
        memory_info, _c.data(), _c.size(), hc_node_dims, 3);

    // Clear and add inputs
    ort_inputs.clear();
    ort_inputs.emplace_back(std::move(input_ort));
    ort_inputs.emplace_back(std::move(sr_ort));
    ort_inputs.emplace_back(std::move(h_ort));
    ort_inputs.emplace_back(std::move(c_ort));

    // Infer
    ort_outputs = session->Run(
        Ort::RunOptions{nullptr},
        input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
        output_node_names.data(), output_node_names.size());

    // Output probability & update h,c recursively
    float speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
    float *hn = ort_outputs[1].GetTensorMutableData<float>();
    std::memcpy(_h.data(), hn, size_hc * sizeof(float));
    float *cn = ort_outputs[2].GetTensorMutableData<float>();
    std::memcpy(_c.data(), cn, size_hc * sizeof(float));

    // Push forward sample index
    current_sample += window_size_samples;

    // Reset temp_end when > threshold
    if ((speech_prob >= threshold))
    {
#ifdef __DEBUG_SPEECH_PROB___
        float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
        printf("{    start: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
        if (temp_end != 0)
        {
            temp_end = 0;
            if (next_start < prev_end)
                next_start = current_sample - window_size_samples;
        }
        if (triggered == false)
        {
            triggered = true;

            current_speech.start = current_sample - window_size_samples;
        }
        return;
    }

    if (
        (triggered == true) && ((current_sample - current_speech.start) > max_speech_samples))
    {
        if (prev_end > 0)
        {
            current_speech.end = prev_end;
            speeches.push_back(current_speech);
            current_speech = timestamp_t();

            // previously reached silence(< neg_thres) and is still not speech(< thres)
            if (next_start < prev_end)
                triggered = false;
            else
            {
                current_speech.start = next_start;
            }
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
        }
        else
        {
            current_speech.end = current_sample;
            speeches.push_back(current_speech);
            current_speech = timestamp_t();
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
        }
        return;
    }
    if ((speech_prob >= (threshold - 0.15)) && (speech_prob < threshold))
    {
        if (triggered)
        {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
            printf("{ speeking: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
        }
        else
        {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
            printf("{  silence: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
        }
        return;
    }

    // 4) End
    if ((speech_prob < (threshold - 0.15)))
    {
#ifdef __DEBUG_SPEECH_PROB___
        float speech = current_sample - window_size_samples - speech_pad_samples; // minus window_size_samples to get precise start time point.
        printf("{      end: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
        if (triggered == true)
        {
            if (temp_end == 0)
            {
                temp_end = current_sample;
            }
            if (current_sample - temp_end > min_silence_samples_at_max_speech)
                prev_end = temp_end;
            // a. silence < min_slience_samples, continue speaking
            if ((current_sample - temp_end) < min_silence_samples)
            {
            }
            // b. silence >= min_slience_samples, end speaking
            else
            {
                current_speech.end = temp_end;
                if (current_speech.end - current_speech.start >static_cast<int32_t>(min_speech_samples))
                {
                    speeches.push_back(current_speech);
                    current_speech = timestamp_t();
                    prev_end = 0;
                    next_start = 0;
                    temp_end = 0;
                    triggered = false;
                }
            }
        }
        else
        {
            // may first windows see end state.
        }
        return;
    }
};

bool SileroVAD::Detect(const std::vector<float> &input_wav){
    assert(input_wav.size() == window_size_samples&&"input_wav.size() != window_size_samples");
    audio_length_samples += input_wav.size();

    //auto start = std::chrono::high_resolution_clock::now();
    predict(input_wav);
    //std::cout<<"predict time: "<<std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()<<"us"<<std::endl;

    if (current_speech.start >= 0)
    {
        current_speech = timestamp_t();
        prev_end = 0;
        next_start = 0;
        temp_end = 0;
        triggered = false;
        return true;
    }else{
        return false;
    }
}

 std::size_t SileroVAD::GetFrameSamples() const{
    return window_size_samples;
 }

 std::chrono::milliseconds SileroVAD::GetFrameDuration() const{
    return std::chrono::milliseconds(window_size_samples/sr_per_ms);
 }


uint32_t SileroVAD::GetSampleRate() const{
    return sample_rate;
}

SileroVAD::SileroVAD(const std::string& ModelPath,
        SampleRate Sample_rate, FrameMS window_frame_ms,
        float Threshold , const std::chrono::milliseconds& min_silence_duration_ms,
        const std::chrono::milliseconds& speech_pad_ms, 
        const std::chrono::milliseconds& min_speech_duration_ms,
        const std::chrono::seconds&  max_speech_duration_s)
{
    init_onnx_model(ModelPath);
    threshold = Threshold;
    sample_rate = static_cast<uint32_t>(Sample_rate);
    sr_per_ms = sample_rate / 1000;

    window_size_samples = static_cast<uint32_t>(window_frame_ms) * sr_per_ms;

    min_speech_samples = sr_per_ms * min_speech_duration_ms.count();
    speech_pad_samples = sr_per_ms * speech_pad_ms.count();

    max_speech_samples = (sample_rate * max_speech_duration_s.count() - window_size_samples - 2 * speech_pad_samples);

    min_silence_samples = sr_per_ms * min_silence_duration_ms.count();
    min_silence_samples_at_max_speech = sr_per_ms * 98;

    input.resize(window_size_samples);
    input_node_dims[0] = 1;
    input_node_dims[1] = window_size_samples;

    _h.resize(size_hc);
    _c.resize(size_hc);
    sr.resize(1);
    sr[0] = sample_rate;
};
