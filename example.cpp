#include <silero_vad/vad.hpp>
#include <SDL2/SDL.h>
#include <cassert>
#include <iostream>

using namespace silero_vad;
using namespace std;

int main(int32_t argc, char** argv) {
    SDL_Init(SDL_INIT_AUDIO);

    int32_t captureId;
    silero_vad::SileroVAD vad("model/silero_vad.onnx");
    
    SDL_AudioSpec capture_spec_requested;
    SDL_AudioSpec capture_spec_obtained;

    SDL_zero(capture_spec_requested);
    SDL_zero(capture_spec_obtained);

    capture_spec_requested.freq     = vad.GetSampleRate();
    capture_spec_requested.format   = AUDIO_F32;
    capture_spec_requested.channels = 1;
    capture_spec_requested.samples  = vad.GetFrameSamples();
    capture_spec_requested.callback = [](void * userdata, uint8_t * stream, int len) {
        SileroVAD* pVAD = reinterpret_cast<SileroVAD*>(userdata);
        static std::vector<float> input_wav(pVAD->GetFrameSamples(), 0.0f);
        static bool lastActivity = false;
        static std::chrono::milliseconds duration;
        bool b;
        
        assert((vad.GetFrameSamples()==len/sizeof(float))&&"Unexpected frame size");
        
        std::copy(reinterpret_cast<const float*>(stream), reinterpret_cast<const float*>(stream)+pVAD->GetFrameSamples(), input_wav.begin());
        b = pVAD->Detect(input_wav);
        if(b){
            //Detect voice activity in this frame
            if(lastActivity==true){
                duration+=pVAD->GetFrameDuration();
            }else{
                duration=pVAD->GetFrameDuration();
            }
        }else{
            //Detect silence in this frame
            if(lastActivity==true){
                std::cout<<"Detect voice activity: "<<duration.count()<<" ms"<<std::endl;
            } 
        }
        lastActivity=b;
    };
    capture_spec_requested.userdata = &vad;

    fprintf(stderr, "%s: attempt to open default capture device ...\n", __func__);
    captureId = SDL_OpenAudioDevice(nullptr, SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);

    if (!captureId) {
        fprintf(stderr, "%s: couldn't open an audio device for capture: %s!\n", __func__, SDL_GetError());

        captureId = 0;
        
        SDL_Quit();

        return 1;
    } else {
        fprintf(stderr, "%s: obtained spec for input device (SDL Id = %d):\n", __func__, captureId);
        fprintf(stderr, "%s:     - sample rate:       %d\n",                   __func__, capture_spec_obtained.freq);
        fprintf(stderr, "%s:     - format:            %d (required: %d)\n",    __func__, capture_spec_obtained.format,
                capture_spec_requested.format);
        fprintf(stderr, "%s:     - channels:          %d (required: %d)\n",    __func__, capture_spec_obtained.channels,
                capture_spec_requested.channels);
        fprintf(stderr, "%s:     - samples per frame: %d\n",                   __func__, capture_spec_obtained.samples);
    }

    SDL_PauseAudioDevice(captureId, 0);

    SDL_Event event;
    while (SDL_WaitEvent(&event)) {
        if(event.type==SDL_QUIT){
            break;
        }
    }

    SDL_PauseAudioDevice(captureId, 1);
    SDL_CloseAudioDevice(captureId);

    SDL_Quit();
    return 0;
}