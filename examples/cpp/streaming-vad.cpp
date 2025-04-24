/*
g++ streaming-vad.cpp \
    -std=c++17 \
    -I /opt/homebrew/Cellar/onnxruntime/1.21.0/include \
    -L /opt/homebrew/Cellar/onnxruntime/1.21.0/lib \
    -I /opt/homebrew/include \
    -L /opt/homebrew/lib \
    -lonnxruntime \
    -lportaudio \
    `pkg-config --cflags --libs opencv4` \
    -pthread \
    -Wl,-rpath,/opt/homebrew/Cellar/onnxruntime/1.21.0/lib \
    -o streaming-vad
./streaming-vad

// Visualize
brew install gnuplot
brew install boost
curl -O https://raw.githubusercontent.com/dstahlke/gnuplot-iostream/master/gnuplot-iostream.h
*/

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <limits>
#include <chrono>
#include <iomanip>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cstdarg>
#include <cmath>    // for std::rint
#if __cplusplus < 201703L
#include <memory>
#endif

//#define __DEBUG_SPEECH_PROB___
#ifdef __APPLE__
    #include "onnxruntime/onnxruntime_cxx_api.h"
#elif __linux__
    #include "onnxruntime_cxx_api.h"
#endif
#include "wav.h" // For reading WAV files
// Visualize
#include "gnuplot-iostream.h"
// PortAudio (Audio Input/Output)
#include <portaudio.h>
#include <fstream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <thread>

// timestamp_t class: stores the start and end (in samples) of a speech segment.
class timestamp_t {
public:
    int start;
    int end;

    timestamp_t(int start = -1, int end = -1)
        : start(start), end(end) { }

    timestamp_t& operator=(const timestamp_t& a) {
        start = a.start;
        end = a.end;
        return *this;
    }

    bool operator==(const timestamp_t& a) const {
        return (start == a.start && end == a.end);
    }

    // Returns a formatted string of the timestamp.
    std::string c_str() const {
        return format("{start:%08d, end:%08d}", start, end);
    }
private:
    // Helper function for formatting.
    std::string format(const char* fmt, ...) const {
        char buf[256];
        va_list args;
        va_start(args, fmt);
        const auto r = std::vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        if (r < 0)
            return {};
        const size_t len = r;
        if (len < sizeof(buf))
            return std::string(buf, len);
#if __cplusplus >= 201703L
        std::string s(len, '\0');
        va_start(args, fmt);
        std::vsnprintf(s.data(), len + 1, fmt, args);
        va_end(args);
        return s;
#else
        auto vbuf = std::unique_ptr<char[]>(new char[len + 1]);
        va_start(args, fmt);
        std::vsnprintf(vbuf.get(), len + 1, fmt, args);
        va_end(args);
        return std::string(vbuf.get(), len);
#endif
    }
};

// 클래스 외부에 save_probability_data 함수 정의
void save_probability_data(const std::vector<float>& probs, int window_size_samples, int sample_rate) {
    std::ofstream file("probability.txt");
    for (size_t i = 0; i < probs.size(); i++) {
        double time = static_cast<double>(i * window_size_samples) / sample_rate;
        file << time << " " << probs[i] << "\n";
    }
    file.close();
}

// VadIterator class: uses ONNX Runtime to detect speech segments.
class VadIterator {
public:
    // window_size_samples를 public으로만 선언
    int window_size_samples;
    
    // probability 벡터를 반환하는 getter 함수를 public으로
    const std::vector<float>& get_speech_probabilities() const {
        return speech_probs;
    }

    // Add getter for threshold
    float get_threshold() const {
        return threshold;
    }

    // Process the entire audio input.
    void process(const std::vector<float>& input_wav) {
        reset_states();
        audio_length_samples = static_cast<int>(input_wav.size());
        // Process audio in chunks of window_size_samples (e.g., 512 samples)
        for (size_t j = 0; j < static_cast<size_t>(audio_length_samples); j += static_cast<size_t>(window_size_samples)) {
            if (j + static_cast<size_t>(window_size_samples) > static_cast<size_t>(audio_length_samples))
                break;
            std::vector<float> chunk(&input_wav[j], &input_wav[j] + window_size_samples);
            predict(chunk);
        }
        if (current_speech.start >= 0) {
            current_speech.end = audio_length_samples;
            speeches.push_back(current_speech);
            current_speech = timestamp_t();
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
        }
    }

    // Returns the detected speech timestamps.
    const std::vector<timestamp_t> get_speech_timestamps() const {
        return speeches;
    }

    // Public method to reset the internal state.
    void reset() {
        reset_states();
    }

public:
    // Constructor
    VadIterator(const std::string ModelPath,
        int Sample_rate = 16000, int windows_frame_size = 32,
        float Threshold = 0.75, int min_silence_duration_ms = 100,
        int speech_pad_ms = 30, int min_speech_duration_ms = 250,
        float max_speech_duration_s = std::numeric_limits<float>::infinity())
        : sample_rate(Sample_rate), 
          threshold(Threshold), 
          speech_pad_samples(speech_pad_ms), 
          prev_end(0)
    {
        sr_per_ms = sample_rate / 1000;  // e.g., 16000 / 1000 = 16
        window_size_samples = windows_frame_size * sr_per_ms; // e.g., 32ms * 16 = 512 samples
        effective_window_size = window_size_samples + context_samples; // e.g., 512 + 64 = 576 samples
        input_node_dims[0] = 1;
        input_node_dims[1] = effective_window_size;
        _state.resize(size_state);
        sr.resize(1);
        sr[0] = sample_rate;
        _context.assign(context_samples, 0.0f);
        min_speech_samples = sr_per_ms * min_speech_duration_ms;
        max_speech_samples = (sample_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples);
        min_silence_samples = sr_per_ms * min_silence_duration_ms;
        min_silence_samples_at_max_speech = sr_per_ms * 98;
        init_onnx_model(ModelPath);
    }

    // Initializes threading settings.
    void init_engine_threads(int inter_threads, int intra_threads) {
        session_options.SetIntraOpNumThreads(intra_threads);
        session_options.SetInterOpNumThreads(inter_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

    // Resets internal state (_state, _context, etc.)
    void reset_states() {
        std::memset(_state.data(), 0, _state.size() * sizeof(float));
        triggered = false;
        temp_end = 0;
        current_sample = 0;
        prev_end = next_start = 0;
        speeches.clear();
        current_speech = timestamp_t();
        std::fill(_context.begin(), _context.end(), 0.0f);
        speech_probs.clear();  // probability 벡터 초기화
    }

    // Inference: runs inference on one chunk of input data.
    // data_chunk is expected to have window_size_samples samples.
    void predict(const std::vector<float>& data_chunk) {
        // Build new input: first context_samples from _context, followed by the current chunk (window_size_samples).
        std::vector<float> new_data(effective_window_size, 0.0f);
        std::copy(_context.begin(), _context.end(), new_data.begin());
        std::copy(data_chunk.begin(), data_chunk.end(), new_data.begin() + context_samples);
        input = new_data;

        // Create input tensor (input_node_dims[1] is already set to effective_window_size).
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), input_node_dims, 2);
        Ort::Value state_ort = Ort::Value::CreateTensor<float>(
            memory_info, _state.data(), _state.size(), state_node_dims, 3);
        Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, sr.data(), sr.size(), sr_node_dims, 1);
        ort_inputs.clear();
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(state_ort));
        ort_inputs.emplace_back(std::move(sr_ort));

        // Run inference.
        ort_outputs = session->Run(
            Ort::RunOptions{ nullptr },
            input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
            output_node_names.data(), output_node_names.size());

        // Speech Probability
        float speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
        speech_probs.push_back(speech_prob);  // probability 저장
        // State
        float* stateN = ort_outputs[1].GetTensorMutableData<float>();
        std::memcpy(_state.data(), stateN, size_state * sizeof(float));
        current_sample += static_cast<unsigned int>(window_size_samples); // Advance by the original window size.

        // If speech is detected (probability >= threshold)
        if (speech_prob >= threshold) {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples;
            printf("{ start: %.3f s (%.3f) %08d}\n", 1.0f * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif
            if (temp_end != 0) {
                temp_end = 0;
                if (next_start < prev_end)
                    next_start = current_sample - window_size_samples;
            }
            if (!triggered) {
                triggered = true;
                current_speech.start = current_sample - window_size_samples;
            }
            // Update context: copy the last context_samples from new_data.
            std::copy(new_data.end() - context_samples, new_data.end(), _context.begin());
            return;
        }

        // If the speech segment becomes too long.
        if (triggered && ((current_sample - current_speech.start) > max_speech_samples)) {
            if (prev_end > 0) {
                current_speech.end = prev_end;
                speeches.push_back(current_speech);
                current_speech = timestamp_t();
                if (next_start < prev_end)
                    triggered = false;
                else
                    current_speech.start = next_start;
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
            }
            else {
                current_speech.end = current_sample;
                speeches.push_back(current_speech);
                current_speech = timestamp_t();
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
            }
            std::copy(new_data.end() - context_samples, new_data.end(), _context.begin());
            return;
        }
        
        if ((speech_prob >= (threshold - 0.15)) && (speech_prob < threshold)) {
            // When the speech probability temporarily drops but is still in speech, update context without changing state.
            std::copy(new_data.end() - context_samples, new_data.end(), _context.begin());
            return;
        }

        if (speech_prob < (threshold - 0.15)) {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples - speech_pad_samples;
            printf("{ end: %.3f s (%.3f) %08d}\n", 1.0f * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif
            if (triggered) {
                if (temp_end == 0)
                    temp_end = current_sample;
                if (current_sample - temp_end > min_silence_samples_at_max_speech)
                    prev_end = temp_end;
                if ((current_sample - temp_end) >= min_silence_samples) {
                    current_speech.end = temp_end;
                    if (current_speech.end - current_speech.start > min_speech_samples) {
                        speeches.push_back(current_speech);
                        current_speech = timestamp_t();
                        prev_end = 0;
                        next_start = 0;
                        temp_end = 0;
                        triggered = false;
                    }
                }
            }
            std::copy(new_data.end() - context_samples, new_data.end(), _context.begin());
            return;
        }
    }

    // void init_onnx_model(const std::wstring& model_path) {
    //     init_engine_threads(1, 1);
    //     session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
    // }
    void init_onnx_model(const std::string& model_path) {
        init_engine_threads(1, 1);
        session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
    }

private:
    // ONNX Runtime resources
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    // Context-related additions
    const int context_samples = 64;
    std::vector<float> _context;
    
    // Effective window size = window_size_samples + context_samples
    int effective_window_size;

    // Additional declaration: samples per millisecond
    int sr_per_ms;

    // ONNX Runtime input/output buffers
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char*> input_node_names = { "input", "state", "sr" };
    std::vector<float> input;
    unsigned int size_state = 2 * 1 * 128;
    std::vector<float> _state;
    std::vector<int64_t> sr;
    int64_t input_node_dims[2] = {};
    const int64_t state_node_dims[3] = { 2, 1, 128 };
    const int64_t sr_node_dims[1] = { 1 };
    std::vector<Ort::Value> ort_outputs;
    std::vector<const char*> output_node_names = { "output", "stateN" };

    // Model configuration parameters
    int sample_rate;
    float threshold;
    int min_silence_samples;
    int min_silence_samples_at_max_speech;
    int min_speech_samples;
    float max_speech_samples;
    int speech_pad_samples;
    int audio_length_samples;

    // State management
    bool triggered = false;
    unsigned int temp_end = 0;
    unsigned int current_sample = 0;
    int prev_end;
    int next_start = 0;
    std::vector<timestamp_t> speeches;
    timestamp_t current_speech;

    std::vector<float> speech_probs;  // probability 저장용 벡터 추가
};

void save_waveform_data(const std::vector<float>& audio_data, int sample_rate) {
    std::ofstream file("waveform.txt");
    for (size_t i = 0; i < audio_data.size(); i++) {
        double time = static_cast<double>(i) / sample_rate;
        file << time << " " << audio_data[i] << "\n";
    }
    file.close();
}

void save_vad_segments(const std::vector<timestamp_t>& stamps, int sample_rate) {
    std::ofstream file("vad_segments.txt");
    for (const auto& stamp : stamps) {
        double start_time = static_cast<double>(stamp.start) / sample_rate;
        double end_time = static_cast<double>(stamp.end) / sample_rate;
        // VAD 구간을 박스로 그리기 위해 y값 범위 설정
        file << start_time << " -1\n";
        file << start_time << " 1\n";
        file << end_time << " 1\n";
        file << end_time << " -1\n";
        file << start_time << " -1\n\n"; // 빈 줄로 구간 구분
    }
    file.close();
}

// Gnuplot 스크립트 수정
void create_gnuplot_script(float threshold) {
    std::ofstream script("plot.gnu");
    script << "set terminal png size 1200,900\n";  // 세로 크기 증가
    script << "set output 'vad_visualization.png'\n";
    script << "set multiplot layout 2,1\n";  // 2개의 그래프를 위한 레이아웃
    
    // 첫 번째 그래프: 오디오 파형과 VAD 구간
    script << "set title 'Audio Waveform with VAD Segments'\n";
    script << "set xlabel 'Time (seconds)'\n";
    script << "set ylabel 'Amplitude'\n";
    script << "set grid\n";
    script << "set style fill transparent solid 0.2\n";
    script << "plot 'waveform.txt' using 1:2 with lines title 'Audio' lt rgb 'blue',\\\n";
    script << "     'vad_segments.txt' using 1:2 with filledcurves title 'VAD segments' lt rgb 'red'\n";
    
    // 두 번째 그래프: Speech Probability
    script << "set title 'Speech Probability Over Time'\n";
    script << "set xlabel 'Time (seconds)'\n";
    script << "set ylabel 'Probability'\n";
    script << "set yrange [0:1]\n";  // probability는 0~1 사이
    script << "set grid\n";
    script << "plot 'probability.txt' using 1:2 with lines title 'Speech Probability' lt rgb 'green',\\\n";
    script << "     " << threshold << " with lines title 'Threshold' lt rgb 'red' dashtype 2\n";  // threshold 선 추가
    
    script << "unset multiplot\n";
    script.close();
}

// visualize_vad_results 함수 수정
void visualize_vad_results(const std::vector<float>& audio_data,
                          const std::vector<timestamp_t>& stamps,
                          const std::vector<float>& probs,
                          int window_size_samples,
                          int sample_rate,
                          float threshold) {
    save_waveform_data(audio_data, sample_rate);
    save_vad_segments(stamps, sample_rate);
    save_probability_data(probs, window_size_samples, sample_rate);
    
    create_gnuplot_script(threshold);
    system("gnuplot plot.gnu");
    
    std::cout << "Visualization saved as 'vad_visualization.png'" << std::endl;
}

// StreamData 구조체에 뮤텍스 추가
struct StreamData {
    VadIterator* vad;
    std::vector<float> buffer;
    std::vector<float> display_probs;
    std::vector<float> display_audio;
    float threshold;
    cv::Mat display_mat;
    static const int DISPLAY_WIDTH = 800;
    static const int DISPLAY_HEIGHT = 600;
    std::mutex mtx;  // 데이터 접근을 위한 뮤텍스 추가
    bool should_update = false;  // 디스플레이 업데이트 필요 여부
};

// OpenCV 시각화 함수 수정
void updateDisplay(StreamData* data) {
    std::lock_guard<std::mutex> lock(data->mtx);  // 뮤텍스 잠금
    
    if (!data->should_update) return;  // 업데이트가 필요없으면 반환
    
    try {
        // 디스플레이 초기화
        data->display_mat = cv::Mat::zeros(data->DISPLAY_HEIGHT, data->DISPLAY_WIDTH, CV_8UC3);
        
        // 오디오 파형 그리기 (위쪽 절반)
        int audio_height = data->DISPLAY_HEIGHT / 2;
        int mid_line = audio_height / 2;
        
        // 오디오 데이터 정규화 및 그리기
        if (!data->display_audio.empty()) {
            float max_amp = *std::max_element(data->display_audio.begin(), data->display_audio.end(),
                [](float a, float b) { return std::abs(a) < std::abs(b); });
            max_amp = std::max(max_amp, 1.0f);  // 0으로 나누기 방지
            
            for (size_t i = 1; i < data->display_audio.size(); ++i) {
                int x1 = (i - 1) * data->DISPLAY_WIDTH / data->display_audio.size();
                int x2 = i * data->DISPLAY_WIDTH / data->display_audio.size();
                int y1 = mid_line + (data->display_audio[i-1] / max_amp * (audio_height/2));
                int y2 = mid_line + (data->display_audio[i] / max_amp * (audio_height/2));
                
                cv::line(data->display_mat, 
                        cv::Point(x1, y1), 
                        cv::Point(x2, y2),
                        cv::Scalar(255, 255, 0),  // 청록색
                        1);
            }
        }
        
        // VAD 확률 그리기 (아래쪽 절반)
        int prob_height = data->DISPLAY_HEIGHT / 2;
        int prob_offset = audio_height;
        
        // 임계값 선 그리기
        int threshold_y = prob_offset + prob_height * (1 - data->threshold);
        cv::line(data->display_mat,
                cv::Point(0, threshold_y),
                cv::Point(data->DISPLAY_WIDTH, threshold_y),
                cv::Scalar(0, 0, 255),  // 빨간색
                1);
        
        // 임계값 레이블 추가
        std::stringstream ss;
        ss << "Threshold (" << std::fixed << std::setprecision(2) << data->threshold << ")";
        cv::putText(data->display_mat, 
                   ss.str(),
                   cv::Point(data->DISPLAY_WIDTH - 200, threshold_y - 10),  // 선 위에 텍스트 배치
                   cv::FONT_HERSHEY_SIMPLEX, 
                   0.6,  // 폰트 크기
                   cv::Scalar(0, 0, 255),  // 빨간색 (선과 동일)
                   2);
        
        // 확률값 그리기
        if (!data->display_probs.empty()) {
            for (size_t i = 1; i < data->display_probs.size(); ++i) {
                int x1 = (i - 1) * data->DISPLAY_WIDTH / data->display_probs.size();
                int x2 = i * data->DISPLAY_WIDTH / data->display_probs.size();
                int y1 = prob_offset + prob_height * (1 - data->display_probs[i-1]);
                int y2 = prob_offset + prob_height * (1 - data->display_probs[i]);
                
                cv::line(data->display_mat,
                        cv::Point(x1, y1),
                        cv::Point(x2, y2),
                        cv::Scalar(0, 255, 0),  // 녹색
                        2);
            }
        }
        
        // 텍스트 추가
        cv::putText(data->display_mat, "Audio Waveform",
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 255), 2);
                    
        cv::putText(data->display_mat, "Speech Probability",
                    cv::Point(10, audio_height + 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 255), 2);
        
        // 화면 표시
        cv::imshow("Real-time VAD", data->display_mat);
        data->should_update = false;  // 업데이트 완료
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
    }
}

// PortAudio 콜백 함수 수정
static int audioCallback(const void* inputBuffer, void* outputBuffer,
                        unsigned long framesPerBuffer,
                        const PaStreamCallbackTimeInfo* timeInfo,
                        PaStreamCallbackFlags statusFlags,
                        void* userData) {
    StreamData* data = (StreamData*)userData;
    const float* in = (const float*)inputBuffer;
    
    if (!in) {
        std::cerr << "No input buffer!" << std::endl;
        return paContinue;
    }
    
    std::vector<float> chunk(in, in + framesPerBuffer);
    
    {
        std::lock_guard<std::mutex> lock(data->mtx);  // 뮤텍스 잠금
        
        data->vad->predict(chunk);
        
        // 디스플레이 버퍼 업데이트
        data->display_audio.insert(data->display_audio.end(), chunk.begin(), chunk.end());
        if (data->display_audio.size() > 16000 * 2) {
            data->display_audio.erase(data->display_audio.begin(), 
                                    data->display_audio.begin() + chunk.size());
        }
        
        // VAD 확률 업데이트
        float prob = data->vad->get_speech_probabilities().back();
        data->display_probs.push_back(prob);
        if (data->display_probs.size() > 200) {
            data->display_probs.erase(data->display_probs.begin());
        }
        
        data->should_update = true;  // 업데이트 필요 표시
    }
    
    return paContinue;
}

int main() {
    // PortAudio 초기화
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio 초기화 실패: " << Pa_GetErrorText(err) << std::endl;
        return 1;
    }
    
    // ONNX 모델 초기화
    std::string model_path = "./model/silero_vad.onnx";
    VadIterator vad(model_path);
    
    // OpenCV 창 생성 및 확인
    cv::namedWindow("Real-time VAD", cv::WINDOW_AUTOSIZE);
    if (cv::getWindowProperty("Real-time VAD", cv::WND_PROP_VISIBLE) < 0) {
        std::cerr << "OpenCV 창 생성 실패!" << std::endl;
        Pa_Terminate();
        return 1;
    }
    std::cout << "OpenCV 창 생성 성공" << std::endl;
    
    // StreamData 초기화
    StreamData data;
    data.vad = &vad;
    data.threshold = vad.get_threshold();
    data.display_mat = cv::Mat::zeros(data.DISPLAY_HEIGHT, data.DISPLAY_WIDTH, CV_8UC3);
    
    // 입력 장치 정보 출력
    int numDevices = Pa_GetDeviceCount();
    std::cout << "사용 가능한 오디오 장치 수: " << numDevices << std::endl;
    
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(Pa_GetDefaultInputDevice());
    if (deviceInfo) {
        std::cout << "기본 입력 장치: " << deviceInfo->name << std::endl;
    }
    
    // 오디오 스트림 설정
    PaStreamParameters inputParameters;
    inputParameters.device = Pa_GetDefaultInputDevice();
    if (inputParameters.device == paNoDevice) {
        std::cerr << "입력 장치를 찾을 수 없습니다!" << std::endl;
        Pa_Terminate();
        return 1;
    }
    inputParameters.channelCount = 1;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = deviceInfo->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;
    
    // stream 변수 선언 추가
    PaStream* stream = nullptr;
    
    err = Pa_OpenStream(&stream,
                       &inputParameters,
                       nullptr,  // 출력 파라미터 없음
                       16000,    // 샘플레이트
                       512,      // 프레임 버퍼 크기
                       paClipOff,// 클리핑 끄기
                       audioCallback,
                       &data);
    
    if (err != paNoError) {
        std::cerr << "스트림 열기 실패: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return 1;
    }
    
    // 스트림 시작
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "스트림 시작 실패: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }
    
    std::cout << "실시간 VAD 시작... 'q'를 누르면 종료됩니다." << std::endl;
    
    // 메인 루프 수정
    while (true) {
        updateDisplay(&data);  // 메인 스레드에서 디스플레이 업데이트
        
        char key = cv::waitKey(30);
        if (key == 'q' || key == 'Q') {
            break;
        }
        
        if (cv::getWindowProperty("Real-time VAD", cv::WND_PROP_VISIBLE) < 1) {
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // 정리
    std::cout << "프로그램 종료 중..." << std::endl;
    cv::destroyAllWindows();
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    
    return 0;
}
