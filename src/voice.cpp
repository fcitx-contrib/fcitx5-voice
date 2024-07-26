#include "voice.h"
#include "common.h"
#include "platform.h"
#include <chrono>
#include <fcitx/inputpanel.h>

namespace fcitx {

VoiceEngine::VoiceEngine(Instance *instance)
    : instance_(instance), audio(10000) {
    // This asks for permission when undetermined, but will still succeed even
    // if user rejects.
    if (!audio.init(-1, WHISPER_SAMPLE_RATE)) {
        FCITX_ERROR() << "audio.init() failed.";
        return;
    }

    eventHandler_ = instance_->watchEvent(
        EventType::InputContextKeyEvent, EventWatcherPhase::PreInputMethod,
        [this](Event &event) {
            auto &keyEvent = static_cast<KeyEvent &>(event);
            if (keyEvent.isRelease() ||
                instance_->currentInputMethod() == "voice") {
                return;
            }
            if (keyEvent.key().check(Key(FcitxKey_Shift_R))) {
                previousIm = instance_->currentInputMethod();
                instance_->setCurrentInputMethod("voice");
                return keyEvent.filterAndAccept();
            }
        });

    has_permission = checkPermission();
    if (!has_permission) {
        // Don't throw as it will remove the IM.
        return;
    }
    thread_ = std::thread([this] {
        const int n_samples_step = (1e-3 * 3000) * WHISPER_SAMPLE_RATE;
        const int n_samples_len = (1e-3 * 10000) * WHISPER_SAMPLE_RATE;
        const int n_samples_keep = (1e-3 * 200) * WHISPER_SAMPLE_RATE;
        const int n_samples_30s = (1e-3 * 30000.0) * WHISPER_SAMPLE_RATE;

        struct whisper_context_params cparams =
            whisper_context_default_params();
        // TODO: no hard coded path
        ctx = whisper_init_from_file_with_params(
            "/Users/liumeo/github/whisper.cpp/models/ggml-base.en.bin",
            cparams);

        std::vector<float> pcmf32(n_samples_30s, 0.0f);
        std::vector<float> pcmf32_old;
        std::vector<float> pcmf32_new(n_samples_30s, 0.0f);
        int n_iter = 0;
        auto t_last = std::chrono::high_resolution_clock::now();
        const auto t_start = t_last;

        while (!quit.load()) {
            if (!is_running) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            while (!quit.load()) {
                audio.get(3000, pcmf32_new);

                if ((int)pcmf32_new.size() > 2 * n_samples_step) {
                    fprintf(stderr,
                            "\n\n%s: WARNING: cannot process audio fast "
                            "enough, dropping audio ...\n\n",
                            __func__);
                    audio.clear();
                    continue;
                }

                if ((int)pcmf32_new.size() >= n_samples_step) {
                    audio.clear();
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            const int n_samples_new = pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            const int n_samples_take = std::min(
                (int)pcmf32_old.size(),
                std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            // printf("processing: take = %d, new = %d, old = %d\n",
            // n_samples_take, n_samples_new, (int) pcmf32_old.size());

            pcmf32.resize(n_samples_new + n_samples_take);

            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }

            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(),
                   n_samples_new * sizeof(float));

            pcmf32_old = pcmf32;
            // {
            //     const auto t_now  =
            //     std::chrono::high_resolution_clock::now(); const auto t_diff
            //     = std::chrono::duration_cast<std::chrono::milliseconds>(t_now
            //     - t_last).count();

            //     if (t_diff < 2000) {
            //         std::this_thread::sleep_for(std::chrono::milliseconds(100));

            //         continue;
            //     }

            //     audio.get(2000, pcmf32_new);

            //     if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, 0.6f,
            //     100.0f, false)) {
            //         audio.get(10000, pcmf32);
            //     } else {
            //         std::this_thread::sleep_for(std::chrono::milliseconds(100));

            //         continue;
            //     }

            //     t_last = t_now;
            // }

            {
                whisper_full_params wparams =
                    whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                wparams.single_segment = false; // TODO: vad
                wparams.max_tokens = 32;
                if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) !=
                    0) {
                    FCITX_ERROR() << "Failed to process audio.";
                    continue;
                }
                const int64_t t1 = (t_last - t_start).count() / 1000000;
                const int64_t t0 = std::max(0.0, t1 - pcmf32.size() * 1000.0 /
                                                          WHISPER_SAMPLE_RATE);

                // FCITX_INFO() << "Transcription " << n_iter << " START | t0 =
                // " << (int)t0 << " ms | t1 = " << (int)t1 <<" ms";
                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const char *text = whisper_full_get_segment_text(ctx, i);
                    FCITX_ERROR() << text;
                }
                // FCITX_INFO() << "### Transcription " << n_iter << " END";
            }
            ++n_iter;
        }
        audio.pause();
    });
}

VoiceEngine::~VoiceEngine() {
    quit.store(true);
    if (thread_.joinable()) {
        thread_.join();
    }
    whisper_free(ctx);
}

void VoiceEngine::reset(const InputMethodEntry &, InputContextEvent &event) {}

void VoiceEngine::keyEvent(const InputMethodEntry &, KeyEvent &keyEvent) {
    if (keyEvent.isRelease() && keyEvent.key().check(Key(FcitxKey_Shift_R))) {
        instance_->setCurrentInputMethod(previousIm);
        return keyEvent.filterAndAccept();
    }
}

void VoiceEngine::activate(const InputMethodEntry &entry,
                           InputContextEvent &event) {
    if (has_permission) {
        audio.resume();
        is_running = true;
    }
};
void VoiceEngine::deactivate(const InputMethodEntry &entry,
                             InputContextEvent &event) {
    if (has_permission) {
        is_running = false;
        audio.pause();
    }
};
} // namespace fcitx

FCITX_ADDON_FACTORY(fcitx::VoiceFactory);
