#ifndef _FCITX5_VOICE_VOICE_H_
#define _FCITX5_VOICE_VOICE_H_

#include "common-sdl.h"
#include "whisper.h"
#include <atomic>
#include <fcitx-utils/i18n.h>
#include <fcitx/addonfactory.h>
#include <fcitx/addonmanager.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputmethodengine.h>
#include <fcitx/instance.h>
#include <thread>

namespace fcitx {

class VoiceEngine final : public InputMethodEngine {
  public:
    VoiceEngine(Instance *instance);
    ~VoiceEngine();

    void keyEvent(const InputMethodEntry &entry, KeyEvent &keyEvent) override;
    void reset(const InputMethodEntry &, InputContextEvent &event) override;
    void activate(const InputMethodEntry &entry,
                  InputContextEvent &event) override;
    void deactivate(const InputMethodEntry &entry,
                    InputContextEvent &event) override;

  private:
    Instance *instance_;
    std::unique_ptr<HandlerTableEntry<EventHandler>> eventHandler_;
    std::string previousIm;
    audio_async audio;
    struct whisper_context *ctx;
    std::thread thread_;
    bool is_running = false;
    bool has_permission = false;
    std::atomic<bool> quit = false;
};

class VoiceFactory : public AddonFactory {
  public:
    AddonInstance *create(AddonManager *manager) override {
        registerDomain("fcitx5-voice", FCITX_INSTALL_LOCALEDIR);
        return new VoiceEngine(manager->instance());
    }
};
} // namespace fcitx

#endif
