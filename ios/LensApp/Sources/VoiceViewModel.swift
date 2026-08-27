import AVFoundation
import LensClient
import Speech
import SwiftUI

enum VoiceState: Equatable {
    case idle
    case requestingPermission
    case recording
    case thinking
    case speaking(String)
    case error(String)
}

@MainActor
final class VoiceViewModel: ObservableObject {
    @Published var voiceState: VoiceState = .idle
    @Published var transcript: String = ""
    @Published var answer: String = ""

    private let client: VoiceClient
    private let scanSessionId: String
    private let userId: String

    private var recognizer: SFSpeechRecognizer?
    private let audioEngine = AVAudioEngine()
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let synthesizer = AVSpeechSynthesizer()
    private var eventTask: Task<Void, Never>?

    init(scanSessionId: String, userId: String) {
        let base = ProcessInfo.processInfo.environment["LENS_API_URL"]
            ?? "http://localhost:8000"
        let key = ProcessInfo.processInfo.environment["LENS_API_KEY"] ?? ""
        self.client = VoiceClient(baseURL: URL(string: base)!, apiKey: key)
        self.scanSessionId = scanSessionId
        self.userId = userId
        self.recognizer = SFSpeechRecognizer(locale: Locale.current)
    }

    // MARK: - Session

    func openSession() async {
        do {
            try await client.connect(userId: userId)
            startEventLoop()
        } catch {
            voiceState = .error("Voice unavailable")
        }
    }

    func closeSession() async {
        eventTask?.cancel()
        await client.disconnect()
        stopAudio()
    }

    // MARK: - Record

    func startRecording() {
        guard voiceState == .idle else { return }
        voiceState = .requestingPermission
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            guard let self else { return }
            Task { @MainActor in
                guard status == .authorized else {
                    self.voiceState = .error("Speech recognition not allowed")
                    return
                }
                self.beginCapture()
            }
        }
    }

    func stopRecording() {
        guard case .recording = voiceState else { return }
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        voiceState = .thinking
    }

    // MARK: - Private

    private func beginCapture() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: .duckOthers)
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            voiceState = .error("Audio session failed")
            return
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        recognitionRequest = request

        let node = audioEngine.inputNode
        let fmt = node.outputFormat(forBus: 0)
        node.installTap(onBus: 0, bufferSize: 1024, format: fmt) { [weak self] buf, _ in
            self?.recognitionRequest?.append(buf)
        }

        do {
            try audioEngine.start()
        } catch {
            voiceState = .error("Microphone unavailable")
            return
        }

        transcript = ""
        voiceState = .recording

        recognitionTask = recognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            if let result {
                Task { @MainActor in
                    self.transcript = result.bestTranscription.formattedString
                }
            }
            if error != nil || result?.isFinal == true {
                let finalText = result?.bestTranscription.formattedString ?? ""
                Task { @MainActor in
                    if !finalText.isEmpty {
                        self.client.ask(text: finalText, scanSessionId: self.scanSessionId)
                    } else {
                        self.voiceState = .idle
                    }
                }
            }
        }
    }

    private func startEventLoop() {
        eventTask = Task { [weak self] in
            guard let self else { return }
            for await event in await self.client.events() {
                await MainActor.run {
                    switch event {
                    case .ack, .progress:
                        break
                    case .speak(let text):
                        self.answer = text
                        self.voiceState = .speaking(text)
                        self.speak(text)
                    case .done:
                        self.voiceState = .idle
                    case .error(let detail):
                        self.voiceState = .error(detail)
                    }
                }
            }
        }
    }

    private func speak(_ text: String) {
        synthesizer.stopSpeaking(at: .immediate)
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: Locale.current.languageCode ?? "en")
        utterance.rate = 0.5
        utterance.pitchMultiplier = 1.05
        synthesizer.speak(utterance)
    }

    private func stopAudio() {
        recognitionTask?.cancel()
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        synthesizer.stopSpeaking(at: .immediate)
    }
}
