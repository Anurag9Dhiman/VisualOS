import SwiftUI

/// Mic button + response strip for voice follow-up inside the card sheet.
struct VoiceFollowUpView: View {
    @StateObject private var vm: VoiceViewModel

    init(scanSessionId: String) {
        _vm = StateObject(wrappedValue: VoiceViewModel(
            scanSessionId: scanSessionId,
            userId: UserSession.userID
        ))
    }

    var body: some View {
        VStack(spacing: 0) {
            Divider()

            // Answer strip — visible while speaking or after
            if !vm.answer.isEmpty {
                Text(vm.answer)
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }

            // Mic row
            HStack(spacing: 16) {
                // Transcript while recording
                if case .recording = vm.voiceState {
                    Text(vm.transcript.isEmpty ? "Listening…" : vm.transcript)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else if case .thinking = vm.voiceState {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Thinking…")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Spacer()
                } else if case .error(let msg) = vm.voiceState {
                    Text(msg)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    Text("Ask a follow-up question")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                MicButton(voiceState: vm.voiceState) {
                    if case .recording = vm.voiceState {
                        vm.stopRecording()
                    } else if vm.voiceState == .idle {
                        vm.startRecording()
                    }
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
        }
        .animation(.easeInOut(duration: 0.2), value: vm.answer)
        .task { await vm.openSession() }
        .onDisappear { Task { await vm.closeSession() } }
    }
}

// MARK: - Mic button

private struct MicButton: View {
    let voiceState: VoiceState
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(buttonColor)
                    .frame(width: 48, height: 48)

                Image(systemName: iconName)
                    .font(.system(size: 20, weight: .medium))
                    .foregroundStyle(.white)
            }
        }
        .disabled(isDisabled)
        .scaleEffect(isRecording ? 1.1 : 1.0)
        .animation(.spring(response: 0.3), value: isRecording)
    }

    private var isRecording: Bool {
        if case .recording = voiceState { return true }
        return false
    }

    private var isDisabled: Bool {
        switch voiceState {
        case .requestingPermission, .thinking: return true
        default: return false
        }
    }

    private var buttonColor: Color {
        switch voiceState {
        case .recording: return .red
        case .thinking:  return .gray
        case .speaking:  return .tint
        default:         return .tint
        }
    }

    private var iconName: String {
        switch voiceState {
        case .recording: return "stop.fill"
        case .thinking:  return "mic.fill"
        case .speaking:  return "speaker.wave.2.fill"
        default:         return "mic.fill"
        }
    }
}

#Preview {
    VStack {
        Spacer()
        VoiceFollowUpView(scanSessionId: "preview-session")
    }
}
