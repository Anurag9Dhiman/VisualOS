import CoreLocation
import LensClient
import SwiftUI

struct ContentView: View {
    @StateObject private var vm = ScanViewModel()
    @StateObject private var locationManager = LocationManager()

    // Drives the result sheet without making ScanState Equatable.
    private var showingCard: Binding<Bool> {
        Binding(
            get: { if case .done = vm.state { return true }; return false },
            set: { if !$0 { vm.reset() } }
        )
    }

    private var currentSessionId: String? {
        if case .done(_, let sid) = vm.state { return sid }
        return nil
    }

    var body: some View {
        ZStack {
            // Full-screen live viewfinder — always running so there's no black flash on reset.
            CameraView(isEnabled: vm.state.isIdle) { imageData in
                Task { await vm.scan(imageData: imageData, location: locationManager.location) }
            }
            .ignoresSafeArea()

            // State overlays sit on top of the camera feed.
            switch vm.state {
            case .idle:
                AppLabel()

            case .scanning:
                ProcessingOverlay(label: "Identifying…")

            case .streaming(let text):
                ProcessingOverlay(label: text.isEmpty ? "Thinking…" : text)

            case .done:
                EmptyView() // handled by .sheet below

            case .failed(let msg):
                ErrorOverlay(message: msg, onRetry: { vm.reset() })
            }
        }
        .sheet(isPresented: showingCard) {
            if case .done(let card, let sessionId) = vm.state {
                CardSheetContent(card: card, sessionId: sessionId, onClose: { vm.reset() })
            }
        }
    }
}

// MARK: - Overlays

private struct AppLabel: View {
    var body: some View {
        VStack {
            Text("LENS OS")
                .font(.caption.weight(.semibold))
                .kerning(2)
                .foregroundStyle(.white.opacity(0.7))
                .padding(.top, 16)
            Spacer()
        }
    }
}

private struct ProcessingOverlay: View {
    let label: String

    var body: some View {
        ZStack {
            Color.black.opacity(0.55).ignoresSafeArea()
            VStack(spacing: 24) {
                ProgressView()
                    .progressViewStyle(.circular)
                    .tint(.white)
                    .scaleEffect(1.4)
                Text(label)
                    .font(.body)
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
                    .animation(.default, value: label)
            }
        }
    }
}

private struct ErrorOverlay: View {
    let message: String
    let onRetry: () -> Void

    var body: some View {
        ZStack {
            Color.black.opacity(0.65).ignoresSafeArea()
            VStack(spacing: 20) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 48))
                    .foregroundStyle(.orange)
                Text(message)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 32)
                Button("Try again", action: onRetry)
                    .buttonStyle(.borderedProminent)
            }
        }
    }
}

// MARK: - Result sheet

private struct CardSheetContent: View {
    let card: ResponseCard
    let sessionId: String?
    let onClose: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollView {
                    CardView(card: card)
                        .padding()
                }

                // Voice follow-up — only when the server gave us a session_id
                if let sid = sessionId {
                    VoiceFollowUpView(scanSessionId: sid)
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                        onClose()
                    }
                }
            }
        }
    }
}

// MARK: - Location

final class LocationManager: NSObject, ObservableObject, CLLocationManagerDelegate {
    @Published private(set) var location: CLLocation?
    private let manager = CLLocationManager()

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
        manager.requestWhenInUseAuthorization()
        manager.startUpdatingLocation()
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        location = locations.last
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
