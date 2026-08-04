import CoreLocation
import LensClient
import PhotosUI
import SwiftUI

struct ContentView: View {
    @StateObject private var vm = ScanViewModel()
    @StateObject private var locationManager = LocationManager()

    @State private var pickerItem: PhotosPickerItem?
    @State private var showPicker = false

    var body: some View {
        NavigationStack {
            Group {
                switch vm.state {
                case .idle:
                    IdleView(onScan: { showPicker = true })

                case .scanning:
                    ScanningView(label: "Identifying…")

                case .streaming(let text):
                    ScanningView(label: text.isEmpty ? "Thinking…" : text)

                case .done(let card):
                    ScrollView {
                        CardView(card: card)
                            .padding()
                    }
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button("Scan again") { vm.reset() }
                        }
                    }

                case .failed(let msg):
                    ErrorView(message: msg, onRetry: { vm.reset() })
                }
            }
            .navigationTitle("Lens OS")
            .navigationBarTitleDisplayMode(.inline)
        }
        .photosPicker(isPresented: $showPicker, selection: $pickerItem, matching: .images)
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            Task {
                guard let data = try? await item.loadTransferable(type: Data.self) else { return }
                await vm.scan(imageData: data, location: locationManager.location)
                pickerItem = nil
            }
        }
    }
}

// MARK: - Sub-views

private struct IdleView: View {
    let onScan: () -> Void

    var body: some View {
        VStack(spacing: 32) {
            Image(systemName: "camera.viewfinder")
                .font(.system(size: 80))
                .foregroundStyle(.secondary)
            Text("Point at a building, monument, or object")
                .font(.headline)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button(action: onScan) {
                Label("Scan", systemImage: "camera.fill")
                    .font(.title3.bold())
                    .padding(.horizontal, 32)
                    .padding(.vertical, 14)
                    .background(.tint)
                    .foregroundStyle(.white)
                    .clipShape(Capsule())
            }
        }
        .padding(40)
    }
}

private struct ScanningView: View {
    let label: String

    var body: some View {
        VStack(spacing: 24) {
            ProgressView()
                .scaleEffect(1.4)
            Text(label)
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
                .animation(.default, value: label)
        }
    }
}

private struct ErrorView: View {
    let message: String
    let onRetry: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 48))
                .foregroundStyle(.orange)
            Text(message)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 32)
            Button("Try again", action: onRetry)
                .buttonStyle(.bordered)
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
