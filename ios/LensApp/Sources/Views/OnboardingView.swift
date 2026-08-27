import AVFoundation
import CoreLocation
import SwiftUI

struct OnboardingView: View {
    @Binding var isOnboarded: Bool
    @State private var page = 0

    var body: some View {
        TabView(selection: $page) {
            WelcomePage(onNext: { withAnimation { page = 1 } })
                .tag(0)
            PermissionsPage(onDone: {
                UserSession.isOnboarded = true
                withAnimation { isOnboarded = true }
            })
            .tag(1)
        }
        .tabViewStyle(.page(indexDisplayMode: .never))
        .ignoresSafeArea()
    }
}

// MARK: - Page 1 — Welcome

private struct WelcomePage: View {
    let onNext: () -> Void

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color.black, Color(white: 0.08)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()

                Image(systemName: "camera.viewfinder")
                    .font(.system(size: 88))
                    .foregroundStyle(.white)
                    .padding(.bottom, 36)

                Text("Lens OS")
                    .font(.largeTitle.bold())
                    .foregroundStyle(.white)
                    .padding(.bottom, 14)

                Text("Point your camera at any building,\nmonument, or statue — get its story\nin under 3 seconds.")
                    .font(.body)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.white.opacity(0.65))
                    .padding(.horizontal, 40)

                Spacer()
                Spacer()

                Button(action: onNext) {
                    Text("Get Started")
                        .font(.headline)
                        .foregroundStyle(.black)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(.white)
                        .clipShape(Capsule())
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 56)
            }
        }
    }
}

// MARK: - Page 2 — Permissions

private struct PermissionsPage: View {
    let onDone: () -> Void

    @State private var cameraGranted = false
    @State private var locationGranted = false

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(white: 0.08), Color.black],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()

                Text("Two quick things")
                    .font(.title.bold())
                    .foregroundStyle(.white)
                    .padding(.bottom, 10)

                Text("Lens OS needs these to work.")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.55))
                    .padding(.bottom, 52)

                PermissionRow(
                    icon: "camera.fill",
                    title: "Camera",
                    description: "To see what you're pointing at.",
                    granted: cameraGranted,
                    onRequest: requestCamera
                )
                .padding(.horizontal, 28)
                .padding(.bottom, 16)

                PermissionRow(
                    icon: "location.fill",
                    title: "Location",
                    description: "Helps narrow down what you've found.",
                    granted: locationGranted,
                    onRequest: requestLocation
                )
                .padding(.horizontal, 28)

                Spacer()
                Spacer()

                Button(action: onDone) {
                    Text(cameraGranted ? "Continue" : "Allow Camera to Continue")
                        .font(.headline)
                        .foregroundStyle(.black)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(cameraGranted ? Color.white : Color(white: 0.45))
                        .clipShape(Capsule())
                }
                .disabled(!cameraGranted)
                .padding(.horizontal, 32)
                .padding(.bottom, 56)
            }
        }
        .onAppear(perform: checkExistingPermissions)
    }

    private func checkExistingPermissions() {
        let camStatus = AVCaptureDevice.authorizationStatus(for: .video)
        cameraGranted = camStatus == .authorized

        let locStatus = CLLocationManager().authorizationStatus
        locationGranted = locStatus == .authorizedWhenInUse || locStatus == .authorizedAlways
    }

    private func requestCamera() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async { cameraGranted = granted }
        }
    }

    private func requestLocation() {
        LocationPermissionRequester.shared.request { granted in
            DispatchQueue.main.async { locationGranted = granted }
        }
    }
}

// MARK: - Permission row

private struct PermissionRow: View {
    let icon: String
    let title: String
    let description: String
    let granted: Bool
    let onRequest: () -> Void

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundStyle(granted ? .green : .white)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.headline)
                    .foregroundStyle(.white)
                Text(description)
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.6))
            }

            Spacer()

            if granted {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .font(.title3)
            } else {
                Button("Allow", action: onRequest)
                    .buttonStyle(.bordered)
                    .tint(.white)
            }
        }
        .padding(16)
        .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
    }
}

// MARK: - CLLocationManager delegate bridge
// Must be held strongly while waiting for the async delegate callback.

private final class LocationPermissionRequester: NSObject, CLLocationManagerDelegate {
    static let shared = LocationPermissionRequester()
    private let manager = CLLocationManager()
    private var completion: ((Bool) -> Void)?

    override init() {
        super.init()
        manager.delegate = self
    }

    func request(completion: @escaping (Bool) -> Void) {
        self.completion = completion
        let status = manager.authorizationStatus
        if status == .notDetermined {
            manager.requestWhenInUseAuthorization()
        } else {
            let granted = status == .authorizedWhenInUse || status == .authorizedAlways
            completion(granted)
        }
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let granted = manager.authorizationStatus == .authorizedWhenInUse
            || manager.authorizationStatus == .authorizedAlways
        completion?(granted)
        completion = nil
    }
}
