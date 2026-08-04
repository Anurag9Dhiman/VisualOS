import CoreLocation
import LensClient
import SwiftUI

enum ScanState {
    case idle
    case scanning
    case streaming(String)  // accumulated tokens so far
    case done(ResponseCard)
    case failed(String)

    var isActive: Bool {
        switch self {
        case .scanning, .streaming: return true
        default: return false
        }
    }
}

@MainActor
final class ScanViewModel: ObservableObject {
    @Published var state: ScanState = .idle

    private let client: AnalyzeClient

    init() {
        let base = ProcessInfo.processInfo.environment["LENS_API_URL"]
            ?? "http://localhost:8000"
        let key = ProcessInfo.processInfo.environment["LENS_API_KEY"] ?? ""
        client = AnalyzeClient(
            baseURL: URL(string: base)!,
            apiKey: key
        )
    }

    func scan(imageData: Data, location: CLLocation?) async {
        state = .scanning
        var accumulated = ""

        do {
            for try await event in client.analyzeStream(
                imageData: imageData,
                lat: location?.coordinate.latitude,
                lng: location?.coordinate.longitude
            ) {
                switch event {
                case .token(let chunk):
                    accumulated += chunk
                    state = .streaming(accumulated)
                case .card(let card):
                    state = .done(card)
                case .error(let detail):
                    state = .failed(detail)
                }
            }
        } catch LensError.unauthorized {
            state = .failed("API key invalid — check LENS_API_KEY.")
        } catch LensError.timeout {
            state = .failed("Took too long. Try again.")
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func reset() {
        state = .idle
    }
}
