/// WebSocket client for the Lens OS voice follow-up endpoint (/v1/ws).
///
/// Usage:
///   let vc = VoiceClient(baseURL: URL(string: "https://lens-os-api.fly.dev")!, apiKey: "lens-...")
///   try await vc.connect(userId: "u1")
///   vc.ask(text: "Who built this?", scanSessionId: "<session-id>")
///   for await event in vc.events { ... }
///   await vc.disconnect()

import Foundation

public actor VoiceClient {
    public enum Event: Sendable {
        case ack
        case progress(String)
        case speak(String)
        case done
        case error(String)
    }

    private let baseURL: URL
    private let apiKey: String
    private var task: URLSessionWebSocketTask?
    private let urlSession: URLSession

    public init(baseURL: URL, apiKey: String) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.urlSession = URLSession(configuration: .default)
    }

    // MARK: - Lifecycle

    public func connect(userId: String) async throws {
        let wsURL = makeWsURL()
        var request = URLRequest(url: wsURL)
        if !apiKey.isEmpty {
            request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")
        }
        let t = urlSession.webSocketTask(with: request)
        t.resume()
        self.task = t

        let sessionId = UUID().uuidString
        try await send(["type": "session_start", "session_id": sessionId, "user_id": userId])
    }

    public func ask(text: String, scanSessionId: String) {
        let msg: [String: Any] = [
            "type": "user_utterance",
            "text": text,
            "entity_refs": ["scan_session_id": scanSessionId],
        ]
        Task { try? await send(msg) }
    }

    public func disconnect() async {
        try? await send(["type": "session_end"])
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
    }

    // MARK: - Receive loop

    /// Yields events from the server until the connection closes or an error occurs.
    public func events() -> AsyncStream<Event> {
        AsyncStream { continuation in
            Task {
                while true {
                    guard let t = task else { continuation.finish(); return }
                    do {
                        let msg = try await t.receive()
                        guard case .string(let text) = msg,
                              let data = text.data(using: .utf8),
                              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                              let type_ = obj["type"] as? String
                        else { continue }

                        switch type_ {
                        case "ack":
                            continuation.yield(.ack)
                        case "progress":
                            let m = obj["message"] as? String ?? ""
                            continuation.yield(.progress(m))
                        case "speak":
                            let t = obj["text"] as? String ?? ""
                            continuation.yield(.speak(t))
                        case "done":
                            continuation.yield(.done)
                        case "error":
                            let d = obj["detail"] as? String ?? "Unknown error"
                            continuation.yield(.error(d))
                            continuation.finish()
                            return
                        default:
                            break
                        }
                    } catch {
                        continuation.finish()
                        return
                    }
                }
            }
        }
    }

    // MARK: - Private

    private func send(_ dict: [String: Any]) async throws {
        guard let t = task,
              let data = try? JSONSerialization.data(withJSONObject: dict),
              let text = String(data: data, encoding: .utf8)
        else { return }
        try await t.send(.string(text))
    }

    private func makeWsURL() -> URL {
        var comps = URLComponents(url: baseURL, resolvingAgainstBaseURL: true)!
        comps.scheme = (baseURL.scheme == "https") ? "wss" : "ws"
        comps.path = "/v1/ws"
        return comps.url!
    }
}
