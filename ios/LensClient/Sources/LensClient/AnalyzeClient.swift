/// HTTP client for the Lens OS API.
///
/// Usage:
///   let client = AnalyzeClient(baseURL: URL(string: "https://lens-os-api.fly.dev")!,
///                              apiKey: "lens-...")
///
///   // Blocking
///   let card = try await client.analyze(imageData: jpeg, lat: 12.95, lng: 77.58)
///
///   // Streaming
///   for try await event in client.analyzeStream(imageData: jpeg, lat: 12.95, lng: 77.58) {
///       switch event {
///       case .token(let chunk): print(chunk, terminator: "")
///       case .card(let card):   showCard(card)
///       case .error(let msg):   showError(msg)
///       }
///   }

import Foundation

public struct AnalyzeClient: Sendable {
    public let baseURL: URL
    private let apiKey: String
    private let session: URLSession

    public init(baseURL: URL, apiKey: String, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.session = session
    }

    // MARK: - POST /analyze (blocking)

    public func analyze(
        imageData: Data,
        imageMimeType: String = "image/jpeg",
        lat: Double? = nil,
        lng: Double? = nil,
        userID: String = "anon",
        userLocale: String = "en-IN"
    ) async throws -> ResponseCard {
        var request = try buildRequest(path: "/analyze")
        let (body, boundary) = buildMultipartBody(
            imageData: imageData, imageMimeType: imageMimeType,
            lat: lat, lng: lng, userID: userID, userLocale: userLocale
        )
        request.httpBody = body
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        let (data, response) = try await session.data(for: request)
        try validateResponse(response, data: data)

        do {
            return try JSONDecoder().decode(ResponseCard.self, from: data)
        } catch {
            throw LensError.decodingError(error)
        }
    }

    // MARK: - POST /analyze/stream (SSE)

    public func analyzeStream(
        imageData: Data,
        imageMimeType: String = "image/jpeg",
        lat: Double? = nil,
        lng: Double? = nil,
        userID: String = "anon",
        userLocale: String = "en-IN"
    ) -> AsyncThrowingStream<StreamEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    var request = try buildRequest(path: "/analyze/stream")
                    let (body, boundary) = buildMultipartBody(
                        imageData: imageData, imageMimeType: imageMimeType,
                        lat: lat, lng: lng, userID: userID, userLocale: userLocale
                    )
                    request.httpBody = body
                    request.setValue(
                        "multipart/form-data; boundary=\(boundary)",
                        forHTTPHeaderField: "Content-Type"
                    )

                    let (bytes, response) = try await session.bytes(for: request)
                    try validateResponse(response, data: nil)

                    for try await line in bytes.lines {
                        guard line.hasPrefix("data: ") else { continue }
                        let json = String(line.dropFirst(6))
                        guard let data = json.data(using: .utf8),
                              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                              let type_ = obj["type"] as? String
                        else { continue }

                        switch type_ {
                        case "token":
                            if let text = obj["text"] as? String {
                                continuation.yield(.token(text))
                            }
                        case "card":
                            if let cardObj = obj["card"],
                               let cardData = try? JSONSerialization.data(withJSONObject: cardObj) {
                                let card = try JSONDecoder().decode(ResponseCard.self, from: cardData)
                                continuation.yield(.card(card))
                            }
                        case "error":
                            let detail = obj["detail"] as? String ?? "Unknown error"
                            continuation.yield(.error(detail))
                        default:
                            break
                        }
                    }
                    continuation.finish()
                } catch let e as LensError {
                    continuation.finish(throwing: e)
                } catch {
                    continuation.finish(throwing: LensError.networkError(error))
                }
            }
        }
    }

    // MARK: - Private helpers

    private func buildRequest(path: String) throws -> URLRequest {
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw LensError.networkError(URLError(.badURL))
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")
        request.timeoutInterval = 30
        return request
    }

    private func buildMultipartBody(
        imageData: Data,
        imageMimeType: String,
        lat: Double?,
        lng: Double?,
        userID: String,
        userLocale: String
    ) -> (Data, String) {
        let boundary = "LensOSBoundary-\(UUID().uuidString)"
        var body = Data()

        func append(_ string: String) {
            if let data = string.data(using: .utf8) { body.append(data) }
        }

        // Image field
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"image\"; filename=\"photo.jpg\"\r\n")
        append("Content-Type: \(imageMimeType)\r\n\r\n")
        body.append(imageData)
        append("\r\n")

        // Optional GPS fields
        if let lat {
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"lat\"\r\n\r\n")
            append("\(lat)\r\n")
        }
        if let lng {
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"lng\"\r\n\r\n")
            append("\(lng)\r\n")
        }

        // user_id
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"user_id\"\r\n\r\n")
        append("\(userID)\r\n")

        // user_locale
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"user_locale\"\r\n\r\n")
        append("\(userLocale)\r\n")

        append("--\(boundary)--\r\n")
        return (body, boundary)
    }

    private func validateResponse(_ response: URLResponse, data: Data?) throws {
        guard let http = response as? HTTPURLResponse else { return }
        switch http.statusCode {
        case 200...299: return
        case 401: throw LensError.unauthorized
        case 415: throw LensError.unsupportedMediaType
        case 504: throw LensError.timeout
        default:
            let detail = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
            throw LensError.serverError(http.statusCode, detail)
        }
    }
}
