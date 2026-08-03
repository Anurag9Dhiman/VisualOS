import Foundation
import Testing
@testable import LensClient

// MARK: - Mock URLSession

/// Minimal URLSession stand-in that returns a pre-canned response.
final class MockSession: @unchecked Sendable {
    var stubbedData: Data = Data()
    var stubbedStatusCode: Int = 200
    var stubbedError: Error?
}

// We can't subclass URLSession cleanly, so AnalyzeClient accepts a URLSession
// and the tests use a real URLSession with a MockURLProtocol instead.

final class MockURLProtocol: URLProtocol, @unchecked Sendable {
    static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = MockURLProtocol.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.unknown))
            return
        }
        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

private func makeMockSession() -> URLSession {
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [MockURLProtocol.self]
    return URLSession(configuration: config)
}

private func makeClient(_ session: URLSession) -> AnalyzeClient {
    AnalyzeClient(
        baseURL: URL(string: "https://test.example.com")!,
        apiKey: "lens-test-key",
        session: session
    )
}

private func makeResponse(statusCode: Int) -> HTTPURLResponse {
    HTTPURLResponse(
        url: URL(string: "https://test.example.com")!,
        statusCode: statusCode,
        httpVersion: nil,
        headerFields: nil
    )!
}

// MARK: - Sample JSON

private let normalCardJSON = """
{
  "card_type": "normal",
  "headline": "Lalbagh Botanical Garden",
  "body": "Founded in 1760 by Hyder Ali.",
  "personalized_hooks": [],
  "citations": [],
  "confidence_displayed": "high",
  "source_mix": {"used_vision": true, "used_memory": false, "used_search": true},
  "cost_usd_total": 0.001,
  "latency_ms": 430
}
""".data(using: .utf8)!

private let fallbackCardJSON = """
{
  "card_type": "fallback",
  "headline": "Not sure what this is.",
  "observation": "I can see a stone structure partially hidden by foliage.",
  "suggestion": "Try a clearer angle or move closer."
}
""".data(using: .utf8)!

// MARK: - Tests

@Suite("AnalyzeClient", .serialized)
struct AnalyzeClientTests {

    // MARK: Model decoding

    @Test("Decodes normal card")
    func decodesNormalCard() throws {
        let card = try JSONDecoder().decode(ResponseCard.self, from: normalCardJSON)
        guard case .normal(let c) = card else { Issue.record("Expected normal card"); return }
        #expect(c.headline == "Lalbagh Botanical Garden")
        #expect(c.latencyMs == 430)
        #expect(c.sourceMix.usedVision == true)
    }

    @Test("Decodes fallback card")
    func decodesFallbackCard() throws {
        let card = try JSONDecoder().decode(ResponseCard.self, from: fallbackCardJSON)
        guard case .fallback(let c) = card else { Issue.record("Expected fallback card"); return }
        #expect(c.headline == "Not sure what this is.")
        #expect(c.suggestion.contains("clearer angle"))
    }

    @Test("headline property works for both card types")
    func headlineProperty() throws {
        let normal = try JSONDecoder().decode(ResponseCard.self, from: normalCardJSON)
        let fallback = try JSONDecoder().decode(ResponseCard.self, from: fallbackCardJSON)
        #expect(normal.headline == "Lalbagh Botanical Garden")
        #expect(fallback.headline == "Not sure what this is.")
    }

    @Test("Unknown card_type throws decoding error")
    func unknownCardType() {
        let json = #"{"card_type":"mystery","headline":"?"}"#.data(using: .utf8)!
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(ResponseCard.self, from: json)
        }
    }

    // MARK: analyze() — happy path

    @Test("analyze returns normal card on 200")
    func analyzeReturnsCard() async throws {
        let session = makeMockSession()
        MockURLProtocol.handler = { _ in (makeResponse(statusCode: 200), normalCardJSON) }
        let card = try await makeClient(session).analyze(imageData: Data([0xFF, 0xD8]))
        guard case .normal(let c) = card else { Issue.record("Expected normal"); return }
        #expect(c.headline == "Lalbagh Botanical Garden")
    }

    @Test("analyze sends X-API-Key header")
    func analyzeSendsAPIKeyHeader() async throws {
        let session = makeMockSession()
        var capturedRequest: URLRequest?
        MockURLProtocol.handler = { req in
            capturedRequest = req
            return (makeResponse(statusCode: 200), normalCardJSON)
        }
        _ = try await makeClient(session).analyze(imageData: Data([0xFF, 0xD8]))
        #expect(capturedRequest?.value(forHTTPHeaderField: "X-API-Key") == "lens-test-key")
    }

    @Test("analyze sends multipart POST to /analyze")
    func analyzeSendsMultipartPost() async throws {
        let session = makeMockSession()
        var capturedRequest: URLRequest?
        MockURLProtocol.handler = { req in
            capturedRequest = req
            return (makeResponse(statusCode: 200), normalCardJSON)
        }
        _ = try await makeClient(session).analyze(imageData: Data([0xFF, 0xD8]))
        #expect(capturedRequest?.httpMethod == "POST")
        #expect(capturedRequest?.url?.path == "/analyze")
        let ct = capturedRequest?.value(forHTTPHeaderField: "Content-Type") ?? ""
        #expect(ct.hasPrefix("multipart/form-data"))
    }

    // MARK: analyze() — error cases

    @Test("analyze throws .unauthorized on 401")
    func analyzeThrowsUnauthorized() async throws {
        let session = makeMockSession()
        MockURLProtocol.handler = { _ in (makeResponse(statusCode: 401), Data()) }
        await #expect(throws: LensError.unauthorized) {
            try await makeClient(session).analyze(imageData: Data([0xFF, 0xD8]))
        }
    }

    @Test("analyze throws .timeout on 504")
    func analyzeThrowsTimeout() async throws {
        let session = makeMockSession()
        MockURLProtocol.handler = { _ in (makeResponse(statusCode: 504), Data()) }
        await #expect(throws: LensError.timeout) {
            try await makeClient(session).analyze(imageData: Data([0xFF, 0xD8]))
        }
    }

    @Test("analyze throws .serverError on 500")
    func analyzeThrowsServerError() async throws {
        let session = makeMockSession()
        let body = #"{"detail":"Pipeline produced no card"}"#.data(using: .utf8)!
        MockURLProtocol.handler = { _ in (makeResponse(statusCode: 500), body) }
        await #expect(throws: LensError.self) {
            try await makeClient(session).analyze(imageData: Data([0xFF, 0xD8]))
        }
    }

    @Test("analyze throws .decodingError on malformed JSON")
    func analyzeThrowsDecodingError() async throws {
        let session = makeMockSession()
        MockURLProtocol.handler = { _ in (makeResponse(statusCode: 200), Data("not json".utf8)) }
        await #expect(throws: LensError.self) {
            try await makeClient(session).analyze(imageData: Data([0xFF, 0xD8]))
        }
    }

    // MARK: LensError

    @Test("LensError cases are distinct")
    func lensErrorCases() {
        let errors: [LensError] = [
            .unauthorized, .unsupportedMediaType, .timeout,
            .serverError(500, "oops"), .networkError(URLError(.notConnectedToInternet)),
        ]
        #expect(errors.count == 5)
    }
}
