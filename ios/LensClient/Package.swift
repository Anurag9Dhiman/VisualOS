// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LensClient",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "LensClient", targets: ["LensClient"]),
    ],
    targets: [
        .target(
            name: "LensClient",
            path: "Sources/LensClient"
        ),
        .testTarget(
            name: "LensClientTests",
            dependencies: ["LensClient"],
            path: "Tests/LensClientTests"
        ),
    ]
)
