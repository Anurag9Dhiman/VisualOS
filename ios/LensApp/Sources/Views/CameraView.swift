import AVFoundation
import SwiftUI

// Wraps AVCaptureSession in a UIViewControllerRepresentable.
// The shutter button lives here so it renders on top of the preview layer.
// Pass isEnabled: false while a scan is in progress to grey out the button.
struct CameraView: UIViewControllerRepresentable {
    let isEnabled: Bool
    let onCapture: (Data) -> Void

    func makeUIViewController(context: Context) -> CameraViewController {
        CameraViewController(onCapture: onCapture)
    }

    func updateUIViewController(_ vc: CameraViewController, context: Context) {
        vc.setShutterEnabled(isEnabled)
    }
}

final class CameraViewController: UIViewController {
    let onCapture: (Data) -> Void

    private let session = AVCaptureSession()
    private let photoOutput = AVCapturePhotoOutput()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private let shutterButton = UIButton(type: .custom)

    init(onCapture: @escaping (Data) -> Void) {
        self.onCapture = onCapture
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        setupSession()
        setupPreviewLayer()
        setupShutterButton()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        guard !session.isRunning else { return }
        DispatchQueue.global(qos: .userInitiated).async { self.session.startRunning() }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        guard session.isRunning else { return }
        DispatchQueue.global(qos: .userInitiated).async { self.session.stopRunning() }
    }

    func setShutterEnabled(_ enabled: Bool) {
        shutterButton.isEnabled = enabled
        UIView.animate(withDuration: 0.2) {
            self.shutterButton.alpha = enabled ? 1.0 : 0.35
        }
    }

    private func setupSession() {
        session.sessionPreset = .photo
        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input),
            session.canAddOutput(photoOutput)
        else { return }
        session.addInput(input)
        session.addOutput(photoOutput)
    }

    private func setupPreviewLayer() {
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.frame = view.bounds
        view.layer.addSublayer(layer)
        previewLayer = layer
    }

    private func setupShutterButton() {
        shutterButton.translatesAutoresizingMaskIntoConstraints = false
        shutterButton.layer.cornerRadius = 36
        shutterButton.layer.borderWidth = 4
        shutterButton.layer.borderColor = UIColor.white.cgColor
        shutterButton.backgroundColor = UIColor.white.withAlphaComponent(0.85)
        shutterButton.addTarget(self, action: #selector(didTapShutter), for: .touchUpInside)
        view.addSubview(shutterButton)

        NSLayoutConstraint.activate([
            shutterButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            shutterButton.bottomAnchor.constraint(
                equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -28
            ),
            shutterButton.widthAnchor.constraint(equalToConstant: 72),
            shutterButton.heightAnchor.constraint(equalToConstant: 72),
        ])
    }

    @objc private func didTapShutter() {
        setShutterEnabled(false)
        let settings = AVCapturePhotoSettings()
        settings.flashMode = .auto
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
}

extension CameraViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        guard error == nil, let data = photo.fileDataRepresentation() else {
            DispatchQueue.main.async { self.setShutterEnabled(true) }
            return
        }
        DispatchQueue.main.async { self.onCapture(data) }
    }
}
