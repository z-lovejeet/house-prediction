import Cocoa
import Vision

let imagePath = "coverpageformat.png"
guard let image = NSImage(contentsOfFile: imagePath),
      let tiffData = image.tiffRepresentation,
      let bitmapImage = NSBitmapImageRep(data: tiffData),
      let cgImage = bitmapImage.cgImage else {
    print("Failed to load image.")
    exit(1)
}

let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
let request = VNRecognizeTextRequest { request, error in
    guard let observations = request.results as? [VNRecognizedTextObservation], error == nil else {
        print("OCR failed: \(error?.localizedDescription ?? "unknown error")")
        return
    }
    
    for observation in observations {
        guard let topCandidate = observation.topCandidates(1).first else { continue }
        // Print bounding box roughly to help infer layout
        let y = observation.boundingBox.origin.y
        let text = topCandidate.string
        print("Y: \(String(format: "%.3f", y)) - \(text)")
    }
}
request.recognitionLevel = .accurate
try? requestHandler.perform([request])
