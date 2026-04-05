"use client";

import { useState, useCallback } from "react";

const API_URL = "http://127.0.0.1:8000";

interface PredictionResult {
  is_fake: boolean;
  confidence: number;
  label: string;
  heatmap_base64: string;
}

interface VideoResult {
  is_fake: boolean;
  confidence: number;
  label: string;
  frame_confidences: number[];
  top_frame_index: number;
  heatmap_base64: string;
  frames_analyzed: number;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isVideo, setIsVideo] = useState(false);
  const [result, setResult] = useState<PredictionResult | VideoResult | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = (f: File) => {
    setFile(f);
    setResult(null);
    setError(null);

    const isVideoFile = f.type.startsWith("video/");
    setIsVideo(isVideoFile);

    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && (f.type.startsWith("image/") || f.type.startsWith("video/"))) {
      handleFile(f);
    } else {
      setError("Please upload an image or video file.");
    }
  }, []);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const endpoint = isVideo
        ? `${API_URL}/predict_video`
        : `${API_URL}/predict`;
      const res = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setIsVideo(false);
    setResult(null);
    setError(null);
  };

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center px-4 py-12">
      {/* Header */}
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold tracking-tight mb-2">
          🔍 Deepfake Detector
        </h1>
        <p className="text-gray-400 text-lg">
          Upload an image or video to detect AI manipulation using our hybrid
          CNN + frequency analysis model.
        </p>
      </div>

      {/* Upload Area */}
      {!result && (
        <div
          onDrop={onDrop}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          className={`w-full max-w-xl border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all
            ${dragging ? "border-blue-400 bg-blue-950" : "border-gray-600 bg-gray-900 hover:border-gray-400"}`}
          onClick={() => document.getElementById("fileInput")?.click()}
        >
          <input
            id="fileInput"
            type="file"
            accept="image/*,video/*"
            className="hidden"
            onChange={onFileChange}
          />
          {preview ? (
            isVideo ? (
              <video
                src={preview}
                controls
                className="mx-auto max-h-64 rounded-xl object-contain"
              />
            ) : (
              <img
                src={preview}
                alt="Preview"
                className="mx-auto max-h-64 rounded-xl object-contain"
              />
            )
          ) : (
            <>
              <div className="text-5xl mb-4">🖼️</div>
              <p className="text-gray-300 text-lg font-medium">
                Drag & drop an image or video here
              </p>
              <p className="text-gray-500 text-sm mt-1">or click to browse</p>
              <p className="text-gray-600 text-xs mt-3">
                Supports JPG, PNG, WEBP, MP4, MOV
              </p>
            </>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-4 bg-red-900 border border-red-500 text-red-200 px-4 py-3 rounded-xl max-w-xl w-full">
          ⚠️ {error}
        </div>
      )}

      {/* Analyze Button */}
      {file && !result && (
        <button
          onClick={analyze}
          disabled={loading}
          className="mt-6 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-900 disabled:cursor-not-allowed
            text-white font-semibold px-8 py-3 rounded-xl transition-all text-lg"
        >
          {loading
            ? "⏳ Analyzing..."
            : isVideo
              ? "🎬 Analyze Video"
              : "🔬 Analyze Image"}
        </button>
      )}

      {/* Results */}
      {result && (
        <div className="w-full max-w-3xl mt-6">
          {/* Verdict */}
          <div
            className={`rounded-2xl p-6 mb-6 text-center border-2
            ${
              result.is_fake
                ? "bg-red-950 border-red-500"
                : "bg-green-950 border-green-500"
            }`}
          >
            <div className="text-6xl mb-3">{result.is_fake ? "🚨" : "✅"}</div>
            <h2 className="text-4xl font-bold mb-2">
              {result.is_fake ? "DEEPFAKE DETECTED" : "LIKELY REAL"}
            </h2>
            <p className="text-xl text-gray-300">
              Confidence:{" "}
              <span className="font-bold text-white">
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </p>

            {/* Confidence Bar */}
            <div className="mt-4 bg-gray-800 rounded-full h-3 max-w-sm mx-auto">
              <div
                className={`h-3 rounded-full transition-all ${result.is_fake ? "bg-red-500" : "bg-green-500"}`}
                style={{ width: `${result.confidence * 100}%` }}
              />
            </div>
          </div>

          {/* For Images */}
          {!isVideo && "heatmap_base64" in result && (
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="bg-gray-900 rounded-2xl p-4">
                <p className="text-gray-400 text-sm font-medium mb-2 text-center">
                  Original Image
                </p>
                <img
                  src={preview!}
                  alt="Original"
                  className="w-full rounded-xl object-contain max-h-64"
                />
              </div>
              <div className="bg-gray-900 rounded-2xl p-4">
                <p className="text-gray-400 text-sm font-medium mb-2 text-center">
                  GradCAM Heatmap
                </p>
                <img
                  src={`data:image/jpeg;base64,${result.heatmap_base64}`}
                  alt="GradCAM"
                  className="w-full rounded-xl object-contain max-h-64"
                />
              </div>
            </div>
          )}

          {/* For Videos */}
          {isVideo && "frames_analyzed" in result && (
            <>
              <div className="bg-gray-900 rounded-2xl p-4 mb-6">
                <p className="text-gray-400 text-sm font-medium mb-3 text-center">
                  Top Frame GradCAM Heatmap
                </p>
                <img
                  src={`data:image/jpeg;base64,${result.heatmap_base64}`}
                  alt="GradCAM"
                  className="w-full rounded-xl object-contain max-h-64"
                />
              </div>

              <div className="bg-gray-900 rounded-2xl p-4 mb-6">
                <p className="text-gray-400 text-sm font-medium mb-2">
                  Analyzed {result.frames_analyzed} frames — highest confidence
                  frame: #{result.top_frame_index}
                </p>
              </div>

              {/* Frame Confidence Chart */}
              <div className="bg-gray-900 rounded-2xl p-6 mb-6">
                <p className="text-gray-400 text-sm font-medium mb-4">
                  Frame Confidence Analysis
                </p>
                <div className="space-y-2">
                  {result.frame_confidences.map((conf, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <span className="text-xs text-gray-400 w-10">#{idx}</span>
                      <div className="flex-1 bg-gray-700 rounded-full h-6 overflow-hidden">
                        <div
                          className={`h-full transition-all flex items-center justify-end pr-2 text-xs font-semibold text-white ${
                            conf > 0.5 ? "bg-red-500" : "bg-green-500"
                          }`}
                          style={{ width: `${conf * 100}%` }}
                        >
                          {(conf * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Explanation */}
          <div className="bg-gray-900 rounded-2xl p-5 mb-6 text-sm text-gray-400">
            <p className="font-semibold text-gray-200 mb-1">How this works</p>
            <p>
              Our hybrid model combines{" "}
              <strong className="text-white">EfficientNet-B4</strong> spatial
              analysis with{" "}
              <strong className="text-white">FFT frequency domain</strong>{" "}
              anomaly detection. The heatmap highlights regions the model
              focused on — warm colours indicate suspected manipulation areas.
            </p>
          </div>

          {/* Analyze another */}
          <button
            onClick={reset}
            className="w-full bg-gray-800 hover:bg-gray-700 text-white font-semibold py-3 rounded-xl transition-all"
          >
            🔄 Analyze Another {isVideo ? "Video" : "Image"}
          </button>
        </div>
      )}

      {/* Footer */}
      <p className="mt-12 text-gray-600 text-xs text-center">
        HP-AI CoE 30-Day Agile Challenge • Hybrid CNN + FFT Deepfake Detection
      </p>
    </main>
  );
}
