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
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900 text-white flex flex-col items-center px-4 py-16">
      {/* Header */}
      <div className="max-w-2xl mx-auto text-center mb-14">
        <div className="mb-4 inline-block">
          <div className="text-5xl font-light tracking-tight bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent">
            Deepfake Detector
          </div>
        </div>

        <p className="text-slate-400 text-base leading-relaxed font-light">
          Analyze images and videos with our hybrid CNN + frequency analysis
          engine. Advanced detection powered by EfficientNet-B4 and FFT-based
          anomaly recognition.
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
          className={`w-full max-w-2xl rounded-2xl p-12 text-center cursor-pointer transition-all duration-300
            ${
              dragging
                ? "border-2 border-blue-400 bg-blue-950/30 shadow-lg shadow-blue-500/10"
                : "border border-slate-700 bg-slate-800/40 hover:border-slate-600 hover:bg-slate-800/50"
            } backdrop-blur-sm`}
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
            <div className="space-y-4">
              {isVideo ? (
                <video
                  src={preview}
                  controls
                  className="mx-auto max-h-72 rounded-lg object-contain shadow-lg"
                />
              ) : (
                <img
                  src={preview}
                  alt="Preview"
                  className="mx-auto max-h-72 rounded-lg object-contain shadow-lg"
                />
              )}
              <p className="text-sm text-slate-400">
                {file?.name} • {(file!.size / 1024 / 1024).toFixed(1)}MB
              </p>
            </div>
          ) : (
            <>
              <div className="mb-4 inline-block">
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-500/20 to-cyan-500/20 flex items-center justify-center">
                  <svg
                    className="w-8 h-8 text-blue-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                </div>
              </div>
              <h3 className="text-lg font-medium text-white mb-1">
                Drop your file here
              </h3>
              <p className="text-sm text-slate-400 mb-3">
                or click to browse your computer
              </p>
              <div className="flex flex-wrap gap-2 justify-center">
                <span className="text-xs px-3 py-1 rounded-full bg-slate-700/50 text-slate-300">
                  JPG
                </span>
                <span className="text-xs px-3 py-1 rounded-full bg-slate-700/50 text-slate-300">
                  PNG
                </span>
                <span className="text-xs px-3 py-1 rounded-full bg-slate-700/50 text-slate-300">
                  WEBP
                </span>
                <span className="text-xs px-3 py-1 rounded-full bg-slate-700/50 text-slate-300">
                  MP4
                </span>
              </div>
            </>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-6 max-w-2xl w-full px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm animate-in fade-in duration-300">
          <div className="flex items-start gap-3">
            <svg
              className="w-5 h-5 flex-shrink-0 mt-0.5"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Analyze Button */}
      {file && !result && (
        <button
          onClick={analyze}
          disabled={loading}
          className="mt-8 group relative px-8 py-3.5 rounded-lg font-medium text-sm transition-all duration-300
            disabled:opacity-50 disabled:cursor-not-allowed
            bg-gradient-to-r from-blue-600 to-cyan-600
            hover:shadow-lg hover:shadow-blue-500/30
            text-white"
        >
          <span className="relative flex items-center justify-center">
            {loading ? (
              <>
                <svg
                  className="w-4 h-4 mr-2 animate-spin"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M14.828 14.828a4 4 0 01-5.656 0M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                Analyzing...
              </>
            ) : (
              <>
                <svg
                  className="w-4 h-4 mr-2"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
                {isVideo ? "Analyze Video" : "Analyze Image"}
              </>
            )}
          </span>
        </button>
      )}

      {/* Results */}
      {result && (
        <div className="w-full max-w-3xl mt-12 space-y-8">
          {/* Verdict Card */}
          <div
            className={`rounded-2xl overflow-hidden backdrop-blur-sm transition-all duration-500
            ${
              result.is_fake
                ? "bg-gradient-to-br from-red-950/40 to-red-900/20 border border-red-500/20 shadow-lg shadow-red-500/10"
                : "bg-gradient-to-br from-emerald-950/40 to-emerald-900/20 border border-emerald-500/20 shadow-lg shadow-emerald-500/10"
            }`}
          >
            <div className="p-8">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">
                    Detection Result
                  </p>
                  <h2
                    className={`text-3xl font-light tracking-tight ${
                      result.is_fake ? "text-red-400" : "text-emerald-400"
                    }`}
                  >
                    {result.is_fake ? "Deepfake Detected" : "Likely Authentic"}
                  </h2>
                </div>
                <div
                  className={`w-12 h-12 rounded-full flex items-center justify-center text-2xl
                  ${result.is_fake ? "bg-red-500/20" : "bg-emerald-500/20"}`}
                >
                  {result.is_fake ? "⚠" : "✓"}
                </div>
              </div>

              {/* Confidence Display */}
              <div className="mb-6">
                <div className="flex items-baseline justify-between mb-3">
                  <span className="text-xs font-medium text-slate-400 uppercase tracking-widest">
                    Confidence Score
                  </span>
                  <span
                    className={`text-2xl font-light tracking-tight ${
                      result.is_fake ? "text-red-300" : "text-emerald-300"
                    }`}
                  >
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                {/* Confidence Bar */}
                <div className="w-full bg-slate-700/30 rounded-full h-1.5 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-1000 ${
                      result.is_fake
                        ? "bg-gradient-to-r from-red-500 to-red-400"
                        : "bg-gradient-to-r from-emerald-500 to-emerald-400"
                    }`}
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
              </div>

              {/* Model Details */}
              <p className="text-xs text-slate-400">
                Analysis powered by{" "}
                <span className="text-slate-300">EfficientNet-B4</span> +{" "}
                <span className="text-slate-300">FFT Detection</span>
              </p>
            </div>
          </div>

          {/* For Images */}
          {!isVideo && "heatmap_base64" in result && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="rounded-xl bg-slate-800/30 border border-slate-700/50 p-4 backdrop-blur-sm">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">
                  Original
                </p>
                <img
                  src={preview!}
                  alt="Original"
                  className="w-full rounded-lg object-contain max-h-72 shadow-lg"
                />
              </div>
              <div className="rounded-xl bg-slate-800/30 border border-slate-700/50 p-4 backdrop-blur-sm">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">
                  Analysis Heatmap
                </p>
                <img
                  src={`data:image/jpeg;base64,${result.heatmap_base64}`}
                  alt="GradCAM"
                  className="w-full rounded-lg object-contain max-h-72 shadow-lg"
                />
              </div>
            </div>
          )}

          {/* For Videos */}
          {isVideo && "frames_analyzed" in result && (
            <>
              <div className="rounded-xl bg-slate-800/30 border border-slate-700/50 p-5 backdrop-blur-sm">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">
                  Top Frame Heatmap
                </p>
                <img
                  src={`data:image/jpeg;base64,${result.heatmap_base64}`}
                  alt="GradCAM"
                  className="w-full rounded-lg object-contain max-h-80 shadow-lg"
                />
              </div>

              <div className="grid grid-cols-2 gap-4 px-1">
                <div className="bg-slate-800/20 rounded-lg p-4 border border-slate-700/30">
                  <p className="text-xs text-slate-400 uppercase tracking-widest font-semibold">
                    Frames Analyzed
                  </p>
                  <p className="text-2xl font-light text-white mt-2">
                    {result.frames_analyzed}
                  </p>
                </div>
                <div className="bg-slate-800/20 rounded-lg p-4 border border-slate-700/30">
                  <p className="text-xs text-slate-400 uppercase tracking-widest font-semibold">
                    Highest Confidence Frame
                  </p>
                  <p className="text-2xl font-light text-blue-400 mt-2">
                    #{result.top_frame_index}
                  </p>
                </div>
              </div>

              {/* Frame Confidence Chart */}
              <div className="rounded-xl bg-slate-800/30 border border-slate-700/50 p-6 backdrop-blur-sm">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-5">
                  Frame-by-Frame Analysis
                </p>
                <div className="space-y-2.5">
                  {result.frame_confidences.map((conf, idx) => (
                    <div key={idx} className="group">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-medium text-slate-400 w-8">
                          #{idx}
                        </span>
                        <span className="text-xs text-slate-500 ml-auto">
                          {(conf * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="h-1.5 bg-slate-700/30 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${
                            conf > 0.5
                              ? "bg-gradient-to-r from-red-500 to-red-400"
                              : "bg-gradient-to-r from-emerald-500 to-emerald-400"
                          }`}
                          style={{ width: `${conf * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Explanation */}
          <div className="rounded-xl bg-slate-800/20 border border-slate-700/30 p-6 backdrop-blur-sm">
            <p className="text-xs font-semibold text-slate-300 uppercase tracking-widest mb-3">
              How It Works
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              Our hybrid model combines{" "}
              <span className="text-slate-300 font-medium">
                spatial CNN analysis
              </span>{" "}
              (via EfficientNet-B4) with{" "}
              <span className="text-slate-300 font-medium">
                frequency domain detection
              </span>{" "}
              (FFT-based anomaly recognition). The heatmap highlights regions
              where manipulation signatures were detected — indicating areas
              with artifactual patterns or frequency inconsistencies.
            </p>
          </div>

          {/* Analyze another */}
          <button
            onClick={reset}
            className="w-full group py-3.5 rounded-lg font-medium text-sm transition-all duration-300
              bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/50 hover:border-slate-600/50
              text-white hover:shadow-lg hover:shadow-slate-500/10"
          >
            <span className="flex items-center justify-center">
              <svg
                className="w-4 h-4 mr-2 group-hover:rotate-180 transition-transform duration-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Analyze Another {isVideo ? "Video" : "Image"}
            </span>
          </button>
        </div>
      )}

      {/* Footer */}
      <footer className="mt-16 text-center">
        <p className="text-xs text-slate-500 tracking-wide">
          Deepfake Detection Engine • Hybrid CNN + FFT Analysis
        </p>
        <p className="text-xs text-slate-600 mt-1">
          HP-AI CoE 30-Day Agile Challenge
        </p>
      </footer>
    </main>
  );
}
