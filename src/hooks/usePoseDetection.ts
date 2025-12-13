import { useState, useEffect, useRef, useCallback } from 'react';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

export type PostureStatus = 'good' | 'sit-straight' | 'move-back' | 'initializing' | 'no-person';

interface PostureAnalysis {
  status: PostureStatus;
  shoulderSlope: number;
  neckAngle: number;
  confidence: number;
}

interface UsePoseDetectionOptions {
  sensitivity: number; // 0-100
  onStatusChange?: (status: PostureStatus, prevStatus: PostureStatus) => void;
}

interface UsePoseDetectionReturn {
  canvasRef: React.RefObject<HTMLCanvasElement>;
  videoRef: React.RefObject<HTMLVideoElement>;
  status: PostureStatus;
  isLoading: boolean;
  error: string | null;
  startDetection: () => Promise<void>;
  stopDetection: () => void;
  resetBaseline: () => void;
  isRunning: boolean;
  analysis: PostureAnalysis | null;
}

// Smoothing buffer for pose data
const SMOOTHING_BUFFER_SIZE = 5;
const BAD_POSTURE_THRESHOLD_MS = 2000;
const NOTIFICATION_COOLDOWN_MS = 5000;

export function usePoseDetection({
  sensitivity,
  onStatusChange,
}: UsePoseDetectionOptions): UsePoseDetectionReturn {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [status, setStatus] = useState<PostureStatus>('initializing');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [analysis, setAnalysis] = useState<PostureAnalysis | null>(null);

  // Baseline values for personalized detection
  const baselineRef = useRef<{
    shoulderSlope: number;
    neckAngle: number;
    faceSize: number;
    headYaw: number;
    spinalRatio: number;
  } | null>(null);

  // Smoothing buffers
  const shoulderSlopeBuffer = useRef<number[]>([]);
  const neckAngleBuffer = useRef<number[]>([]);
  const faceSizeBuffer = useRef<number[]>([]);
  const headYawBuffer = useRef<number[]>([]);
  const spinalRatioBuffer = useRef<number[]>([]);

  // Bad posture timing and notification
  const badPostureStartRef = useRef<number | null>(null);
  const prevStatusRef = useRef<PostureStatus>('initializing');
  const lastNotificationTimeRef = useRef<number>(0);

  const resetBaseline = useCallback(() => {
    baselineRef.current = null;
    shoulderSlopeBuffer.current = [];
    neckAngleBuffer.current = [];
    faceSizeBuffer.current = [];
    headYawBuffer.current = [];
    spinalRatioBuffer.current = [];
    badPostureStartRef.current = null;
    lastNotificationTimeRef.current = 0;
  }, []);

  const getSmoothedValue = (buffer: number[], newValue: number): number => {
    buffer.push(newValue);
    if (buffer.length > SMOOTHING_BUFFER_SIZE) {
      buffer.shift();
    }
    const sum = buffer.reduce((a, b) => a + b, 0);
    return sum / buffer.length;
  };

  const sendNotification = (message: string) => {
    if (
      'Notification' in window &&
      Notification.permission === 'granted' &&
      document.visibilityState === 'hidden' &&
      Date.now() - lastNotificationTimeRef.current > NOTIFICATION_COOLDOWN_MS
    ) {
      new Notification('Posture Pal', {
        body: message,
        icon: '/icon.png',
        silent: true,
      });
      lastNotificationTimeRef.current = Date.now();
    }
  };

  const analyzePosture = useCallback(
    (landmarks: any[]): PostureAnalysis => {
      // MediaPipe Pose landmark indices
      const leftShoulder = landmarks[11];
      const rightShoulder = landmarks[12];
      const leftEar = landmarks[7];
      const rightEar = landmarks[8];
      const nose = landmarks[0];

      // Calculate shoulder slope (difference in Y between shoulders)
      const rawShoulderSlope = Math.abs(leftShoulder.y - rightShoulder.y);
      const shoulderSlope = getSmoothedValue(shoulderSlopeBuffer.current, rawShoulderSlope);

      // Calculate neck angle (head forward position relative to shoulders)
      const shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2;
      const rawNeckAngle = nose.x - shoulderMidX;
      const neckAngle = getSmoothedValue(neckAngleBuffer.current, Math.abs(rawNeckAngle));

      // Calculate head yaw (nose position relative to ears center)
      const earMidX = (leftEar.x + rightEar.x) / 2;
      const rawHeadYaw = nose.x - earMidX;
      const headYaw = getSmoothedValue(headYawBuffer.current, Math.abs(rawHeadYaw));

      // Calculate face size (proxy for distance from screen)
      const rawFaceSize = Math.abs(leftEar.x - rightEar.x);
      const faceSize = getSmoothedValue(faceSizeBuffer.current, rawFaceSize);

      // Calculate vertical spinal ratio (nose to shoulder vertical distance normalized by face size)
      // If user slouches, this vertical distance decreases relative to their face width
      const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
      const rawSpinalDistance = Math.abs(shoulderMidY - nose.y); // Vertical distance
      // We normalize by face size so moving back/forth doesn't trigger it.
      // Ratio = Vertical Distance / Horizontal Face Width
      const rawSpinalRatio = rawSpinalDistance / rawFaceSize;
      const spinalRatio = getSmoothedValue(spinalRatioBuffer.current, rawSpinalRatio);

      // Average confidence
      const confidence = (leftShoulder.visibility + rightShoulder.visibility + nose.visibility) / 3;

      // Set baseline if not set
      if (!baselineRef.current && shoulderSlopeBuffer.current.length >= SMOOTHING_BUFFER_SIZE) {
        baselineRef.current = {
          shoulderSlope,
          neckAngle,
          faceSize,
          headYaw,
          spinalRatio,
        };
      }

      // Sensitivity affects thresholds (higher sensitivity = stricter)
      const sensitivityMultiplier = 1 + (sensitivity - 50) / 100;

      // Thresholds relative to baseline
      const baseline = baselineRef.current || { shoulderSlope: 0.03, neckAngle: 0.05, faceSize: 0.15, headYaw: 0.02, spinalRatio: 1.5 };

      const shoulderThreshold = 0.04 / sensitivityMultiplier;
      const neckThreshold = 0.06 / sensitivityMultiplier;
      const headYawThreshold = 0.03 / sensitivityMultiplier;
      // Adjusted to be responsive: High sens (1.5) -> ~0.1 (Strict). Low sens (0.5) -> ~0.3 (Loose).
      const spinalRatioThreshold = 0.18 / (sensitivityMultiplier * 1.2);
      const distanceThreshold = baseline.faceSize * 1.4 / sensitivityMultiplier;


      // Determine raw status
      let rawStatus: PostureStatus = 'good';

      if (faceSize > distanceThreshold) {
        rawStatus = 'move-back';
      } else if (shoulderSlope > baseline.shoulderSlope + shoulderThreshold) {
        rawStatus = 'sit-straight';
      } else if (neckAngle > baseline.neckAngle + neckThreshold) {
        rawStatus = 'sit-straight';
      } else if (headYaw > baseline.headYaw + headYawThreshold) {
        rawStatus = 'sit-straight';
      } else if (spinalRatio < baseline.spinalRatio - spinalRatioThreshold) {
        // Current ratio is significantly less than baseline (implosion/slouching)
        rawStatus = 'sit-straight';
      }

      // Apply time threshold for bad posture
      let finalStatus = rawStatus;

      if (rawStatus !== 'good') {
        if (!badPostureStartRef.current) {
          badPostureStartRef.current = Date.now();
        }
        const badDuration = Date.now() - badPostureStartRef.current;
        if (badDuration < BAD_POSTURE_THRESHOLD_MS) {
          finalStatus = 'good'; // Not bad long enough
        } else {
          const message = rawStatus === 'move-back' ? 'You are too close to the screen!' : 'Sit up straight!';
          sendNotification(message);
        }
      } else {
        badPostureStartRef.current = null;
      }

      return {
        status: finalStatus,
        shoulderSlope: shoulderSlope * 100,
        neckAngle: neckAngle * 100,
        confidence,
      };
    },
    [sensitivity]
  );

  const detectPose = useCallback(() => {
    if (!poseLandmarkerRef.current || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx || video.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(detectPose);
      return;
    }

    // Match canvas size to video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Clear and draw video frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
      const results = poseLandmarkerRef.current.detectForVideo(video, performance.now());

      if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];

        // Draw skeleton
        const drawingUtils = new DrawingUtils(ctx);

        // Custom styling for connections
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
          color: 'rgba(16, 185, 129, 0.7)',
          lineWidth: 3,
        });

        // Draw landmarks with different colors for key points
        landmarks.forEach((landmark, index) => {
          const isKeyPoint = [0, 7, 8, 11, 12].includes(index);
          const radius = isKeyPoint ? 8 : 4;
          const color = isKeyPoint ? 'hsl(158, 64%, 42%)' : 'rgba(16, 185, 129, 0.5)';

          ctx.beginPath();
          ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, radius, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();

          if (isKeyPoint) {
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        });

        // Analyze posture
        const postureAnalysis = analyzePosture(landmarks);
        setAnalysis(postureAnalysis);

        if (postureAnalysis.status !== prevStatusRef.current) {
          onStatusChange?.(postureAnalysis.status, prevStatusRef.current);
          prevStatusRef.current = postureAnalysis.status;
        }
        setStatus(postureAnalysis.status);
      } else {
        setStatus('no-person');
        setAnalysis(null);
      }
    } catch (err) {
      console.error('Pose detection error:', err);
    }

    if (document.hidden) {
      timerRef.current = setTimeout(detectPose, 1000);
    } else {
      animationFrameRef.current = requestAnimationFrame(detectPose);
    }
  }, [analyzePosture, onStatusChange]);

  const startDetection = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    // Request Notification Permission
    if ('Notification' in window && Notification.permission === 'default') {
      try {
        await Notification.requestPermission();
      } catch (e) {
        console.warn('Notification permission request failed', e);
      }
    }
    try {
      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // Initialize MediaPipe
      if (!poseLandmarkerRef.current) {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );

        poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numPoses: 1,
        });
      }

      setIsRunning(true);
      setStatus('good');
      resetBaseline();
      animationFrameRef.current = requestAnimationFrame(detectPose);
    } catch (err) {
      console.error('Failed to start detection:', err);
      setError(err instanceof Error ? err.message : 'Failed to start camera');
    } finally {
      setIsLoading(false);
    }
  }, [detectPose, resetBaseline]);

  const stopDetection = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsRunning(false);
    setStatus('initializing');
    setAnalysis(null);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopDetection();
      if (poseLandmarkerRef.current) {
        poseLandmarkerRef.current.close();
      }
    };
  }, [stopDetection]);

  return {
    canvasRef,
    videoRef,
    status,
    isLoading,
    error,
    startDetection,
    stopDetection,
    resetBaseline,
    isRunning,
    analysis,
  };
}
