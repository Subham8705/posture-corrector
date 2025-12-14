import { useState, useEffect, useRef, useCallback } from 'react';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import { usePostureHistory } from './usePostureHistory';

export type PostureStatus = 'good' | 'sit-straight' | 'move-back' | 'initializing' | 'no-person';

interface PostureAnalysis {
  status: PostureStatus;
  shoulderSlope: number;
  neckAngle: number;
  confidence: number;
}

interface UsePoseDetectionOptions {
  sensitivity: number; // 0-100
  soundEnabled: boolean;
  notificationsEnabled: boolean;
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
  stats: {
    goodDuration: number;
    badDuration: number;
  };
  resetStats: () => void;
  sessionHistory: Array<{ timestamp: number; score: number }>;
}

// Smoothing buffer for pose data
const SMOOTHING_BUFFER_SIZE = 5;
const BAD_POSTURE_THRESHOLD_MS = 2000;
const NOTIFICATION_COOLDOWN_MS = 5000;
const AUDIO_COOLDOWN_MS = 3000;
const HISTORY_UPDATE_INTERVAL = 1000;

export function usePoseDetection({
  sensitivity,
  soundEnabled,
  notificationsEnabled,
  onStatusChange,
}: UsePoseDetectionOptions): UsePoseDetectionReturn {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const keepAliveOscillatorRef = useRef<OscillatorNode | null>(null);
  const keepAliveGainRef = useRef<GainNode | null>(null);

  const [status, setStatus] = useState<PostureStatus>('initializing');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [analysis, setAnalysis] = useState<PostureAnalysis | null>(null);
  const [sessionHistory, setSessionHistory] = useState<Array<{ timestamp: number; score: number }>>([]);

  const { saveDailyStats } = usePostureHistory();
  const lastHistoryUpdateRef = useRef<number>(0);

  // Baseline values for personalized detection
  const baselineRef = useRef<{
    shoulderSlope: number;
    neckAngle: number;
    faceSize: number;
    headYaw: number;
    spinalRatio: number;
  } | null>(null);

  const [stats, setStats] = useState({ goodDuration: 0, badDuration: 0 });
  const lastStatsUpdateRef = useRef<number>(0);

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
  const lastAudioTimeRef = useRef<number>(0);

  const resetBaseline = useCallback(() => {
    baselineRef.current = null;
    shoulderSlopeBuffer.current = [];
    neckAngleBuffer.current = [];
    faceSizeBuffer.current = [];
    spinalRatioBuffer.current = [];
    badPostureStartRef.current = null;
    lastNotificationTimeRef.current = 0;
    lastAudioTimeRef.current = 0;
  }, []);

  const resetStats = useCallback(() => {
    setStats({ goodDuration: 0, badDuration: 0 });
    setSessionHistory([]);
    lastStatsUpdateRef.current = Date.now();
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
    if (!notificationsEnabled) return;
    if (
      'Notification' in window &&
      Notification.permission === 'granted' &&
      document.visibilityState === 'hidden' &&
      Date.now() - lastNotificationTimeRef.current > NOTIFICATION_COOLDOWN_MS
    ) {
      new Notification('PosturePal', {
        body: message,
        icon: 'https://www.shutterstock.com/image-vector/correct-posture-position-line-icon-260nw-2002371824.jpg',
        silent: true,
      });
      lastNotificationTimeRef.current = Date.now();
    }
  };

  const playAlertSound = useCallback(() => {
    if (!soundEnabled) return;
    if (Date.now() - lastAudioTimeRef.current < AUDIO_COOLDOWN_MS) return;

    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }

      const ctx = audioContextRef.current;
      if (ctx.state === 'suspended') {
        ctx.resume();
      }

      const oscillator = ctx.createOscillator();
      const gainNode = ctx.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(ctx.destination);

      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(880, ctx.currentTime);
      oscillator.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.1);

      gainNode.gain.setValueAtTime(0.1, ctx.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.1);

      oscillator.start(ctx.currentTime);
      oscillator.stop(ctx.currentTime + 0.1);

      lastAudioTimeRef.current = Date.now();
    } catch (e) {
      console.error('Audio playback failed', e);
    }
  }, []);

  const toggleSilentAudio = useCallback((enable: boolean) => {
    if (enable) {
      try {
        if (!audioContextRef.current) {
          audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        }
        const ctx = audioContextRef.current;
        if (ctx.state === 'suspended') ctx.resume();

        if (!keepAliveOscillatorRef.current) {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();

          osc.type = 'sine';
          osc.frequency.setValueAtTime(440, ctx.currentTime);
          gain.gain.setValueAtTime(0.0001, ctx.currentTime); // Inaudible

          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start();

          keepAliveOscillatorRef.current = osc;
          keepAliveGainRef.current = gain;
        }
      } catch (e) {
        console.error('Silent audio failed:', e);
      }
    } else {
      if (keepAliveOscillatorRef.current) {
        try {
          keepAliveOscillatorRef.current.stop();
          keepAliveOscillatorRef.current.disconnect();
          keepAliveGainRef.current?.disconnect();
        } catch (e) { /* ignore */ }
        keepAliveOscillatorRef.current = null;
        keepAliveGainRef.current = null;
      }
    }
  }, []);

  const analyzePosture = useCallback(
    (landmarks: any[]): PostureAnalysis => {
      const leftShoulder = landmarks[11];
      const rightShoulder = landmarks[12];
      const leftEar = landmarks[7];
      const rightEar = landmarks[8];
      const nose = landmarks[0];

      const rawShoulderSlope = Math.abs(leftShoulder.y - rightShoulder.y);
      const shoulderSlope = getSmoothedValue(shoulderSlopeBuffer.current, rawShoulderSlope);

      const shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2;
      const rawNeckAngle = nose.x - shoulderMidX;
      const neckAngle = getSmoothedValue(neckAngleBuffer.current, Math.abs(rawNeckAngle));

      const earMidX = (leftEar.x + rightEar.x) / 2;
      const rawHeadYaw = nose.x - earMidX;
      const headYaw = getSmoothedValue(headYawBuffer.current, Math.abs(rawHeadYaw));

      const rawFaceSize = Math.abs(leftEar.x - rightEar.x);
      const faceSize = getSmoothedValue(faceSizeBuffer.current, rawFaceSize);

      const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
      const rawSpinalDistance = Math.abs(shoulderMidY - nose.y);
      const rawSpinalRatio = rawSpinalDistance / rawFaceSize;
      const spinalRatio = getSmoothedValue(spinalRatioBuffer.current, rawSpinalRatio);

      const confidence = (leftShoulder.visibility + rightShoulder.visibility + nose.visibility) / 3;

      if (!baselineRef.current && shoulderSlopeBuffer.current.length >= SMOOTHING_BUFFER_SIZE) {
        baselineRef.current = {
          shoulderSlope,
          neckAngle,
          faceSize,
          headYaw,
          spinalRatio,
        };
      }

      const sensitivityMultiplier = 1 + (sensitivity - 50) / 100;
      const baseline = baselineRef.current || { shoulderSlope: 0.03, neckAngle: 0.05, faceSize: 0.15, headYaw: 0.02, spinalRatio: 1.5 };

      const shoulderThreshold = 0.04 / sensitivityMultiplier;
      const neckThreshold = 0.06 / sensitivityMultiplier;
      const headYawThreshold = 0.03 / sensitivityMultiplier;
      const spinalRatioThreshold = 0.18 / (sensitivityMultiplier * 1.2);
      const distanceThreshold = baseline.faceSize * 1.4 / sensitivityMultiplier;

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
        rawStatus = 'sit-straight';
      }

      let finalStatus = rawStatus;

      if (rawStatus !== 'good') {
        if (!badPostureStartRef.current) {
          badPostureStartRef.current = Date.now();
        }
        const badDuration = Date.now() - badPostureStartRef.current;
        if (badDuration < BAD_POSTURE_THRESHOLD_MS) {
          finalStatus = 'good';
        } else {
          const message = rawStatus === 'move-back' ? 'You are too close to the screen!' : 'Sit up straight!';
          sendNotification(message);
          playAlertSound();
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
    [sensitivity, playAlertSound, notificationsEnabled]
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

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
      const results = poseLandmarkerRef.current.detectForVideo(video, performance.now());

      if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        const drawingUtils = new DrawingUtils(ctx);

        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
          color: 'rgba(16, 185, 129, 0.7)',
          lineWidth: 3,
        });

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

        const postureAnalysis = analyzePosture(landmarks);
        setAnalysis(postureAnalysis);

        if (postureAnalysis.status !== prevStatusRef.current) {
          onStatusChange?.(postureAnalysis.status, prevStatusRef.current);
          prevStatusRef.current = postureAnalysis.status;
        }
        setStatus(postureAnalysis.status);

        if (lastStatsUpdateRef.current > 0) {
          const now = Date.now();
          const timeDiff = now - lastStatsUpdateRef.current;
          setStats(prev => {
            let goodDelta = 0;
            let badDelta = 0;
            if (postureAnalysis.status === 'good') {
              goodDelta = timeDiff;
            } else if (['sit-straight', 'move-back'].includes(postureAnalysis.status)) {
              badDelta = timeDiff;
            }

            if (goodDelta > 0 || badDelta > 0) {
              saveDailyStats(goodDelta, badDelta);
            }

            return {
              goodDuration: prev.goodDuration + goodDelta,
              badDuration: prev.badDuration + badDelta
            };
          });
          lastStatsUpdateRef.current = now;
        } else {
          lastStatsUpdateRef.current = Date.now();
        }

        // Update session history
        const now = Date.now();
        if (now - lastHistoryUpdateRef.current > HISTORY_UPDATE_INTERVAL) {
          const score = postureAnalysis.status === 'good' ? 1 : 0;
          setSessionHistory(prev => {
            const newHistory = [...prev, { timestamp: now, score }];
            // Keep roughly last 10 minutes of detailed history (600 points)
            if (newHistory.length > 600) return newHistory.slice(-600);
            return newHistory;
          });

          // Also save to daily stats roughly every second (accumulated duration)
          // We calculate differential since last update
          const timeSinceLastUpdate = now - lastHistoryUpdateRef.current;
          // Only save if meaningful time passed
          if (timeSinceLastUpdate < 5000) { // Limit frequency of writes/updates slightly for safety if needed, 
            // actually let's blindly trust the timeDiff from stats update logic above?
            // The stats update logic accumulates to state.
            // Let's use the `timeDiff` calculated above for stats.
          }

          lastHistoryUpdateRef.current = now;
        }

        // Persist to local storage periodically (e.g., every 5 seconds) or just piggyback on this loop?
        // Let's use the stats state accumulation. 
        // Better: We should pass the Delta time to saveDailyStats.

      } else {
        setStatus('no-person');
        setAnalysis(null);
        lastStatsUpdateRef.current = Date.now();
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

    if ('Notification' in window && Notification.permission === 'default') {
      try {
        await Notification.requestPermission();
      } catch (e) {
        console.warn('Notification permission request failed', e);
      }
    }
    try {
      toggleSilentAudio(true);

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
        videoRef.current.setAttribute('playsinline', 'true');
        await videoRef.current.play();
      }

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
      resetStats();
      lastStatsUpdateRef.current = Date.now();
      detectPose();
    } catch (err) {
      console.error('Failed to start detection:', err);
      setError(err instanceof Error ? err.message : 'Failed to start camera');
      toggleSilentAudio(false);
    } finally {
      setIsLoading(false);
    }
  }, [detectPose, resetBaseline, toggleSilentAudio, resetStats]);

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
    toggleSilentAudio(false);
  }, [toggleSilentAudio]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden && isRunning) {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
        if (!timerRef.current) {
          detectPose();
        }
      } else if (!document.hidden && isRunning) {
        if (timerRef.current) {
          clearTimeout(timerRef.current);
          timerRef.current = null;
        }
        if (!animationFrameRef.current) {
          detectPose();
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [isRunning, detectPose]);

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
    stats,
    resetStats,
    sessionHistory,
  };
}
