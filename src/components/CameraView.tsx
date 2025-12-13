import { motion } from 'framer-motion';
import { Video, VideoOff, Loader2, RotateCcw } from 'lucide-react';
import { forwardRef } from 'react';
import type { PostureStatus } from '@/hooks/usePoseDetection';

interface CameraViewProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  isRunning: boolean;
  isLoading: boolean;
  status: PostureStatus;
  onStart: () => void;
  onStop: () => void;
  onRecalibrate: () => void;
}

const statusBorderColors = {
  good: 'ring-status-good',
  'sit-straight': 'ring-status-warning',
  'move-back': 'ring-status-danger',
  'no-person': 'ring-muted-foreground',
  initializing: 'ring-primary',
};

export const CameraView = forwardRef<HTMLDivElement, CameraViewProps>(
  ({ videoRef, canvasRef, isRunning, isLoading, status, onStart, onStop, onRecalibrate }, ref) => {
    const ringColor = isRunning ? statusBorderColors[status] : 'ring-border';

    return (
      <motion.div
        ref={ref}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="relative overflow-hidden rounded-2xl bg-card shadow-lg"
      >
        {/* Video Container */}
        <div
          className={`relative aspect-video overflow-hidden rounded-2xl ring-4 transition-all duration-500 ${ringColor}`}
        >
          {/* Video Element (hidden, used for pose detection) */}
          <video
            ref={videoRef}
            className="absolute inset-0 h-full w-full scale-x-[-1] object-cover"
            playsInline
            muted
          />

          {/* Canvas Overlay for Skeleton */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 h-full w-full scale-x-[-1] object-cover"
          />

          {/* Placeholder when not running */}
          {!isRunning && (
            <div className="absolute inset-0 flex items-center justify-center bg-muted">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-center"
              >
                {isLoading ? (
                  <div className="flex flex-col items-center gap-3">
                    <Loader2 className="h-12 w-12 animate-spin text-primary" />
                    <p className="text-muted-foreground">Starting camera...</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-4">
                    <div className="flex h-20 w-20 items-center justify-center rounded-full bg-primary/10">
                      <VideoOff className="h-10 w-10 text-primary" />
                    </div>
                    <div>
                      <p className="text-lg font-medium text-foreground">Camera Off</p>
                      <p className="mt-1 text-sm text-muted-foreground">
                        Click Start to begin posture monitoring
                      </p>
                    </div>
                  </div>
                )}
              </motion.div>
            </div>
          )}

          {/* Live indicator */}
          {isRunning && (
            <>
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="absolute left-4 top-4 flex items-center gap-2 rounded-full bg-card/90 px-3 py-1.5 backdrop-blur-sm"
              >
                <span className="relative flex h-2.5 w-2.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-status-danger opacity-75" />
                  <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-status-danger" />
                </span>
                <span className="text-sm font-medium">LIVE</span>
              </motion.div>

              <motion.button
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onRecalibrate}
                className="absolute right-4 top-4 flex items-center gap-2 rounded-full bg-card/90 px-3 py-1.5 backdrop-blur-sm transition-colors hover:bg-card text-sm font-medium"
                title="Recalibrate Posture"
              >
                <RotateCcw className="h-4 w-4" />
                <span className="hidden sm:inline">Recalibrate</span>
              </motion.button>
            </>
          )}
        </div>

        {/* Controls */}
        <div className="p-4">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={isRunning ? onStop : onStart}
            disabled={isLoading}
            className={`flex w-full items-center justify-center gap-2 rounded-xl px-6 py-3 font-medium transition-colors ${isRunning
              ? 'bg-destructive text-destructive-foreground hover:bg-destructive/90'
              : 'bg-primary text-primary-foreground hover:bg-primary/90'
              } disabled:cursor-not-allowed disabled:opacity-50`}
          >
            {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : isRunning ? (
              <>
                <VideoOff className="h-5 w-5" />
                Stop Monitoring
              </>
            ) : (
              <>
                <Video className="h-5 w-5" />
                Start Monitoring
              </>
            )}
          </motion.button>
        </div>
      </motion.div>
    );
  }
);

CameraView.displayName = 'CameraView';
