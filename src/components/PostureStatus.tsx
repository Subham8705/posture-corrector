import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, AlertTriangle, MoveHorizontal, User, Loader2 } from 'lucide-react';
import type { PostureStatus as PostureStatusType } from '@/hooks/usePoseDetection';

interface PostureStatusProps {
  status: PostureStatusType;
  analysis?: {
    shoulderSlope: number;
    neckAngle: number;
    confidence: number;
  } | null;
}

const statusConfig = {
  good: {
    icon: CheckCircle2,
    label: 'Good Posture',
    description: 'Keep it up! Your posture looks great.',
    bgClass: 'bg-status-good-bg',
    textClass: 'text-status-good',
    borderClass: 'border-status-good/30',
    glowClass: 'shadow-glow-good',
  },
  'sit-straight': {
    icon: AlertTriangle,
    label: 'Sit Up Straight',
    description: 'Your shoulders are uneven or you\'re leaning forward.',
    bgClass: 'bg-status-warning-bg',
    textClass: 'text-status-warning',
    borderClass: 'border-status-warning/30',
    glowClass: 'shadow-glow-warning',
  },
  'move-back': {
    icon: MoveHorizontal,
    label: 'Move Back',
    description: 'You\'re too close to the screen. Move back a bit.',
    bgClass: 'bg-status-danger-bg',
    textClass: 'text-status-danger',
    borderClass: 'border-status-danger/30',
    glowClass: 'shadow-glow-danger',
  },
  'no-person': {
    icon: User,
    label: 'No Person Detected',
    description: 'Make sure you\'re visible in the camera frame.',
    bgClass: 'bg-muted',
    textClass: 'text-muted-foreground',
    borderClass: 'border-border',
    glowClass: '',
  },
  initializing: {
    icon: Loader2,
    label: 'Initializing',
    description: 'Setting up pose detection...',
    bgClass: 'bg-muted',
    textClass: 'text-muted-foreground',
    borderClass: 'border-border',
    glowClass: '',
  },
};

export function PostureStatus({ status, analysis }: PostureStatusProps) {
  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={status}
        initial={{ opacity: 0, y: 10, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -10, scale: 0.98 }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
        className={`rounded-2xl border-2 p-6 transition-shadow duration-500 ${config.bgClass} ${config.borderClass} ${config.glowClass}`}
      >
        <div className="flex items-start gap-4">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.1, type: 'spring', stiffness: 200 }}
            className={`flex h-14 w-14 items-center justify-center rounded-xl ${config.bgClass}`}
          >
            <Icon
              className={`h-8 w-8 ${config.textClass} ${status === 'initializing' ? 'animate-spin' : ''}`}
            />
          </motion.div>
          <div className="flex-1">
            <h3 className={`text-xl font-semibold ${config.textClass}`}>{config.label}</h3>
            <p className="mt-1 text-muted-foreground">{config.description}</p>
          </div>
        </div>

        {analysis && status !== 'initializing' && status !== 'no-person' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            transition={{ delay: 0.2 }}
            className="mt-4 grid grid-cols-2 gap-3"
          >
            <div className="rounded-lg bg-background/50 p-3">
              <p className="text-xs font-medium text-muted-foreground">Shoulder Tilt</p>
              <p className="mt-1 text-lg font-semibold">{analysis.shoulderSlope.toFixed(1)}°</p>
            </div>
            <div className="rounded-lg bg-background/50 p-3">
              <p className="text-xs font-medium text-muted-foreground">Forward Lean</p>
              <p className="mt-1 text-lg font-semibold">{analysis.neckAngle.toFixed(1)}°</p>
            </div>
          </motion.div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
