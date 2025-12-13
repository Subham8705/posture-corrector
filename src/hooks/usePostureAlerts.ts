import { useCallback, useRef, useState, useEffect } from 'react';
import type { PostureStatus } from './usePoseDetection';

interface UsePostureAlertsOptions {
  soundEnabled: boolean;
  notificationsEnabled: boolean;
  cooldownMs?: number;
}

interface UsePostureAlertsReturn {
  triggerAlert: (status: PostureStatus, prevStatus: PostureStatus) => void;
  notificationPermission: NotificationPermission | 'default';
  requestNotificationPermission: () => Promise<void>;
}

const ALERT_COOLDOWN_MS = 10000; // 10 seconds between alerts

export function usePostureAlerts({
  soundEnabled,
  notificationsEnabled,
  cooldownMs = ALERT_COOLDOWN_MS,
}: UsePostureAlertsOptions): UsePostureAlertsReturn {
  const lastAlertTimeRef = useRef<number>(0);
  const audioContextRef = useRef<AudioContext | null>(null);
  const [notificationPermission, setNotificationPermission] = useState<NotificationPermission>('default');

  // Check notification permission on mount
  useEffect(() => {
    if ('Notification' in window) {
      setNotificationPermission(Notification.permission);
    }
  }, []);

  const requestNotificationPermission = useCallback(async () => {
    if (!('Notification' in window)) return;

    try {
      const permission = await Notification.requestPermission();
      setNotificationPermission(permission);
    } catch (err) {
      console.error('Failed to request notification permission:', err);
    }
  }, []);

  const playAlertSound = useCallback(() => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }

      const ctx = audioContextRef.current;
      const oscillator = ctx.createOscillator();
      const gainNode = ctx.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(ctx.destination);

      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(440, ctx.currentTime);
      oscillator.frequency.setValueAtTime(520, ctx.currentTime + 0.1);
      
      gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);

      oscillator.start(ctx.currentTime);
      oscillator.stop(ctx.currentTime + 0.3);
    } catch (err) {
      console.error('Failed to play alert sound:', err);
    }
  }, []);

  const showNotification = useCallback((status: PostureStatus) => {
    if (!('Notification' in window) || Notification.permission !== 'granted') return;

    const messages: Record<PostureStatus, { title: string; body: string }> = {
      'sit-straight': {
        title: 'Posture Check',
        body: 'Time to sit up straight! Your shoulders are uneven.',
      },
      'move-back': {
        title: 'Too Close to Screen',
        body: 'Move back from your screen to reduce eye strain.',
      },
      good: { title: '', body: '' },
      initializing: { title: '', body: '' },
      'no-person': { title: '', body: '' },
    };

    const message = messages[status];
    if (message.title) {
      new Notification(message.title, {
        body: message.body,
        icon: '/favicon.ico',
        tag: 'posture-alert',
      });
    }
  }, []);

  const triggerAlert = useCallback(
    (status: PostureStatus, prevStatus: PostureStatus) => {
      // Only alert when transitioning from good to bad
      if (prevStatus !== 'good' || status === 'good' || status === 'initializing' || status === 'no-person') {
        return;
      }

      const now = Date.now();
      if (now - lastAlertTimeRef.current < cooldownMs) {
        return;
      }

      lastAlertTimeRef.current = now;

      if (soundEnabled) {
        playAlertSound();
      }

      if (notificationsEnabled) {
        showNotification(status);
      }
    },
    [soundEnabled, notificationsEnabled, cooldownMs, playAlertSound, showNotification]
  );

  return {
    triggerAlert,
    notificationPermission,
    requestNotificationPermission,
  };
}
