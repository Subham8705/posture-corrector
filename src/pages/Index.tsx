import { useState, useCallback, useEffect } from 'react';
import { Helmet } from 'react-helmet-async';
import { Header } from '@/components/Header';
import { CameraView } from '@/components/CameraView';
import { PostureStatus } from '@/components/PostureStatus';
import { ControlPanel } from '@/components/ControlPanel';
import { Tips } from '@/components/Tips';
import { StatsCard } from '@/components/StatsCard';
import { usePoseDetection } from '@/hooks/usePoseDetection';
import { usePostureAlerts } from '@/hooks/usePostureAlerts';
import { toast } from 'sonner';

const Index = () => {
  const [sensitivity, setSensitivity] = useState(() => {
    const saved = localStorage.getItem('posture-sensitivity');
    return saved ? parseInt(saved, 10) : 50;
  });

  // Persist sensitivity changes
  useEffect(() => {
    localStorage.setItem('posture-sensitivity', sensitivity.toString());
  }, [sensitivity]);

  const [soundEnabled, setSoundEnabled] = useState(true);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);

  const { triggerAlert, notificationPermission, requestNotificationPermission } = usePostureAlerts({
    soundEnabled,
    notificationsEnabled,
  });

  const handleStatusChange = useCallback(
    (status: any, prevStatus: any) => {
      triggerAlert(status, prevStatus);
    },
    [triggerAlert]
  );

  const {
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
  } = usePoseDetection({
    sensitivity,
    onStatusChange: handleStatusChange,
  });

  const handleStart = async () => {
    try {
      await startDetection();
      toast.success('Posture monitoring started', {
        description: 'Sit in your ideal posture to calibrate.',
      });
    } catch (err) {
      toast.error('Failed to start camera', {
        description: error || 'Please allow camera access and try again.',
      });
    }
  };

  const handleStop = () => {
    stopDetection();
    toast.info('Monitoring stopped');
  };

  const handleResetBaseline = () => {
    resetBaseline();
    toast.success('Baseline reset', {
      description: 'Sit in your ideal posture to recalibrate.',
    });
  };

  const handleRequestNotifications = async () => {
    await requestNotificationPermission();
    if (Notification.permission === 'granted') {
      setNotificationsEnabled(true);
      toast.success('Notifications enabled');
    } else if (Notification.permission === 'denied') {
      toast.error('Notifications blocked', {
        description: 'Please enable notifications in your browser settings.',
      });
    }
  };

  return (
    <>
      <Helmet>
        <title>PosturePal - Real-Time Posture Corrector</title>
        <meta
          name="description"
          content="Monitor and improve your posture in real-time using your webcam. Get instant feedback and alerts when you slouch."
        />
      </Helmet>

      <div className="min-h-screen bg-background">
        <Header />

        <main className="container mx-auto px-4 py-6">
          <div className="grid gap-6 lg:grid-cols-5">
            {/* Left Panel - Camera View */}
            <div className="space-y-6 lg:col-span-3">
              <CameraView
                videoRef={videoRef}
                canvasRef={canvasRef}
                isRunning={isRunning}
                isLoading={isLoading}
                status={status}
                onStart={handleStart}
                onStop={handleStop}
                onRecalibrate={handleResetBaseline}
              />
              <StatsCard stats={stats} onReset={resetStats} />
            </div>

            {/* Right Panel - Status and Controls */}
            <div className="space-y-6 lg:col-span-2">
              <PostureStatus status={status} analysis={analysis} />

              <ControlPanel
                sensitivity={sensitivity}
                onSensitivityChange={setSensitivity}
                soundEnabled={soundEnabled}
                onSoundToggle={setSoundEnabled}
                notificationsEnabled={notificationsEnabled}
                onNotificationsToggle={setNotificationsEnabled}
                notificationPermission={notificationPermission}
                onRequestNotificationPermission={handleRequestNotifications}
                onResetBaseline={handleResetBaseline}
                isRunning={isRunning}
              />
              <Tips />
            </div>
          </div>
        </main>
      </div>
    </>
  );
};

export default Index;
