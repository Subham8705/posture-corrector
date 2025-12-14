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
import { PostureGraph } from '@/components/PostureGraph';
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

  // usePostureAlerts used ONLY for permission request logic now
  const { notificationPermission, requestNotificationPermission } = usePostureAlerts({
    soundEnabled,
    notificationsEnabled,
  });

  // We don't need handleStatusChange to trigger alerts anymore, 
  // because usePoseDetection handles them internally with the toggles.

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
    sessionHistory,
  } = usePoseDetection({
    sensitivity,
    soundEnabled,
    notificationsEnabled,
    // onStatusChange removed as it was only used for alerts
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
        <title>PosturePal AI - Free Real-Time Posture Coach</title>
        <meta
          name="description"
          content="Improve your posture instantly with PosturePal AI. A free, privacy-first webcam posture corrector that runs entirely in your browser."
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
            </div>

            {/* Full Width - Graphs and Tips */}
            <div className="lg:col-span-5 space-y-6">
              <PostureGraph sessionHistory={sessionHistory} />
              <Tips />
            </div>
          </div>
        </main>
      </div>
    </>
  );
};

export default Index;
