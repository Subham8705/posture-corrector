import { motion } from 'framer-motion';
import { Volume2, VolumeX, Bell, BellOff, RotateCcw, Settings } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';

interface ControlPanelProps {
  sensitivity: number;
  onSensitivityChange: (value: number) => void;
  soundEnabled: boolean;
  onSoundToggle: (enabled: boolean) => void;
  notificationsEnabled: boolean;
  onNotificationsToggle: (enabled: boolean) => void;
  notificationPermission: NotificationPermission | 'default';
  onRequestNotificationPermission: () => void;
  onResetBaseline: () => void;
  isRunning: boolean;
}

export function ControlPanel({
  sensitivity,
  onSensitivityChange,
  soundEnabled,
  onSoundToggle,
  notificationsEnabled,
  onNotificationsToggle,
  notificationPermission,
  onRequestNotificationPermission,
  onResetBaseline,
  isRunning,
}: ControlPanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="rounded-2xl border border-border bg-card p-6 shadow-sm"
    >
      <div className="mb-6 flex items-center gap-2">
        <Settings className="h-5 w-5 text-primary" />
        <h2 className="text-lg font-semibold">Settings</h2>
      </div>

      <div className="space-y-6">
        {/* Sensitivity Slider */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="sensitivity" className="text-sm font-medium">
              Detection Sensitivity
            </Label>
            <span className="text-sm text-muted-foreground">{sensitivity}%</span>
          </div>
          <Slider
            id="sensitivity"
            value={[sensitivity]}
            onValueChange={([value]) => onSensitivityChange(value)}
            min={20}
            max={80}
            step={5}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">
            Higher sensitivity detects smaller posture deviations
          </p>
        </div>

        {/* Sound Toggle */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {soundEnabled ? (
              <Volume2 className="h-5 w-5 text-primary" />
            ) : (
              <VolumeX className="h-5 w-5 text-muted-foreground" />
            )}
            <div>
              <Label htmlFor="sound" className="text-sm font-medium">
                Sound Alerts
              </Label>
              <p className="text-xs text-muted-foreground">Play a tone when posture is bad</p>
            </div>
          </div>
          <Switch id="sound" checked={soundEnabled} onCheckedChange={onSoundToggle} />
        </div>

        {/* Notifications Toggle */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {notificationsEnabled ? (
              <Bell className="h-5 w-5 text-primary" />
            ) : (
              <BellOff className="h-5 w-5 text-muted-foreground" />
            )}
            <div>
              <Label htmlFor="notifications" className="text-sm font-medium">
                Browser Notifications
              </Label>
              <p className="text-xs text-muted-foreground">Get notified even when tab is hidden</p>
            </div>
          </div>
          {notificationPermission !== 'granted' ? (
            <Button
              variant="outline"
              size="sm"
              onClick={onRequestNotificationPermission}
              className="text-xs"
            >
              Enable
            </Button>
          ) : (
            <Switch
              id="notifications"
              checked={notificationsEnabled}
              onCheckedChange={onNotificationsToggle}
            />
          )}
        </div>

        {/* Reset Baseline Button */}
        <Button
          variant="outline"
          onClick={onResetBaseline}
          disabled={!isRunning}
          className="w-full"
        >
          <RotateCcw className="mr-2 h-4 w-4" />
          Reset Baseline Posture
        </Button>
        <p className="text-xs text-muted-foreground text-center">
          Sit in your ideal posture and click to recalibrate
        </p>
      </div>
    </motion.div>
  );
}
