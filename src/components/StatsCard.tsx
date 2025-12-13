import { motion } from 'framer-motion';
import { Activity, RotateCcw, Timer, Trophy } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface StatsCardProps {
    stats: {
        goodDuration: number;
        badDuration: number;
    };
    onReset: () => void;
}

export function StatsCard({ stats, onReset }: StatsCardProps) {
    const totalDuration = stats.goodDuration + stats.badDuration;
    const goodPercentage = totalDuration > 0
        ? Math.round((stats.goodDuration / totalDuration) * 100)
        : 0;

    const formatTime = (ms: number) => {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);

        if (hours > 0) {
            return `${hours}h ${minutes % 60}m`;
        }
        return `${minutes}m ${seconds % 60}s`;
    };

    const getScoreColor = (score: number) => {
        if (score >= 80) return 'text-status-good';
        if (score >= 50) return 'text-status-warning';
        return 'text-status-danger';
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="rounded-2xl border border-border bg-card p-6 shadow-sm"
        >
            <div className="mb-6 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Trophy className="h-5 w-5 text-primary" />
                    <h2 className="text-lg font-semibold">Session Stats</h2>
                </div>
                <Button variant="ghost" size="icon" onClick={onReset} title="Reset Stats">
                    <RotateCcw className="h-4 w-4 text-muted-foreground" />
                </Button>
            </div>

            <div className="space-y-6">
                {/* Health Score */}
                <div className="text-center">
                    <div className={`text-4xl font-bold ${getScoreColor(goodPercentage)}`}>
                        {goodPercentage}%
                    </div>
                    <p className="text-sm text-muted-foreground">Posture Score</p>
                </div>

                {/* Progress Bar */}
                <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
                    <motion.div
                        className="h-full bg-status-good"
                        initial={{ width: 0 }}
                        animate={{ width: `${goodPercentage}%` }}
                        transition={{ duration: 0.5 }}
                    />
                </div>

                {/* Details Grid */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-xl bg-secondary/50 p-3 text-center">
                        <div className="mb-1 flex justify-center">
                            <Timer className="h-4 w-4 text-status-good" />
                        </div>
                        <div className="font-semibold">{formatTime(stats.goodDuration)}</div>
                        <div className="text-xs text-muted-foreground">Focused</div>
                    </div>
                    <div className="rounded-xl bg-secondary/50 p-3 text-center">
                        <div className="mb-1 flex justify-center">
                            <Activity className="h-4 w-4 text-status-danger" />
                        </div>
                        <div className="font-semibold">{formatTime(stats.badDuration)}</div>
                        <div className="text-xs text-muted-foreground">Slouching</div>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}
