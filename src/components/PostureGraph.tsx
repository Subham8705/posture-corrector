import {
    Area,
    AreaChart,
    Bar,
    BarChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
    CartesianGrid,
    Legend
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { usePostureHistory } from '@/hooks/usePostureHistory';
import { useMemo } from 'react';

interface PostureGraphProps {
    sessionHistory: Array<{ timestamp: number; score: number }>;
}

export function PostureGraph({ sessionHistory }: PostureGraphProps) {
    const { getWeeklySummary } = usePostureHistory();
    const weeklyStats = useMemo(() => getWeeklySummary(), [getWeeklySummary]);

    // Format session data for graph
    const sessionData = useMemo(() => {
        return sessionHistory.map(point => ({
            time: new Date(point.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
            score: point.score,
        }));
    }, [sessionHistory]);

    // Format weekly data (convert ms to minutes)
    const weeklyData = useMemo(() => {
        return weeklyStats.map(day => ({
            date: new Date(day.date).toLocaleDateString([], { weekday: 'short' }),
            Good: Math.round(day.goodDuration / 1000 / 60),
            Bad: Math.round(day.badDuration / 1000 / 60),
        }));
    }, [weeklyStats]);

    const gradientOffset = () => {
        const dataMax = Math.max(...sessionData.map((i) => i.score));
        const dataMin = Math.min(...sessionData.map((i) => i.score));

        if (dataMax <= 0) {
            return 0;
        }
        if (dataMin >= 0) {
            return 1;
        }

        return 0.5; // Default middle
    };

    const off = gradientOffset();

    return (
        <div className="grid gap-6 md:grid-cols-2">
            {/* Session Graph */}
            <Card>
                <CardHeader>
                    <CardTitle>Session Timeline</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="h-[300px] w-full">
                        {sessionData.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={sessionData}>
                                    <defs>
                                        <linearGradient id="splitColor" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                                            <stop offset="95%" stopColor="#ef4444" stopOpacity={0.8} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                                    <XAxis dataKey="time" hide />
                                    <YAxis domain={[0, 1]} hide />
                                    <Tooltip
                                        labelStyle={{ color: 'black' }}
                                        formatter={(value: number) => [value === 1 ? 'Good' : 'Bad', 'Posture']}
                                    />
                                    <Area
                                        type="step"
                                        dataKey="score"
                                        stroke="#000"
                                        fill="url(#splitColor)"
                                        animationDuration={500}
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="flex h-full items-center justify-center text-muted-foreground">
                                Start monitoring to see real-time data
                            </div>
                        )}
                    </div>
                </CardContent>
            </Card>

            {/* Weekly Graph */}
            <Card>
                <CardHeader>
                    <CardTitle>Weekly History (Minutes)</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={weeklyData}>
                                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                                <XAxis dataKey="date" />
                                <YAxis />
                                <Tooltip
                                    labelStyle={{ color: 'black' }}
                                    cursor={{ fill: 'transparent' }}
                                />
                                <Legend />
                                <Bar dataKey="Good" stackId="a" fill="#10b981" radius={[0, 0, 4, 4]} />
                                <Bar dataKey="Bad" stackId="a" fill="#ef4444" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
