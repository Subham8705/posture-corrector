import { useState, useEffect, useCallback } from 'react';

export interface DailyStats {
    date: string; // ISO date string YYYY-MM-DD
    goodDuration: number;
    badDuration: number;
}

const STORAGE_KEY = 'posture-history';

export function usePostureHistory() {
    const [weeklyStats, setWeeklyStats] = useState<DailyStats[]>([]);

    // Load history from local storage on mount
    useEffect(() => {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored) {
                const parsed = JSON.parse(stored);
                // Ensure we have an array
                if (Array.isArray(parsed)) {
                    // Sort by date just in case
                    const sorted = parsed.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
                    // Keep only last 30 days or so to prevent bloat, but for now just load all
                    setWeeklyStats(sorted);
                }
            }
        } catch (e) {
            console.error('Failed to load posture history:', e);
        }
    }, []);

    const saveDailyStats = useCallback((goodDuration: number, badDuration: number) => {
        const today = new Date().toISOString().split('T')[0];

        setWeeklyStats(prev => {
            const newStats = [...prev];
            const todayIndex = newStats.findIndex(s => s.date === today);

            if (todayIndex >= 0) {
                // Update existing entry
                newStats[todayIndex] = {
                    date: today,
                    goodDuration: newStats[todayIndex].goodDuration + goodDuration,
                    badDuration: newStats[todayIndex].badDuration + badDuration
                };
            } else {
                // Add new entry
                newStats.push({
                    date: today,
                    goodDuration,
                    badDuration
                });
            }

            // Save to local storage
            try {
                // Keep only last 30 days
                const recentStats = newStats.slice(-30);
                localStorage.setItem(STORAGE_KEY, JSON.stringify(recentStats));
                return recentStats;
            } catch (e) {
                console.error('Failed to save posture history:', e);
                return newStats;
            }
        });
    }, []);

    const getWeeklySummary = useCallback(() => {
        // Return last 7 days, filling in gaps if necessary
        const days = [];
        for (let i = 6; i >= 0; i--) {
            const d = new Date();
            d.setDate(d.getDate() - i);
            const dateStr = d.toISOString().split('T')[0];
            const found = weeklyStats.find(s => s.date === dateStr);
            days.push(found || { date: dateStr, goodDuration: 0, badDuration: 0 });
        }
        return days;
    }, [weeklyStats]);

    return {
        weeklyStats,
        saveDailyStats,
        getWeeklySummary
    };
}
