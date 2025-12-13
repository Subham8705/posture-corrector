import { motion } from 'framer-motion';
import { Lightbulb, Monitor, Armchair, Eye } from 'lucide-react';

const tips = [
  {
    icon: Monitor,
    title: 'Screen Position',
    description: 'Keep your screen at eye level, about an arm\'s length away.',
  },
  {
    icon: Armchair,
    title: 'Chair Setup',
    description: 'Sit with your back against the chair, feet flat on the floor.',
  },
  {
    icon: Eye,
    title: '20-20-20 Rule',
    description: 'Every 20 min, look at something 20 feet away for 20 seconds.',
  },
];

export function Tips() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="rounded-2xl border border-border bg-card p-6 shadow-sm"
    >
      <div className="mb-4 flex items-center gap-2">
        <Lightbulb className="h-5 w-5 text-status-warning" />
        <h2 className="text-lg font-semibold">Posture Tips</h2>
      </div>

      <div className="space-y-4">
        {tips.map((tip, index) => (
          <motion.div
            key={tip.title}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 + index * 0.1 }}
            className="flex gap-3"
          >
            <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg bg-secondary">
              <tip.icon className="h-4 w-4 text-secondary-foreground" />
            </div>
            <div>
              <p className="text-sm font-medium">{tip.title}</p>
              <p className="text-xs text-muted-foreground">{tip.description}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
