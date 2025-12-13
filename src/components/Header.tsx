import { motion } from 'framer-motion';
import { Activity } from 'lucide-react';

export function Header() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="border-b border-border bg-card/50 backdrop-blur-sm"
    >
      <div className="container mx-auto flex items-center justify-between px-4 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary">
            <Activity className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">PostureGuard</h1>
            <p className="text-xs text-muted-foreground">Real-time posture monitoring</p>
          </div>
        </div>
        
        <nav className="hidden items-center gap-4 sm:flex">
          <span className="rounded-full bg-status-good-bg px-3 py-1 text-xs font-medium text-status-good">
            Web-Based
          </span>
          <span className="rounded-full bg-secondary px-3 py-1 text-xs font-medium text-secondary-foreground">
            Privacy First
          </span>
        </nav>
      </div>
    </motion.header>
  );
}
