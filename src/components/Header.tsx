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
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-background overflow-hidden ring-1 ring-border">
            <img
              src="https://res.cloudinary.com/dpa0sb1tm/image/upload/v1750759481/logobg_hu36yx.webp"
              alt="PosturePal Logo"
              className="h-full w-full object-cover"
            />
          </div>
          <div>
            <div className="flex items-baseline gap-2">
              <h1 className="text-xl font-bold text-foreground">PosturePal AI</h1>
              <span className="text-[10px] sm:text-xs text-muted-foreground font-medium">
                Made by Subham
              </span>
            </div>
            <p className="text-xs text-muted-foreground">AI-Powered Posture Coach</p>
          </div>
        </div>

        <nav className="hidden items-center gap-4 sm:flex">
          <span className="rounded-full bg-status-good-bg px-3 py-1 text-xs font-medium text-status-good">
            Web-Based
          </span>
          <span className="rounded-full bg-secondary px-3 py-1 text-xs font-medium text-secondary-foreground">
            Health First
          </span>
        </nav>
      </div>
    </motion.header>
  );
}
