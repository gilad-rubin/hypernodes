/** @type {import('tailwindcss').Config} */
const path = require('path');

module.exports = {
  content: [
    // Scan the Python file that generates HTML with Tailwind classes
    // Absolute path to ensure it works from any directory
    path.join(__dirname, '../../src/hypernodes/viz/js/html_generator.py'),
  ],
  theme: {
    extend: {},
  },
  plugins: [],
  // Safelist only dynamically-generated classes that Tailwind can't detect
  // (Classes built via string interpolation like `border-${colors.border}-300`)
  safelist: [
    // Dynamic border colors for node types (slate, amber, indigo, cyan, fuchsia)
    'border-slate-300', 'border-slate-400', 'border-slate-500',
    'border-amber-300', 'border-amber-400', 'border-amber-500',
    'border-indigo-300', 'border-indigo-400', 'border-indigo-500',
    'border-cyan-300', 'border-cyan-400', 'border-cyan-500',
    'border-fuchsia-300', 'border-fuchsia-400', 'border-fuchsia-500',
    // Hover border variants
    'hover:border-slate-400', 'hover:border-slate-500',
    'hover:border-amber-400', 'hover:border-amber-500',
    'hover:border-indigo-400', 'hover:border-indigo-500',
    'hover:border-cyan-400', 'hover:border-cyan-500',
    'hover:border-fuchsia-400', 'hover:border-fuchsia-500',
    // Dynamic shadow colors
    'shadow-slate-200', 'shadow-amber-200', 'shadow-indigo-200', 'shadow-cyan-200', 'shadow-fuchsia-200',
    'hover:shadow-slate-200', 'hover:shadow-amber-200', 'hover:shadow-indigo-200', 'hover:shadow-cyan-200', 'hover:shadow-fuchsia-200',
    // Dynamic opacity border variants (for dark mode)
    'border-slate-500/40', 'border-slate-500/70',
    'border-amber-500/40', 'border-amber-500/70',
    'border-indigo-500/40', 'border-indigo-500/70',
    'border-cyan-500/40', 'border-cyan-500/70',
    'border-fuchsia-500/40', 'border-fuchsia-500/70',
    'hover:border-slate-500/70', 'hover:border-amber-500/70', 'hover:border-indigo-500/70', 'hover:border-cyan-500/70', 'hover:border-fuchsia-500/70',
    // Dynamic shadow opacity variants
    'hover:shadow-slate-500/20', 'hover:shadow-amber-500/20', 'hover:shadow-indigo-500/20', 'hover:shadow-cyan-500/20', 'hover:shadow-fuchsia-500/20',
  ],
}
