/// <reference types="vitest" />
import { defineConfig } from 'vite';
import path from 'path';
import react from '@vitejs/plugin-react-swc';

// https://vitejs.dev/config/
export default defineConfig({
  base: '',
  plugins: [react()],
  build: {
    outDir: '../../trulens/dashboard/components/record_viewer_otel/dist',
  },
  test: {
    environment: 'jsdom',
    include: ['**/*.test.[jt]s?(x)'],
    setupFiles: ['vitest.setup.ts'],
    globals: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
