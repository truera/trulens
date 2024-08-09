import { defineConfig } from 'vite';
import path from 'path'
import react from '@vitejs/plugin-react-swc';

// https://vitejs.dev/config/
export default defineConfig({
  base: '',
  plugins: [react()],
  build: {
    outDir: '../../trulens/dashboard/react_components/record_viewer/dist',
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
